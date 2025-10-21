// feature_extractor.rs - Main feature extraction orchestration
//
// This module coordinates the entire extraction pipeline using
// parallel processing with Rayon for massive speedups

use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::types::{
    ExtractionConfig, Features, FeatureSample, SignalType, Dataset,
    ExtractionProgress, LongFormatFeatures, LongFormatFeatureRow,
};
use crate::entropy::EntropyFeatures;
use crate::signal_processing::{SignalProcessor, ProcessedSignal};
use crate::data_loader::{DataLoader, SignalMetadata};

/// Main feature extractor that orchestrates the entire pipeline
///
/// This is like the main class that coordinates everything
pub struct FeatureExtractor {
    /// Extraction configuration
    config: ExtractionConfig,

    /// Signal processor for normalization
    processor: SignalProcessor,

    /// Data loader for CSV files
    loader: DataLoader,

    /// Progress tracking
    progress: Arc<MultiProgress>,

    /// Collected features (long format)
    features: Arc<Mutex<LongFormatFeatures>>,

    /// Error counter
    errors: Arc<Mutex<usize>>,
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(config: ExtractionConfig, progress: MultiProgress) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Create components
        let processor = SignalProcessor::with_frequency(100.0);
        let loader = DataLoader::new(
            PathBuf::from("data"),
            config.dataset,
        );

        Ok(Self {
            config,
            processor,
            loader,
            progress: Arc::new(progress),
            features: Arc::new(Mutex::new(LongFormatFeatures::new())),
            errors: Arc::new(Mutex::new(0)),
        })
    }

    /// Extract features from all configured signals
    ///
    /// This is the main entry point that processes everything
    pub fn extract_all(&mut self) -> Result<LongFormatFeatures> {
        let start_time = Instant::now();

        log::info!("Starting extraction for {:?} dataset", self.config.dataset);
        log::info!("Signal types: {:?}", self.config.signal_types);
        log::info!("Dimensions: {:?}", self.config.dimensions);
        log::info!("Taus: {:?}", self.config.taus);

        // Process each signal type
        for signal_type in &self.config.signal_types {
            self.process_signal_type(*signal_type)?;
        }

        // Get final results
        // Clone the Arc and then extract features from it
        let features_arc = self.features.clone();
        let features = features_arc.lock().unwrap().clone();

        let elapsed = start_time.elapsed();
        let num_rows = features.len();
        log::info!(
            "Extraction complete: {} rows in {:.2}s ({:.1} rows/s)",
            num_rows,
            elapsed.as_secs_f64(),
            num_rows as f64 / elapsed.as_secs_f64()
        );

        // Log error count
        let error_count = *self.errors.lock().unwrap();
        if error_count > 0 {
            log::warn!("Encountered {} errors during extraction", error_count);
        }

        Ok(features)
    }

    /// Process all files for a signal type
    fn process_signal_type(&self, signal_type: SignalType) -> Result<()> {
        // Get files for this signal type
        let files = self.loader.get_files(signal_type)?;

        if files.is_empty() {
            log::warn!("No files found for {:?}", signal_type);
            return Ok(());
        }

        log::info!("Processing {:?}: {} files", signal_type, files.len());

        // Create progress bar for this signal type
        let pb = self.progress.add(ProgressBar::new(files.len() as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template(&format!("  {}: {{bar:40}} {{pos}}/{{len}} {{msg}}", signal_type.as_str()))?
                .progress_chars("##-")
        );

        // Process files in parallel using Rayon
        // par_iter() automatically parallelizes the iteration
        files.par_iter().for_each(|file_path| {
            // Process single file
            match self.process_file(file_path, signal_type) {
                Ok(count) => {
                    pb.inc(1);
                    pb.set_message(format!("{} samples", count));
                }
                Err(e) => {
                    log::error!("Failed to process {:?}: {}", file_path, e);
                    *self.errors.lock().unwrap() += 1;
                    pb.inc(1);
                }
            }
        });

        pb.finish_with_message("Complete");

        Ok(())
    }

    /// Process a single CSV file with multiple columns
    fn process_file(&self, path: &PathBuf, signal_type: SignalType) -> Result<usize> {
        // Load all columns from file
        let data = self.loader.load_file(path)?;
        let mut row_count = 0;

        // Build file_name path for output
        let file_name = format!("data/{}/{}/{}",
            self.config.dataset.as_str(),
            signal_type.as_str(),
            path.file_name().unwrap_or_default().to_string_lossy()
        );

        // Process each column in parallel
        // Using par_iter() on the HashMap
        let results: Vec<_> = data
            .par_iter()
            .flat_map(|(column_name, signal)| {
                // Process single column/signal - returns Vec<LongFormatFeatureRow>
                self.process_column(column_name, signal, signal_type, &file_name)
            })
            .collect();

        // Add results to features collection
        let mut features = self.features.lock().unwrap();
        for row in results {
            features.add(row);
            row_count += 1;
        }

        Ok(row_count)
    }

    /// Process a single column (one experimental condition)
    /// Returns Vec of long-format rows (one per dimension × tau combination)
    fn process_column(
        &self,
        column_name: &str,
        raw_signal: &[f64],
        signal_type: SignalType,
        file_name: &str,
    ) -> Vec<LongFormatFeatureRow> {
        // Process signal (remove NaN, normalize)
        let processed = match self.processor.process_signal(raw_signal) {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Check if signal is valid
        if !processed.is_valid() {
            return Vec::new();
        }

        let mut rows = Vec::new();

        // Calculate entropy features for each dimension and tau
        // Each combination creates one row in long format
        for &dimension in &self.config.dimensions {
            for &tau in &self.config.taus {
                match EntropyFeatures::calculate(
                    processed.as_slice(),
                    dimension,
                    tau,
                ) {
                    Ok(entropy_features) => {
                        // Create long-format row
                        let row = LongFormatFeatureRow::new(
                            file_name.to_string(),
                            column_name.to_string(),
                            processed.valid_length,
                            dimension,
                            tau,
                            &entropy_features,
                            processed.nan_percentage,
                        );
                        rows.push(row);
                    }
                    Err(e) => {
                        log::debug!(
                            "Failed to calculate entropy for {} (d={}, t={}): {}",
                            column_name,
                            dimension,
                            tau,
                            e
                        );
                    }
                }
            }
        }

        rows
    }

    /// Extract features from a single signal (for testing)
    pub fn extract_single(
        &self,
        signal: &[f64],
        signal_type: SignalType,
    ) -> Result<FeatureSample> {
        // Process signal
        let processed = self.processor.process_signal(signal)
            .ok_or_else(|| anyhow::anyhow!("Signal processing failed"))?;

        // Create sample
        let mut sample = FeatureSample::new(
            "test_signal".to_string(),
            0,
            self.config.dataset,
            signal_type,
            processed.valid_length,
            processed.nan_percentage,
        );

        // Calculate features
        for &dimension in &self.config.dimensions {
            for &tau in &self.config.taus {
                let features = EntropyFeatures::calculate(
                    processed.as_slice(),
                    dimension,
                    tau,
                )?;

                sample.add_features(
                    dimension,
                    tau,
                    features.permutation_entropy,
                    features.complexity,
                    features.fisher_shannon,
                    features.fisher_information,
                    features.renyi_complexity,
                    features.renyi_pe,
                    features.tsallis_complexity,
                    features.tsallis_pe,
                );
            }
        }

        Ok(sample)
    }
}

// ============================================================================
// Batch Processing for Maximum Parallelism
// ============================================================================

/// Process signals in batches for maximum parallelism
///
/// This function demonstrates Rayon's power for parallel processing
pub fn batch_process_signals(
    signals: Vec<(String, Vec<f64>)>,
    config: &ExtractionConfig,
    signal_type: SignalType,
) -> Vec<FeatureSample> {
    let processor = SignalProcessor::new();

    // Process all signals in parallel
    // Rayon automatically manages thread pool
    signals
        .par_iter()  // Parallel iterator
        .filter_map(|(name, signal)| {
            // Process each signal independently
            let metadata = SignalMetadata::from_column_name(name);

            // Process signal
            let processed = processor.process_signal(signal)?;
            if !processed.is_valid() {
                return None;
            }

            // Create sample
            let mut sample = FeatureSample::new(
                name.clone(),
                metadata.pain_state.to_binaryclass(),
                config.dataset,
                signal_type,
                processed.valid_length,
                processed.nan_percentage,
            );

            // Calculate all features
            for &dimension in &config.dimensions {
                for &tau in &config.taus {
                    if let Ok(features) = EntropyFeatures::calculate(
                        processed.as_slice(),
                        dimension,
                        tau,
                    ) {
                        sample.add_features(
                            dimension,
                            tau,
                            features.permutation_entropy,
                            features.complexity,
                            features.fisher_shannon,
                            features.fisher_information,
                            features.renyi_complexity,
                            features.renyi_pe,
                            features.tsallis_complexity,
                            features.tsallis_pe,
                        );
                    }
                }
            }

            Some(sample)
        })
        .collect()  // Collect parallel results
}

// ============================================================================
// Performance Monitoring
// ============================================================================

/// Monitor extraction performance
pub struct PerformanceMonitor {
    start_time: Instant,
    samples_processed: usize,
    bytes_processed: usize,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            samples_processed: 0,
            bytes_processed: 0,
        }
    }

    /// Update with processed sample
    pub fn update(&mut self, sample_size: usize) {
        self.samples_processed += 1;
        self.bytes_processed += sample_size * std::mem::size_of::<f64>();
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> PerformanceMetrics {
        let elapsed = self.start_time.elapsed().as_secs_f64();

        PerformanceMetrics {
            samples_per_second: self.samples_processed as f64 / elapsed,
            mb_per_second: (self.bytes_processed as f64 / 1_000_000.0) / elapsed,
            total_samples: self.samples_processed,
            elapsed_seconds: elapsed,
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub samples_per_second: f64,
    pub mb_per_second: f64,
    pub total_samples: usize,
    pub elapsed_seconds: f64,
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Performance: {:.1} samples/s, {:.2} MB/s, {} total samples in {:.1}s",
            self.samples_per_second,
            self.mb_per_second,
            self.total_samples,
            self.elapsed_seconds
        )
    }
}

// Key Rust concepts in this module:
// 1. Arc<Mutex<T>>: Thread-safe shared state
// 2. Rayon par_iter(): Automatic parallelization
// 3. filter_map(): Combine filtering and transformation
// 4. Result<T> propagation: Error handling throughout
// 5. Progress bars: Real-time feedback
// 6. Parallel collection: Gathering results from parallel iteration
// 7. Performance monitoring: Tracking throughput
// 8. Zero-copy where possible: Using references
// 9. Lock management: Minimizing lock contention
// 10. Fearless concurrency: Compiler ensures thread safety