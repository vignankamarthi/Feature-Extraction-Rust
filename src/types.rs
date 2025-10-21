// types.rs - Custom types and data structures
//
// In Rust, we define our data structures using structs and enums
// This is like TypeScript interfaces or Python dataclasses

use serde::{Deserialize, Serialize};  // For JSON/CSV serialization
use std::path::{Path, PathBuf};
use anyhow::Result;
use std::fs::File;
use csv::{Reader, Writer};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Enums - Like TypeScript discriminated unions or Python Enums
// ============================================================================

/// Signal types we can process
///
/// In Rust, enums can have associated data (unlike Python/Java)
/// This is like TypeScript's discriminated unions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SignalType {
    Bvp,   // Blood Volume Pulse
    Eda,   // Electrodermal Activity
    Resp,  // Respiration
    SpO2,  // Blood Oxygen Saturation
}

impl SignalType {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SignalType::Bvp => "Bvp",
            SignalType::Eda => "Eda",
            SignalType::Resp => "Resp",
            SignalType::SpO2 => "SpO2",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bvp" => Some(SignalType::Bvp),
            "eda" => Some(SignalType::Eda),
            "resp" => Some(SignalType::Resp),
            "spo2" => Some(SignalType::SpO2),
            _ => None,
        }
    }

    /// Get all signal types
    pub fn all() -> Vec<SignalType> {
        vec![
            SignalType::Bvp,
            SignalType::Eda,
            SignalType::Resp,
            SignalType::SpO2,
        ]
    }
}

/// Dataset split types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Dataset {
    Train,
    Validation,
    Test,
}

impl Dataset {
    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Dataset::Train => "train",
            Dataset::Validation => "validation",
            Dataset::Test => "test",
        }
    }

    /// Get directory path for this dataset
    pub fn directory(&self) -> PathBuf {
        PathBuf::from("data").join(self.as_str())
    }
}

/// Pain state labels (4-class: baseline, low, high, rest)
///
/// Matches Python notebook label extraction exactly
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PainState {
    Baseline,
    Low,
    High,
    Rest,
    Unknown,
}

impl PainState {
    /// Extract from column name (matches Python LabelExtractor)
    pub fn from_column_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        if name_lower.contains("baseline") {
            PainState::Baseline
        } else if name_lower.contains("high") {
            PainState::High
        } else if name_lower.contains("low") {
            PainState::Low
        } else if name_lower.contains("rest") {
            PainState::Rest
        } else {
            PainState::Unknown
        }
    }

    /// Convert to string (state column)
    pub fn as_str(&self) -> &'static str {
        match self {
            PainState::Baseline => "baseline",
            PainState::Low => "low",
            PainState::High => "high",
            PainState::Rest => "rest",
            PainState::Unknown => "unknown",
        }
    }

    /// Convert to binary class (binaryclass column)
    /// baseline=0, low=1, high=2, rest=3, unknown=-1
    pub fn to_binaryclass(&self) -> i32 {
        match self {
            PainState::Baseline => 0,
            PainState::Low => 1,
            PainState::High => 2,
            PainState::Rest => 3,
            PainState::Unknown => -1,
        }
    }
}

// ============================================================================
// Configuration Structures
// ============================================================================

/// Main extraction configuration
///
/// This struct holds all parameters for feature extraction
/// Like a config object in TypeScript or dataclass in Python
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub dataset: Dataset,
    pub signal_types: Vec<SignalType>,
    pub nan_threshold: f64,      // Maximum % of NaN values allowed
    pub dimensions: Vec<usize>,  // Embedding dimensions
    pub taus: Vec<usize>,        // Time delays
    pub output_dir: PathBuf,     // Output directory for results
}

impl ExtractionConfig {
    /// Create default configuration
    pub fn default() -> Self {
        Self {
            dataset: Dataset::Train,
            signal_types: SignalType::all(),
            nan_threshold: 85.0,  // Allow up to 85% NaN (for pain segments)
            dimensions: vec![3, 4, 5, 6, 7],
            taus: vec![1, 2, 3],
            output_dir: PathBuf::from("results"),
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.dimensions.is_empty() {
            anyhow::bail!("No dimensions specified");
        }

        if self.taus.is_empty() {
            anyhow::bail!("No time delays specified");
        }

        if self.nan_threshold < 0.0 || self.nan_threshold > 100.0 {
            anyhow::bail!("NaN threshold must be between 0 and 100");
        }

        Ok(())
    }

    /// Get total number of features per sample
    pub fn num_features(&self) -> usize {
        // 8 entropy measures × dimensions × taus
        8 * self.dimensions.len() * self.taus.len()
    }
}

// ============================================================================
// Data Structures for Features
// ============================================================================

/// A single sample with extracted features
///
/// This is like a row in our output CSV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSample {
    // Metadata
    pub file_id: String,         // Column name (e.g., "6_LOW_1")
    pub label: i32,              // Binary label (0 or 1)
    pub dataset: String,         // train/validation/test
    pub signal_type: String,     // Bvp/Eda/Resp/SpO2
    pub signal_length: usize,    // Number of valid data points
    pub nan_percentage: f64,     // Percentage of NaN values

    // Features - stored as HashMap for flexibility
    #[serde(flatten)]  // Flatten into parent object when serializing
    pub features: HashMap<String, f64>,
}

impl FeatureSample {
    /// Create new sample
    pub fn new(
        file_id: String,
        label: i32,
        dataset: Dataset,
        signal_type: SignalType,
        signal_length: usize,
        nan_percentage: f64,
    ) -> Self {
        Self {
            file_id,
            label,
            dataset: dataset.as_str().to_string(),
            signal_type: signal_type.as_str().to_string(),
            signal_length,
            nan_percentage,
            features: HashMap::new(),
        }
    }

    /// Add entropy features for a specific dimension and tau
    pub fn add_features(
        &mut self,
        dimension: usize,
        tau: usize,
        pe: f64,
        complexity: f64,
        fisher_s: f64,
        fisher_i: f64,
        renyi_c: f64,
        renyi_pe: f64,
        tsallis_c: f64,
        tsallis_pe: f64,
    ) {
        // Create feature names following Python convention
        self.features.insert(format!("PE_d{}_t{}", dimension, tau), pe);
        self.features.insert(format!("C_d{}_t{}", dimension, tau), complexity);
        self.features.insert(format!("Fisher_S_d{}_t{}", dimension, tau), fisher_s);
        self.features.insert(format!("Fisher_I_d{}_t{}", dimension, tau), fisher_i);
        self.features.insert(format!("Renyi_C_d{}_t{}", dimension, tau), renyi_c);
        self.features.insert(format!("Renyi_PE_d{}_t{}", dimension, tau), renyi_pe);
        self.features.insert(format!("Tsallis_C_d{}_t{}", dimension, tau), tsallis_c);
        self.features.insert(format!("Tsallis_PE_d{}_t{}", dimension, tau), tsallis_pe);
    }

    /// Get file ID
    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    /// Get features as iterator
    pub fn features(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.features.iter()
    }
}

// ============================================================================
// Long-Format Feature Row (Notebook-Aligned)
// ============================================================================

/// Long-format feature row (16 columns - matches Python notebook exactly)
///
/// Each row represents one signal × dimension × tau combination.
/// For a signal with 5 dimensions (3-7) and 3 taus (1-3), this generates 15 rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongFormatFeatureRow {
    pub file_name: String,        // Full path: "data/train/Bvp/1.csv"
    pub signal: String,           // Column name: "1_Baseline_1"
    pub signallength: usize,      // Valid samples after NaN removal
    pub pe: f64,                  // Permutation Entropy
    pub comp: f64,                // Statistical Complexity
    pub fisher_shannon: f64,      // Fisher-Shannon Entropy
    pub fisher_info: f64,         // Fisher Information
    pub renyipe: f64,             // Renyi Permutation Entropy
    pub renyicomp: f64,           // Renyi Complexity
    pub tsallispe: f64,           // Tsallis Permutation Entropy
    pub tsalliscomp: f64,         // Tsallis Complexity
    pub dimension: usize,         // Embedding dimension (3-7)
    pub tau: usize,               // Time delay (1-3)
    pub state: String,            // State label: "baseline", "low", "high", "rest"
    pub binaryclass: i32,         // Numeric encoding: 0, 1, 2, 3
    pub nan_percentage: f64,      // Data quality metric
}

impl LongFormatFeatureRow {
    /// Create new long-format row from entropy features
    pub fn new(
        file_name: String,
        signal: String,
        signallength: usize,
        dimension: usize,
        tau: usize,
        entropy_features: &crate::entropy::EntropyFeatures,
        nan_percentage: f64,
    ) -> Self {
        // Extract state from signal name
        let pain_state = PainState::from_column_name(&signal);

        Self {
            file_name,
            signal,
            signallength,
            pe: entropy_features.permutation_entropy,
            comp: entropy_features.complexity,
            fisher_shannon: entropy_features.fisher_shannon,
            fisher_info: entropy_features.fisher_information,
            renyipe: entropy_features.renyi_pe,
            renyicomp: entropy_features.renyi_complexity,
            tsallispe: entropy_features.tsallis_pe,
            tsalliscomp: entropy_features.tsallis_complexity,
            dimension,
            tau,
            state: pain_state.as_str().to_string(),
            binaryclass: pain_state.to_binaryclass(),
            nan_percentage,
        }
    }
}

/// Collection of long-format feature rows
///
/// Represents complete feature dataset in notebook-aligned long format
#[derive(Clone)]
pub struct LongFormatFeatures {
    pub rows: Vec<LongFormatFeatureRow>,
}

impl LongFormatFeatures {
    /// Create empty collection
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
        }
    }

    /// Add a row
    pub fn add(&mut self, row: LongFormatFeatureRow) {
        self.rows.push(row);
    }

    /// Get number of rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Save to CSV file (notebook-aligned format)
    pub fn save_to_csv(&self, path: &Path) -> Result<()> {
        if self.rows.is_empty() {
            anyhow::bail!("No features to save");
        }

        let file = File::create(path)?;
        let mut writer = Writer::from_writer(file);

        // Write header (exact column order from notebook)
        let header = vec![
            "file_name", "signal", "signallength",
            "pe", "comp", "fisher_shannon", "fisher_info",
            "renyipe", "renyicomp", "tsallispe", "tsalliscomp",
            "dimension", "tau", "state", "binaryclass", "nan_percentage"
        ];
        writer.write_record(&header)?;

        // Write data rows
        for row in &self.rows {
            let record = vec![
                row.file_name.clone(),
                row.signal.clone(),
                row.signallength.to_string(),
                format!("{:.6}", row.pe),
                format!("{:.6}", row.comp),
                format!("{:.6}", row.fisher_shannon),
                format!("{:.6}", row.fisher_info),
                format!("{:.6}", row.renyipe),
                format!("{:.6}", row.renyicomp),
                format!("{:.6}", row.tsallispe),
                format!("{:.6}", row.tsalliscomp),
                row.dimension.to_string(),
                row.tau.to_string(),
                row.state.clone(),
                row.binaryclass.to_string(),
                format!("{:.2}", row.nan_percentage),
            ];
            writer.write_record(&record)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from CSV file
    pub fn load_from_csv(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = Reader::from_reader(file);

        let mut features = LongFormatFeatures::new();

        for result in reader.records() {
            let record = result?;

            let row = LongFormatFeatureRow {
                file_name: record.get(0).unwrap_or("").to_string(),
                signal: record.get(1).unwrap_or("").to_string(),
                signallength: record.get(2).unwrap_or("0").parse()?,
                pe: record.get(3).unwrap_or("0").parse()?,
                comp: record.get(4).unwrap_or("0").parse()?,
                fisher_shannon: record.get(5).unwrap_or("0").parse()?,
                fisher_info: record.get(6).unwrap_or("0").parse()?,
                renyipe: record.get(7).unwrap_or("0").parse()?,
                renyicomp: record.get(8).unwrap_or("0").parse()?,
                tsallispe: record.get(9).unwrap_or("0").parse()?,
                tsalliscomp: record.get(10).unwrap_or("0").parse()?,
                dimension: record.get(11).unwrap_or("0").parse()?,
                tau: record.get(12).unwrap_or("0").parse()?,
                state: record.get(13).unwrap_or("").to_string(),
                binaryclass: record.get(14).unwrap_or("0").parse()?,
                nan_percentage: record.get(15).unwrap_or("0").parse()?,
            };

            features.add(row);
        }

        Ok(features)
    }

    /// Get unique datasets from file paths
    pub fn datasets(&self) -> Vec<String> {
        let mut datasets = HashSet::new();
        for row in &self.rows {
            // Extract dataset from "data/train/..." -> "train"
            if let Some(parts) = row.file_name.split('/').nth(1) {
                datasets.insert(parts.to_string());
            }
        }
        datasets.into_iter().collect()
    }

    /// Get unique signal types from file paths
    pub fn signal_types(&self) -> Vec<String> {
        let mut signal_types = HashSet::new();
        for row in &self.rows {
            // Extract signal type from "data/train/Bvp/..." -> "Bvp"
            if let Some(parts) = row.file_name.split('/').nth(2) {
                signal_types.insert(parts.to_string());
            }
        }
        signal_types.into_iter().collect()
    }

    /// Iterator over rows
    pub fn iter(&self) -> std::slice::Iter<LongFormatFeatureRow> {
        self.rows.iter()
    }
}

/// Collection of feature samples
///
/// This represents our complete feature dataset
#[derive(Clone)]
pub struct Features {
    pub samples: Vec<FeatureSample>,
}

impl Features {
    /// Create empty collection
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Add a sample
    pub fn add(&mut self, sample: FeatureSample) {
        self.samples.push(sample);
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get number of features per sample
    pub fn num_features(&self) -> usize {
        if let Some(first) = self.samples.first() {
            first.features.len()
        } else {
            0
        }
    }

    /// Count NaN values in features
    pub fn count_nans(&self) -> usize {
        let mut count = 0;
        for sample in &self.samples {
            for value in sample.features.values() {
                if value.is_nan() {
                    count += 1;
                }
            }
        }
        count
    }

    /// Save to CSV file
    ///
    /// This is like pandas.DataFrame.to_csv()
    pub fn save_to_csv(&self, path: &Path) -> Result<()> {
        if self.samples.is_empty() {
            anyhow::bail!("No features to save");
        }

        // Create CSV writer
        let file = File::create(path)?;
        let mut writer = Writer::from_writer(file);

        // Get all feature keys in sorted order
        let mut feature_keys: Vec<String> = self.samples[0]
            .features
            .keys()
            .cloned()
            .collect();
        feature_keys.sort();

        // Write header
        let mut header = vec![
            "file_id".to_string(),
            "label".to_string(),
            "dataset".to_string(),
            "signal_type".to_string(),
            "signal_length".to_string(),
            "nan_percentage".to_string(),
        ];
        header.extend(feature_keys.clone());
        writer.write_record(&header)?;

        // Write data rows
        for sample in &self.samples {
            let mut row = vec![
                sample.file_id.clone(),
                sample.label.to_string(),
                sample.dataset.clone(),
                sample.signal_type.clone(),
                sample.signal_length.to_string(),
                format!("{:.2}", sample.nan_percentage),
            ];

            // Add feature values in same order as header
            for key in &feature_keys {
                let value = sample.features.get(key).unwrap_or(&f64::NAN);
                row.push(format!("{:.6}", value));
            }

            writer.write_record(&row)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load from CSV file
    ///
    /// This is like pandas.read_csv()
    pub fn load_from_csv(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = Reader::from_reader(file);

        let mut features = Features::new();

        // Read header
        let headers = reader.headers()?.clone();

        // Process each row
        for result in reader.records() {
            let record = result?;

            // Parse metadata
            let file_id = record.get(0).unwrap_or("").to_string();
            let label: i32 = record.get(1).unwrap_or("0").parse()?;
            let dataset = record.get(2).unwrap_or("").to_string();
            let signal_type = record.get(3).unwrap_or("").to_string();
            let signal_length: usize = record.get(4).unwrap_or("0").parse()?;
            let nan_percentage: f64 = record.get(5).unwrap_or("0").parse()?;

            let mut sample = FeatureSample {
                file_id,
                label,
                dataset,
                signal_type,
                signal_length,
                nan_percentage,
                features: HashMap::new(),
            };

            // Parse features (columns 6 onwards)
            for i in 6..record.len() {
                if let Some(key) = headers.get(i) {
                    if let Some(value_str) = record.get(i) {
                        if let Ok(value) = value_str.parse::<f64>() {
                            sample.features.insert(key.to_string(), value);
                        }
                    }
                }
            }

            features.add(sample);
        }

        Ok(features)
    }

    /// Filter by signal type
    pub fn filter_by_signal(&self, signal_type: SignalType) -> Vec<&FeatureSample> {
        self.samples
            .iter()
            .filter(|s| s.signal_type == signal_type.as_str())
            .collect()
    }

    /// Filter by dataset
    pub fn filter_by_dataset(&self, dataset: Dataset) -> Vec<&FeatureSample> {
        self.samples
            .iter()
            .filter(|s| s.dataset == dataset.as_str())
            .collect()
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("Total samples: {}\n", self.len()));
        summary.push_str(&format!("Features per sample: {}\n", self.num_features()));
        summary.push_str(&format!("NaN values: {}\n", self.count_nans()));

        // Count by dataset
        for dataset in &[Dataset::Train, Dataset::Validation, Dataset::Test] {
            let count = self.filter_by_dataset(*dataset).len();
            summary.push_str(&format!("  {}: {} samples\n", dataset.as_str(), count));
        }

        // Count by signal type
        for signal_type in SignalType::all() {
            let count = self.filter_by_signal(signal_type).len();
            summary.push_str(&format!("  {}: {} samples\n", signal_type.as_str(), count));
        }

        summary
    }

    /// Iterator over samples
    pub fn iter(&self) -> std::slice::Iter<FeatureSample> {
        self.samples.iter()
    }

    /// Get feature count
    pub fn feature_count(&self) -> usize {
        self.num_features()
    }

    /// Get unique signal types
    pub fn signal_types(&self) -> Vec<String> {
        let mut types = std::collections::HashSet::new();
        for sample in &self.samples {
            types.insert(sample.signal_type.clone());
        }
        types.into_iter().collect()
    }

    /// Get unique datasets
    pub fn datasets(&self) -> Vec<String> {
        let mut datasets = std::collections::HashSet::new();
        for sample in &self.samples {
            datasets.insert(sample.dataset.clone());
        }
        datasets.into_iter().collect()
    }
}

// ============================================================================
// Progress Tracking
// ============================================================================

/// Progress information for extraction
#[derive(Debug, Clone)]
pub struct ExtractionProgress {
    pub current_file: usize,
    pub total_files: usize,
    pub current_signal: SignalType,
    pub samples_extracted: usize,
    pub errors_encountered: usize,
}

impl ExtractionProgress {
    /// Create new progress tracker
    pub fn new(total_files: usize) -> Self {
        Self {
            current_file: 0,
            total_files,
            current_signal: SignalType::Bvp,
            samples_extracted: 0,
            errors_encountered: 0,
        }
    }

    /// Update progress
    pub fn update(&mut self, file_index: usize, samples: usize) {
        self.current_file = file_index;
        self.samples_extracted += samples;
    }

    /// Get percentage complete
    pub fn percentage(&self) -> f64 {
        if self.total_files == 0 {
            0.0
        } else {
            (self.current_file as f64 / self.total_files as f64) * 100.0
        }
    }
}

// Key Rust concepts in this file:
// 1. Enums with methods: More powerful than traditional enums
// 2. Structs: Data containers with associated functions
// 3. Traits: Serialize/Deserialize for automatic conversion
// 4. Option<T>: Rust's way of handling nullable values
// 5. Result<T, E>: Error handling without exceptions
// 6. impl blocks: Adding methods to types
// 7. Pattern matching: Exhaustive case handling
// 8. Ownership: Structs own their data
// 9. References: & for borrowing without ownership
// 10. Type safety: Compiler ensures correct types everywhere