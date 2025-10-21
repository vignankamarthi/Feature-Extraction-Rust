// data_loader.rs - CSV data loading and parsing
//
// Handles loading multi-column CSV files where each column
// represents a different experimental condition

use std::path::{Path, PathBuf};
use std::fs::{self, File};
use csv::Reader;
use anyhow::Result;
use std::collections::HashMap;

use crate::types::{SignalType, Dataset, PainState};

/// Data loader for multi-column CSV files
///
/// Each CSV file has 48 columns, each representing a different
/// experimental condition (e.g., "6_BASELINE_3", "6_LOW_1")
#[derive(Debug, Clone)]
pub struct DataLoader {
    /// Base data directory
    pub data_dir: PathBuf,

    /// Current dataset being loaded
    pub dataset: Dataset,
}

impl DataLoader {
    /// Create new data loader
    pub fn new(data_dir: PathBuf, dataset: Dataset) -> Self {
        Self { data_dir, dataset }
    }

    /// Get all CSV files for a signal type
    ///
    /// Returns list of paths to CSV files
    pub fn get_files(&self, signal_type: SignalType) -> Result<Vec<PathBuf>> {
        let signal_dir = self.data_dir
            .join(self.dataset.as_str())
            .join(signal_type.as_str());

        if !signal_dir.exists() {
            anyhow::bail!("Directory not found: {:?}", signal_dir);
        }

        // Read directory and filter for CSV files
        let mut files: Vec<PathBuf> = fs::read_dir(&signal_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.to_lowercase() == "csv")
                    .unwrap_or(false)
            })
            .collect();

        // Sort files for consistent ordering
        files.sort();

        Ok(files)
    }

    /// Load a single CSV file with multiple columns
    ///
    /// Returns HashMap where key is column name and value is signal data
    pub fn load_file(&self, path: &Path) -> Result<HashMap<String, Vec<f64>>> {
        let file = File::open(path)?;
        let mut reader = Reader::from_reader(file);

        // Get headers (column names)
        let headers = reader.headers()?.clone();
        let column_names: Vec<String> = headers
            .iter()
            .map(|h| h.to_string())
            .collect();

        // Initialize storage for each column
        let mut data: HashMap<String, Vec<f64>> = HashMap::new();
        for name in &column_names {
            data.insert(name.clone(), Vec::new());
        }

        // Read all rows
        for result in reader.records() {
            let record = result?;

            // Parse each value and add to corresponding column
            for (idx, value_str) in record.iter().enumerate() {
                if let Some(column_name) = column_names.get(idx) {
                    // Parse as f64, use NaN for empty or invalid values
                    let value = value_str.parse::<f64>().unwrap_or(f64::NAN);
                    data.get_mut(column_name).unwrap().push(value);
                }
            }
        }

        Ok(data)
    }

    /// Load and process all files for a signal type
    ///
    /// Returns iterator over (column_name, signal_data) pairs
    pub fn load_all_signals(
        &self,
        signal_type: SignalType,
    ) -> Result<impl Iterator<Item = (String, PathBuf, String, Vec<f64>)>> {
        let files = self.get_files(signal_type)?;
        let mut all_signals = Vec::new();

        for file_path in files {
            let file_name = file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            match self.load_file(&file_path) {
                Ok(data) => {
                    // Add each column as a separate signal
                    for (column_name, signal_data) in data {
                        all_signals.push((
                            file_name.clone(),
                            file_path.clone(),
                            column_name,
                            signal_data,
                        ));
                    }
                }
                Err(e) => {
                    log::warn!("Failed to load {:?}: {}", file_path, e);
                }
            }
        }

        Ok(all_signals.into_iter())
    }

    /// Count total number of signals to process
    pub fn count_signals(&self) -> Result<usize> {
        let mut total = 0;

        for signal_type in SignalType::all() {
            let files = self.get_files(signal_type)?;

            // Each file has approximately 48 columns
            total += files.len() * 48;
        }

        Ok(total)
    }

    /// Get a sample of data for testing
    ///
    /// Loads first file of first available signal type
    pub fn get_sample_data(&self) -> Result<(String, Vec<f64>)> {
        for signal_type in SignalType::all() {
            if let Ok(files) = self.get_files(signal_type) {
                if let Some(first_file) = files.first() {
                    if let Ok(data) = self.load_file(first_file) {
                        // Return first column of first file
                        if let Some((name, signal)) = data.into_iter().next() {
                            return Ok((name, signal));
                        }
                    }
                }
            }
        }

        anyhow::bail!("No sample data found")
    }
}

/// Metadata extracted from column name
#[derive(Debug, Clone)]
pub struct SignalMetadata {
    pub subject_id: String,
    pub condition: String,
    pub trial: String,
    pub pain_state: PainState,
}

impl SignalMetadata {
    /// Parse metadata from column name
    ///
    /// Column names are like "6_BASELINE_3" or "6_LOW_1"
    pub fn from_column_name(name: &str) -> Self {
        let parts: Vec<&str> = name.split('_').collect();

        let subject_id = parts.get(0).unwrap_or(&"unknown").to_string();
        let condition = parts.get(1).unwrap_or(&"unknown").to_string();
        let trial = parts.get(2).unwrap_or(&"1").to_string();

        let pain_state = PainState::from_column_name(name);

        Self {
            subject_id,
            condition,
            trial,
            pain_state,
        }
    }
}

// ============================================================================
// Batch Loading for Parallel Processing
// ============================================================================

/// Batch of signals for parallel processing
#[derive(Debug, Clone)]
pub struct SignalBatch {
    pub signals: Vec<(String, Vec<f64>)>,  // (column_name, data)
    pub signal_type: SignalType,
    pub file_path: PathBuf,
}

/// Load signals in batches for parallel processing
pub fn load_in_batches(
    data_dir: &Path,
    dataset: Dataset,
    batch_size: usize,
) -> Result<Vec<SignalBatch>> {
    let loader = DataLoader::new(data_dir.to_path_buf(), dataset);
    let mut batches = Vec::new();

    for signal_type in SignalType::all() {
        let files = loader.get_files(signal_type)?;

        for file_path in files {
            match loader.load_file(&file_path) {
                Ok(data) => {
                    let signals: Vec<(String, Vec<f64>)> = data.into_iter().collect();

                    // Split into batches
                    for chunk in signals.chunks(batch_size) {
                        batches.push(SignalBatch {
                            signals: chunk.to_vec(),
                            signal_type,
                            file_path: file_path.clone(),
                        });
                    }
                }
                Err(e) => {
                    log::warn!("Failed to load {:?}: {}", file_path, e);
                }
            }
        }
    }

    Ok(batches)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_csv(dir: &Path, name: &str) -> Result<()> {
        let file_path = dir.join(name);
        let mut file = File::create(file_path)?;

        // Write header
        writeln!(file, "6_BASELINE_1,6_LOW_1,6_HIGH_1")?;

        // Write some data
        for i in 0..100 {
            writeln!(file, "{},{},{}", i as f64, i as f64 * 2.0, i as f64 * 3.0)?;
        }

        Ok(())
    }

    #[test]
    fn test_load_csv() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let data_dir = temp_dir.path().join("data/train/Bvp");
        fs::create_dir_all(&data_dir)?;

        create_test_csv(&data_dir, "test.csv")?;

        let loader = DataLoader::new(temp_dir.path().join("data"), Dataset::Train);
        let files = loader.get_files(SignalType::Bvp)?;

        assert_eq!(files.len(), 1);

        let data = loader.load_file(&files[0])?;
        assert_eq!(data.len(), 3);  // 3 columns

        // Check first column
        let baseline = data.get("6_BASELINE_1").unwrap();
        assert_eq!(baseline.len(), 100);
        assert_eq!(baseline[0], 0.0);

        Ok(())
    }

    #[test]
    fn test_metadata_parsing() {
        let meta = SignalMetadata::from_column_name("6_LOW_1");
        assert_eq!(meta.subject_id, "6");
        assert_eq!(meta.condition, "LOW");
        assert_eq!(meta.trial, "1");
        assert_eq!(meta.pain_state, PainState::Pain);

        let meta2 = SignalMetadata::from_column_name("12_BASELINE_3");
        assert_eq!(meta2.subject_id, "12");
        assert_eq!(meta2.condition, "BASELINE");
        assert_eq!(meta2.trial, "3");
        assert_eq!(meta2.pain_state, PainState::NoPain);
    }
}

// Key concepts demonstrated:
// 1. File I/O: Reading CSV files efficiently
// 2. HashMap: Key-value storage for column data
// 3. Iterator: Lazy evaluation for memory efficiency
// 4. Error handling: Result<T> throughout
// 5. Path handling: Cross-platform file paths
// 6. Pattern matching: Parsing column names
// 7. Chunking: Batching for parallel processing
// 8. Testing: Using tempfile for test isolation
// 9. Logging: warn! for non-fatal errors
// 10. Option chaining: Handling nullable values elegantly