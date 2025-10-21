// signal_processing.rs - Signal preprocessing and normalization
//
// This module handles all signal preprocessing tasks
// Similar to scipy.signal and sklearn.preprocessing in Python

use ndarray::{Array1, ArrayView1};
use anyhow::Result;
use statrs::statistics::Statistics;  // For mean, std dev, etc.

/// Signal processor for preprocessing physiological signals
///
/// This struct holds configuration for signal processing
/// Like a class in Python/Java but with separated data and methods
#[derive(Debug, Clone)]
pub struct SignalProcessor {
    /// Sampling frequency in Hz
    pub sampling_frequency: f64,

    /// Whether to remove linear trend
    pub detrend: bool,

    /// Minimum valid signal length after NaN removal
    pub min_signal_length: usize,
}

impl SignalProcessor {
    /// Create new signal processor with default settings
    ///
    /// Default is like __init__ in Python or constructor in Java
    pub fn new() -> Self {
        Self {
            sampling_frequency: 100.0,  // 100 Hz default for physiological signals
            detrend: false,              // Don't detrend by default
            min_signal_length: 100,      // Need at least 100 valid points
        }
    }

    /// Create with custom sampling frequency
    pub fn with_frequency(sampling_frequency: f64) -> Self {
        Self {
            sampling_frequency,
            ..Self::new()  // .. means "use other fields from new()"
        }
    }

    /// Z-score normalization (standardization)
    ///
    /// Transforms signal to have mean=0 and std=1
    /// This is like sklearn.preprocessing.StandardScaler
    ///
    /// # Arguments
    /// * `signal` - Input signal array
    ///
    /// # Returns
    /// Normalized signal with mean=0, std=1
    pub fn z_score_normalize(&self, signal: &[f64]) -> Vec<f64> {
        // Handle edge cases
        if signal.is_empty() {
            return Vec::new();
        }

        // Calculate mean
        // In Rust, we need to be explicit about type conversions
        let mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;

        // Calculate standard deviation
        // This is the sample standard deviation (n-1 denominator)
        let variance: f64 = signal
            .iter()
            .map(|&x| (x - mean).powi(2))  // powi(2) is power of integer
            .sum::<f64>() / (signal.len() - 1) as f64;

        let std_dev = variance.sqrt();

        // Handle constant signal (std_dev = 0)
        if std_dev < 1e-10 {
            // Return zeros if signal is constant
            return vec![0.0; signal.len()];
        }

        // Apply z-score normalization
        // map() transforms each element (like list comprehension in Python)
        signal
            .iter()
            .map(|&x| (x - mean) / std_dev)
            .collect()  // collect() turns iterator into Vec
    }

    /// Remove NaN values from signal
    ///
    /// Returns clean signal and percentage of NaN values
    /// Like pandas.dropna() but returns both clean data and NaN info
    pub fn remove_nans(&self, signal: &[f64]) -> (Vec<f64>, f64) {
        let original_len = signal.len();

        // Filter out NaN values
        // filter() keeps only elements where predicate is true
        let clean_signal: Vec<f64> = signal
            .iter()
            .filter(|&&x| !x.is_nan())  // Keep non-NaN values
            .cloned()  // Clone the values (copy from reference)
            .collect();

        let nan_count = original_len - clean_signal.len();
        let nan_percentage = (nan_count as f64 / original_len as f64) * 100.0;

        (clean_signal, nan_percentage)
    }

    /// Validate signal for processing
    ///
    /// Checks if signal meets minimum requirements
    pub fn validate_signal(&self, signal: &[f64]) -> Result<()> {
        if signal.len() < self.min_signal_length {
            anyhow::bail!(
                "Signal too short: {} < {} required",
                signal.len(),
                self.min_signal_length
            );
        }

        // Check if all values are finite
        let non_finite_count = signal.iter().filter(|&&x| !x.is_finite()).count();
        if non_finite_count > 0 {
            anyhow::bail!("Signal contains {} non-finite values", non_finite_count);
        }

        Ok(())
    }

    /// Process a raw signal through complete pipeline
    ///
    /// This is the main processing function that combines all steps
    /// Returns None if signal doesn't meet requirements
    pub fn process_signal(&self, raw_signal: &[f64]) -> Option<ProcessedSignal> {
        // Remove NaN values
        let (clean_signal, nan_percentage) = self.remove_nans(raw_signal);

        // Check if too many NaNs
        if nan_percentage > 85.0 {
            return None;  // Too many NaNs (>85%)
        }

        // Check minimum length
        if clean_signal.len() < self.min_signal_length {
            return None;  // Too short after NaN removal
        }

        // Apply z-score normalization
        let normalized = self.z_score_normalize(&clean_signal);

        // Optionally detrend
        let processed = if self.detrend {
            self.detrend_signal(&normalized)
        } else {
            normalized
        };

        Some(ProcessedSignal {
            data: processed,
            original_length: raw_signal.len(),
            valid_length: clean_signal.len(),
            nan_percentage,
        })
    }

    /// Remove linear trend from signal
    ///
    /// Like scipy.signal.detrend()
    fn detrend_signal(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len() as f64;

        // Create time vector
        let t: Vec<f64> = (0..signal.len())
            .map(|i| i as f64)
            .collect();

        // Calculate linear regression coefficients
        // This is least squares fitting: y = mx + b

        let t_mean = t.iter().sum::<f64>() / n;
        let y_mean = signal.iter().sum::<f64>() / n;

        // Calculate slope (m)
        let numerator: f64 = t.iter()
            .zip(signal.iter())
            .map(|(ti, yi)| (ti - t_mean) * (yi - y_mean))
            .sum();

        let denominator: f64 = t.iter()
            .map(|ti| (ti - t_mean).powi(2))
            .sum();

        let slope = numerator / denominator;
        let intercept = y_mean - slope * t_mean;

        // Remove trend
        signal
            .iter()
            .enumerate()
            .map(|(i, &y)| y - (slope * i as f64 + intercept))
            .collect()
    }

    /// Apply moving average filter
    ///
    /// Like pandas.rolling().mean()
    pub fn moving_average(&self, signal: &[f64], window_size: usize) -> Vec<f64> {
        if window_size == 0 || window_size > signal.len() {
            return signal.to_vec();
        }

        let mut result = Vec::with_capacity(signal.len());

        // For each position in signal
        for i in 0..signal.len() {
            // Determine window bounds
            let start = if i >= window_size / 2 {
                i - window_size / 2
            } else {
                0
            };

            let end = if i + window_size / 2 < signal.len() {
                i + window_size / 2 + 1
            } else {
                signal.len()
            };

            // Calculate mean of window
            let window_sum: f64 = signal[start..end].iter().sum();
            let window_mean = window_sum / (end - start) as f64;

            result.push(window_mean);
        }

        result
    }

    /// Detect and remove outliers using z-score method
    ///
    /// Points with |z-score| > threshold are considered outliers
    pub fn remove_outliers(&self, signal: &[f64], z_threshold: f64) -> Vec<f64> {
        let normalized = self.z_score_normalize(signal);

        // Mark outliers
        let outlier_mask: Vec<bool> = normalized
            .iter()
            .map(|&z| z.abs() > z_threshold)
            .collect();

        // Calculate mean of non-outliers for replacement
        let non_outlier_mean = signal
            .iter()
            .zip(outlier_mask.iter())
            .filter(|(_, &is_outlier)| !is_outlier)
            .map(|(&value, _)| value)
            .sum::<f64>()
            / outlier_mask.iter().filter(|&&x| !x).count() as f64;

        // Replace outliers with mean
        signal
            .iter()
            .zip(outlier_mask.iter())
            .map(|(&value, &is_outlier)| {
                if is_outlier {
                    non_outlier_mean
                } else {
                    value
                }
            })
            .collect()
    }

    /// Calculate signal quality metrics
    ///
    /// Returns various quality indicators
    pub fn calculate_quality_metrics(&self, signal: &[f64]) -> SignalQuality {
        let (clean_signal, nan_percentage) = self.remove_nans(signal);

        // Calculate basic statistics
        let mean = if !clean_signal.is_empty() {
            clean_signal.iter().sum::<f64>() / clean_signal.len() as f64
        } else {
            f64::NAN
        };

        let variance = if clean_signal.len() > 1 {
            clean_signal
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (clean_signal.len() - 1) as f64
        } else {
            f64::NAN
        };

        let std_dev = variance.sqrt();

        // Signal-to-noise ratio estimate
        // SNR = mean^2 / variance (simplified version)
        let snr = if variance > 0.0 {
            mean.powi(2) / variance
        } else {
            f64::INFINITY
        };

        SignalQuality {
            mean,
            std_dev,
            nan_percentage,
            valid_samples: clean_signal.len(),
            total_samples: signal.len(),
            snr,
        }
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/// Processed signal with metadata
#[derive(Debug, Clone)]
pub struct ProcessedSignal {
    /// Processed signal data
    pub data: Vec<f64>,

    /// Original signal length (before NaN removal)
    pub original_length: usize,

    /// Valid signal length (after NaN removal)
    pub valid_length: usize,

    /// Percentage of NaN values
    pub nan_percentage: f64,
}

impl ProcessedSignal {
    /// Check if signal is valid for feature extraction
    pub fn is_valid(&self) -> bool {
        self.valid_length >= 100 && self.nan_percentage <= 85.0
    }

    /// Get signal as slice
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}

/// Signal quality metrics
#[derive(Debug, Clone)]
pub struct SignalQuality {
    pub mean: f64,
    pub std_dev: f64,
    pub nan_percentage: f64,
    pub valid_samples: usize,
    pub total_samples: usize,
    pub snr: f64,  // Signal-to-noise ratio
}

impl SignalQuality {
    /// Check if signal quality is acceptable
    pub fn is_acceptable(&self) -> bool {
        self.nan_percentage < 85.0
            && self.valid_samples >= 100
            && self.std_dev > 0.0
            && self.snr.is_finite()
    }

    /// Get quality score (0-1, higher is better)
    pub fn quality_score(&self) -> f64 {
        let nan_score = (100.0 - self.nan_percentage) / 100.0;
        let length_score = (self.valid_samples as f64 / 1000.0).min(1.0);
        let snr_score = (self.snr / 10.0).min(1.0).max(0.0);

        // Weighted average
        (nan_score * 0.5 + length_score * 0.3 + snr_score * 0.2)
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Resample signal to new length
///
/// Simple linear interpolation resampling
/// Like scipy.signal.resample but simpler
pub fn resample(signal: &[f64], new_length: usize) -> Vec<f64> {
    if signal.is_empty() || new_length == 0 {
        return Vec::new();
    }

    if signal.len() == new_length {
        return signal.to_vec();
    }

    let mut resampled = Vec::with_capacity(new_length);
    let ratio = (signal.len() - 1) as f64 / (new_length - 1) as f64;

    for i in 0..new_length {
        let source_idx = i as f64 * ratio;
        let idx_low = source_idx.floor() as usize;
        let idx_high = (idx_low + 1).min(signal.len() - 1);
        let fraction = source_idx - idx_low as f64;

        // Linear interpolation
        let value = signal[idx_low] * (1.0 - fraction) + signal[idx_high] * fraction;
        resampled.push(value);
    }

    resampled
}

/// Calculate autocorrelation of signal
///
/// Like numpy.correlate(signal, signal, mode='full')
pub fn autocorrelation(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut result = Vec::with_capacity(n);

    // Normalize signal first
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    // Calculate autocorrelation for each lag
    for lag in 0..n {
        let mut sum = 0.0;
        let count = n - lag;

        for i in 0..count {
            sum += centered[i] * centered[i + lag];
        }

        result.push(sum / count as f64);
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_z_score_normalization() {
        let processor = SignalProcessor::new();
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let normalized = processor.z_score_normalize(&signal);

        // Check mean is approximately 0
        let mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);

        // Check std is approximately 1
        let variance: f64 = normalized
            .iter()
            .map(|&x| x.powi(2))
            .sum::<f64>() / (normalized.len() - 1) as f64;
        assert_relative_eq!(variance, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nan_removal() {
        let processor = SignalProcessor::new();
        let signal = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];

        let (clean, nan_pct) = processor.remove_nans(&signal);

        assert_eq!(clean, vec![1.0, 2.0, 3.0]);
        assert_relative_eq!(nan_pct, 40.0, epsilon = 1e-10);  // 2/5 = 40%
    }

    #[test]
    fn test_moving_average() {
        let processor = SignalProcessor::new();
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let smoothed = processor.moving_average(&signal, 3);

        // Middle values should be averages of 3 points
        assert_relative_eq!(smoothed[2], 3.0, epsilon = 1e-10);  // (2+3+4)/3
    }

    #[test]
    fn test_constant_signal() {
        let processor = SignalProcessor::new();
        let signal = vec![5.0; 100];

        let normalized = processor.z_score_normalize(&signal);

        // Constant signal should normalize to all zeros
        for &val in &normalized {
            assert_relative_eq!(val, 0.0, epsilon = 1e-10);
        }
    }
}

// Key Rust concepts demonstrated:
// 1. Struct methods: impl blocks add functionality
// 2. Option<T>: Handling nullable returns elegantly
// 3. Iterators: Functional programming with map/filter
// 4. Slices: Borrowing parts of arrays efficiently
// 5. Pattern matching: Elegant control flow
// 6. Error handling: Result<T, E> for fallible operations
// 7. Testing: Built-in test framework
// 8. Floating point: Careful handling of NaN/Infinity
// 9. Memory efficiency: Vec capacity hints
// 10. Zero-copy where possible: Using references