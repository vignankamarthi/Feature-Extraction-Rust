// entropy.rs - Core entropy calculation module
// This module implements all 5 entropy measures for physiological signal analysis
//
// For Python/Java/TypeScript developers:
// - Rust modules are like Python modules or TypeScript namespaces
// - Functions are public with 'pub' keyword (like export in JS)
// - No classes - just functions and structs with impl blocks

use std::collections::HashMap;
use ndarray::{Array1, Array2};  // Like NumPy arrays
use statrs::distribution::{Continuous, Normal};
use num_traits::Float;
use anyhow::Result;

// Type alias for cleaner code (like TypeScript type alias)
type Signal = Vec<f64>;
type Pattern = Vec<usize>;

/// Permutation Entropy (PE) - Bandt & Pompe (2002)
///
/// Measures complexity of ordinal patterns in time series
/// Lower values = more regular/predictable patterns
///
/// # Arguments
/// * `signal` - Input time series data (like numpy array)
/// * `dimension` - Embedding dimension (typically 3-7)
/// * `tau` - Time delay (typically 1-3)
///
/// # Returns
/// Normalized permutation entropy value [0, 1]
///
/// # Example
/// ```rust
/// let signal = vec![1.0, 2.0, 1.5, 3.0, 2.5];
/// let pe = permutation_entropy(&signal, 3, 1).unwrap();
/// ```
pub fn permutation_entropy(signal: &[f64], dimension: usize, tau: usize) -> Result<f64> {
    // Input validation (like Python's assert)
    if signal.len() < dimension * tau {
        anyhow::bail!("Signal too short for given dimension and tau");
    }

    // Extract ordinal patterns
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;

    // Count pattern frequencies
    // HashMap is like Python's dict or JS Map
    let mut pattern_counts: HashMap<Pattern, usize> = HashMap::new();

    // Iterate over patterns
    // for..in is like Python's for loop
    for pattern in patterns {
        // entry() gets or creates entry, like Python's defaultdict
        *pattern_counts.entry(pattern).or_insert(0) += 1;
    }

    // Calculate probabilities and entropy
    let total = pattern_counts.values().sum::<usize>() as f64;
    let mut entropy = 0.0;

    for count in pattern_counts.values() {
        let p = *count as f64 / total;
        if p > 0.0 {
            // log2 for bits (like np.log2)
            entropy -= p * p.log2();
        }
    }

    // Normalize by maximum possible entropy
    let max_entropy = (factorial(dimension) as f64).log2();
    Ok(entropy / max_entropy)
}

/// Statistical Complexity Measure - Rosso et al. (2007)
///
/// Measures departure from equilibrium distribution
/// Combines entropy with disequilibrium measure
///
/// This is more sophisticated than simple entropy - it captures
/// the "interesting" complexity between complete order and randomness
pub fn statistical_complexity(signal: &[f64], dimension: usize, tau: usize) -> Result<f64> {
    // Get permutation entropy first
    let pe = permutation_entropy(signal, dimension, tau)?;

    // Calculate disequilibrium (Jensen-Shannon divergence)
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;
    let disequilibrium = calculate_disequilibrium(&patterns, dimension)?;

    // Complexity = entropy × disequilibrium
    // This captures both unpredictability AND structure
    Ok(pe * disequilibrium)
}

/// Fisher Information Measure - Martin et al. (2003)
///
/// Measures local sensitivity to parameter changes
/// Higher values = more information about system state
///
/// Returns tuple of (Permutation Entropy, Fisher Information)
/// Note: First value is PE (ordpy naming convention), second is Fisher Information
pub fn fisher_information(signal: &[f64], dimension: usize, tau: usize) -> Result<(f64, f64)> {
    // Calculate patterns
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;

    // CRITICAL: ordpy includes missing states (zero probabilities) for Fisher Information
    // Build FULL probability distribution including zeros for missing patterns
    let n_possible = factorial(dimension);
    let mut full_prob_dist = vec![0.0; n_possible];

    // Count pattern occurrences
    let mut pattern_counts: std::collections::HashMap<Pattern, usize> = std::collections::HashMap::new();
    for pattern in &patterns {
        *pattern_counts.entry(pattern.clone()).or_insert(0) += 1;
    }

    // Map patterns to indices and fill probabilities
    let total = patterns.len() as f64;
    for (pattern, count) in pattern_counts.iter() {
        // Convert pattern to unique index (lexicographic ordering)
        let idx = pattern_to_index(pattern);
        if idx < n_possible {
            full_prob_dist[idx] = *count as f64 / total;
        }
    }

    // Permutation entropy (first return value) - MUST match PE exactly
    let pe = permutation_entropy(signal, dimension, tau)?;

    // CRITICAL ordpy special case: if only one pattern exists (one prob = 1.0),
    // return Fisher_I = 1.0 (maximum information)
    let max_prob_count = full_prob_dist.iter().filter(|&&p| (p - 1.0).abs() < 1e-10).count();
    if max_prob_count == 1 {
        return Ok((0.0, 1.0));  // PE=0, Fisher_I=1.0
    }

    // Fisher Information - ordpy formula with FULL distribution including zeros
    // F = sum(diff(sqrt(probabilities[::-1]))^2) / 2

    // Reverse probabilities (ordpy uses [::-1])
    let mut prob_reversed = full_prob_dist.clone();
    prob_reversed.reverse();

    // Take square root
    let sqrt_probs: Vec<f64> = prob_reversed.iter().map(|&p| p.sqrt()).collect();

    // Calculate differences and sum squares
    let mut fisher_info = 0.0;
    for i in 1..sqrt_probs.len() {
        let diff = sqrt_probs[i] - sqrt_probs[i-1];
        fisher_info += diff * diff;
    }
    fisher_info /= 2.0;

    Ok((pe, fisher_info))
}

/// Convert pattern to unique index for full probability distribution
fn pattern_to_index(pattern: &[usize]) -> usize {
    // Lehmer code / factorial number system
    let mut index = 0;
    let n = pattern.len();

    for i in 0..n {
        let mut smaller_count = 0;
        for j in (i+1)..n {
            if pattern[j] < pattern[i] {
                smaller_count += 1;
            }
        }
        index = index * (n - i) + smaller_count;
    }

    index
}

/// Renyi Entropy - Generalization of Shannon entropy
///
/// Parameter q controls the weighting of probabilities
/// q=1 gives Shannon entropy, q→∞ gives min-entropy
///
/// Returns tuple of (Renyi Complexity, Renyi PE)
pub fn renyi_entropy(signal: &[f64], dimension: usize, tau: usize, q: f64) -> Result<(f64, f64)> {
    // Extract patterns and calculate probabilities
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;
    let prob_dist = calculate_probability_distribution(&patterns, dimension)?;

    let n = factorial(dimension) as f64;
    let uniform_dist = 1.0 / n;
    let n_states_not_occur = n - prob_dist.len() as f64;

    // Calculate Renyi entropy (normalized)
    let renyi_pe = if (q - 1.0).abs() < 1e-10 {
        // Shannon entropy case (q=1 limit)
        let h: f64 = prob_dist.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        h / n.ln()  // Normalize
    } else {
        // Renyi entropy formula
        let sum: f64 = prob_dist.iter()
            .map(|&p| p.powf(q))
            .sum();
        ((1.0 / (1.0 - q)) * sum.ln()) / n.ln()  // Normalize
    };

    // Jensen-Renyi divergence (complexity measure)
    let jr_div = if (q - 1.0).abs() < 1e-10 {
        // Shannon case - use JS divergence
        let mut s_p_plus_u_over_2 = 0.0;
        for &p in &prob_dist {
            let mix = (uniform_dist + p) / 2.0;
            if mix > 0.0 {
                s_p_plus_u_over_2 -= mix * mix.ln();
            }
        }
        // Account for missing states
        let mix_missing = 0.5 * uniform_dist;
        if mix_missing > 0.0 {
            s_p_plus_u_over_2 -= mix_missing * mix_missing.ln() * n_states_not_occur;
        }

        let s_of_p_over_2: f64 = prob_dist.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum::<f64>() / 2.0;

        let s_of_u_over_2 = n.ln() / 2.0;

        s_p_plus_u_over_2 - s_of_p_over_2 - s_of_u_over_2
    } else {
        // General Renyi case - Equation 4 from Physica A 498 (2018) 74-85
        let first_term: f64 = prob_dist.iter()
            .map(|&p| {
                let mix = (p + uniform_dist) / 2.0;
                mix.powf(1.0 - q) * p.powf(q)
            })
            .sum();

        let second_term: f64 = prob_dist.iter()
            .map(|&p| {
                let mix = (p + uniform_dist) / 2.0;
                (1.0 / n.powf(q)) * mix.powf(1.0 - q)
            })
            .sum();

        let missing_term = n_states_not_occur * (1.0 / n.powf(q)) * (1.0 / (2.0 * n)).powf(1.0 - q);

        (1.0 / (2.0 * (q - 1.0))) * (first_term.ln() + (second_term + missing_term).ln())
    };

    // Maximum Jensen-Renyi divergence
    let jr_div_max = if (q - 1.0).abs() < 1e-10 {
        -0.5 * (((n + 1.0) / n) * (n + 1.0).ln() + n.ln() - 2.0 * (2.0 * n).ln())
    } else {
        // Equation 5 from Physica A 498 (2018) 74-85
        let term1 = ((n + 1.0).powf(1.0 - q) + n - 1.0) / (2.0_f64.powf(1.0 - q) * n);
        let term2 = (1.0 - q) * ((n + 1.0) / (2.0 * n)).ln();
        (term1.ln() + term2) / (2.0 * (q - 1.0))
    };

    let renyi_complexity = renyi_pe * jr_div / jr_div_max;

    // Match Python/ordpy labeling: function returns (entropy, complexity) despite name
    Ok((renyi_pe, renyi_complexity))
}

/// Tsallis Entropy - Non-extensive entropy measure
///
/// Captures long-range correlations and non-extensive systems
/// Parameter q controls the non-extensivity (q=1 gives Shannon)
///
/// This is useful for systems with memory or long-range dependencies
pub fn tsallis_entropy(signal: &[f64], dimension: usize, tau: usize, q: f64) -> Result<(f64, f64)> {
    // Extract patterns and probabilities
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;
    let prob_dist = calculate_probability_distribution(&patterns, dimension)?;

    let n = factorial(dimension) as f64;
    let uniform_dist = 1.0 / n;
    let n_states_not_occur = n - prob_dist.len() as f64;

    // Calculate Tsallis entropy (normalized)
    let tsallis_pe = if (q - 1.0).abs() < 1e-10 {
        // Shannon entropy case (q=1 limit)
        let h: f64 = prob_dist.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();
        h / n.ln()  // Normalize
    } else {
        // Tsallis entropy formula
        let sum: f64 = prob_dist.iter()
            .map(|&p| p.powf(q))
            .sum();
        ((1.0 - sum) / (q - 1.0)) / ((n.powf(1.0 - q) - 1.0) / (1.0 - q))  // Normalize
    };

    // Jensen-Tsallis divergence (using q-logarithm)
    // Equation 10 from Physical Review E 95, 062106 (2017)

    let first_term: f64 = prob_dist.iter()
        .map(|&p| {
            let ratio = (uniform_dist + p) / (2.0 * p);
            p * logq(ratio, q)
        })
        .sum();
    let first_term = -0.5 * first_term;

    let second_term: f64 = prob_dist.iter()
        .map(|&p| {
            let val = n * (uniform_dist + p) / 2.0;
            logq(val, q)
        })
        .sum();

    let missing_term = logq(0.5, q) * n_states_not_occur;
    let second_term = -(0.5 / n) * (second_term + missing_term);

    let jt_div = first_term + second_term;

    // Maximum Jensen-Tsallis divergence
    // Equation 11 from Physical Review E 95, 062106 (2017)
    let jt_div_max = if (q - 1.0).abs() < 1e-10 {
        -0.5 * (((n + 1.0) / n) * (n + 1.0).ln() + n.ln() - 2.0 * (2.0 * n).ln())
    } else {
        let numerator = 2.0_f64.powf(2.0 - q) * n
            - (1.0 + n).powf(1.0 - q)
            - n * (1.0 + 1.0 / n).powf(1.0 - q)
            - n + 1.0;
        let denominator = (1.0 - q) * 2.0_f64.powf(2.0 - q) * n;
        numerator / denominator
    };

    let tsallis_complexity = tsallis_pe * jt_div / jt_div_max;

    // Match Python/ordpy labeling: function returns (entropy, complexity) despite name
    Ok((tsallis_pe, tsallis_complexity))
}

/// q-logarithm function for Tsallis entropy
fn logq(x: f64, q: f64) -> f64 {
    if (q - 1.0).abs() < 1e-10 {
        x.ln()
    } else {
        (x.powf(1.0 - q) - 1.0) / (1.0 - q)
    }
}

// ============================================================================
// Helper Functions (Private - not exposed outside module)
// ============================================================================

/// Extract ordinal patterns from signal
///
/// This is the core operation for permutation-based entropies
/// Converts time series segments into rank patterns
fn extract_ordinal_patterns(signal: &[f64], dimension: usize, tau: usize) -> Result<Vec<Pattern>> {
    let mut patterns = Vec::new();
    let n_patterns = signal.len() - (dimension - 1) * tau;

    // Slide window through signal
    for i in 0..n_patterns {
        // Extract embedded vector
        let mut embedded: Vec<(f64, usize)> = Vec::with_capacity(dimension);

        for j in 0..dimension {
            let idx = i + j * tau;
            embedded.push((signal[idx], j));
        }

        // Sort by value to get ranks (ordinal pattern)
        // sort_by is like Python's sorted() with key function
        embedded.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Extract pattern (ranks)
        let pattern: Pattern = embedded.iter().map(|&(_, idx)| idx).collect();
        patterns.push(pattern);
    }

    Ok(patterns)
}

/// Time-delay embedding of signal
///
/// Creates matrix where each row is a delayed version
/// Like sklearn.preprocessing.TimeDelayEmbedding
fn embed_signal(signal: &[f64], dimension: usize, tau: usize) -> Result<Array2<f64>> {
    let n_vectors = signal.len() - (dimension - 1) * tau;

    // Create 2D array (like np.zeros((n_vectors, dimension)))
    let mut embedded = Array2::zeros((n_vectors, dimension));

    for i in 0..n_vectors {
        for j in 0..dimension {
            embedded[[i, j]] = signal[i + j * tau];
        }
    }

    Ok(embedded)
}

/// Calculate probability distribution from patterns
fn calculate_probability_distribution(patterns: &[Pattern], dimension: usize) -> Result<Vec<f64>> {
    let mut pattern_counts: HashMap<Pattern, usize> = HashMap::new();

    // Count occurrences
    for pattern in patterns {
        *pattern_counts.entry(pattern.clone()).or_insert(0) += 1;
    }

    // Convert to probability distribution
    let total = patterns.len() as f64;
    let mut prob_dist: Vec<f64> = pattern_counts
        .values()
        .map(|&count| count as f64 / total)
        .collect();

    // Sort for consistent ordering
    prob_dist.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Ok(prob_dist)
}

/// Calculate disequilibrium (Jensen-Shannon divergence from uniform)
///
/// CRITICAL: Must match ordpy's exact formula including missing patterns contribution
/// ordpy line 681: s_of_p_plus_u_over_2 = -sum(p_plus_u_over_2*log(p_plus_u_over_2))
///                                         - (0.5*u)*log(0.5*u)*n_missing
fn calculate_disequilibrium(patterns: &[Pattern], dimension: usize) -> Result<f64> {
    let prob_dist = calculate_probability_distribution(patterns, dimension)?;

    // Uniform distribution (equilibrium)
    let n = factorial(dimension) as f64;
    let uniform_dist = 1.0 / n;
    let n_states_not_occuring = n - prob_dist.len() as f64;

    // Calculate JS divergence EXACTLY as ordpy does (3-term formula)

    // Term 1: s_of_p_plus_u_over_2 = H((P+U)/2)
    // This includes BOTH observed patterns AND missing patterns!
    let mut s_of_p_plus_u_over_2 = 0.0;

    // Contribution from observed patterns
    for &p in &prob_dist {
        let p_plus_u_over_2 = (uniform_dist + p) / 2.0;
        s_of_p_plus_u_over_2 -= p_plus_u_over_2 * p_plus_u_over_2.ln();
    }

    // CRITICAL: Contribution from missing patterns (where p=0)
    // For each missing pattern: -(u/2)*ln(u/2)
    let half_u = 0.5 * uniform_dist;
    s_of_p_plus_u_over_2 -= half_u * half_u.ln() * n_states_not_occuring;

    // Term 2: s_of_p_over_2 = H(P) / 2
    let s_of_p_over_2: f64 = prob_dist.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>() / 2.0;

    // Term 3: s_of_u_over_2 = H(U) / 2 = ln(n) / 2
    let s_of_u_over_2 = n.ln() / 2.0;

    // Jensen-Shannon divergence
    // JS(P||U) = H((P+U)/2) - H(P)/2 - H(U)/2
    let js_div = s_of_p_plus_u_over_2 - s_of_p_over_2 - s_of_u_over_2;

    // Maximum JS divergence (normalization)
    // js_div_max = -0.5*(((n+1)/n)*ln(n+1) + ln(n) - 2*ln(2*n))
    let js_div_max = -0.5 * (
        ((n + 1.0) / n) * (n + 1.0).ln() +
        n.ln() -
        2.0 * (2.0 * n).ln()
    );

    Ok(js_div / js_div_max)
}

/// Calculate factorial (n!)
///
/// Like math.factorial in Python
fn factorial(n: usize) -> usize {
    match n {
        0 | 1 => 1,
        _ => (2..=n).product(),  // Product of 2×3×...×n
    }
}

// ============================================================================
// Comprehensive Entropy Suite
// ============================================================================

/// Calculate all entropy measures at once
///
/// This is the main function used by the feature extractor
/// Returns all 8 entropy values in a struct
#[derive(Debug, Clone)]
pub struct EntropyFeatures {
    pub permutation_entropy: f64,
    pub complexity: f64,
    pub fisher_shannon: f64,
    pub fisher_information: f64,
    pub renyi_complexity: f64,
    pub renyi_pe: f64,
    pub tsallis_complexity: f64,
    pub tsallis_pe: f64,
}

impl EntropyFeatures {
    /// Calculate all entropy features for a signal
    pub fn calculate(signal: &[f64], dimension: usize, tau: usize) -> Result<Self> {
        // Calculate each entropy type
        // Using ? for error propagation (like Python's raise)

        let pe = permutation_entropy(signal, dimension, tau)?;
        let complexity = statistical_complexity(signal, dimension, tau)?;

        let (fisher_shannon, fisher_information) = fisher_information(signal, dimension, tau)?;

        // Use q=1 for Renyi/Tsallis (Shannon entropy limit - matches ordpy defaults)
        // CRITICAL: renyi_entropy returns (pe, complexity) NOT (complexity, pe)
        let (renyi_pe, renyi_complexity) = renyi_entropy(signal, dimension, tau, 1.0)?;
        let (tsallis_pe, tsallis_complexity) = tsallis_entropy(signal, dimension, tau, 1.0)?;

        Ok(Self {
            permutation_entropy: pe,
            complexity,
            fisher_shannon,
            fisher_information,
            renyi_complexity,
            renyi_pe,
            tsallis_complexity,
            tsallis_pe,
        })
    }

    /// Convert to vector for CSV output
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.permutation_entropy,
            self.complexity,
            self.fisher_shannon,
            self.fisher_information,
            self.renyi_complexity,
            self.renyi_pe,
            self.tsallis_complexity,
            self.tsallis_pe,
        ]
    }

    /// Get feature names for CSV header
    pub fn feature_names(dimension: usize, tau: usize) -> Vec<String> {
        vec![
            format!("PE_d{}_t{}", dimension, tau),
            format!("C_d{}_t{}", dimension, tau),
            format!("Fisher_S_d{}_t{}", dimension, tau),
            format!("Fisher_I_d{}_t{}", dimension, tau),
            format!("Renyi_C_d{}_t{}", dimension, tau),
            format!("Renyi_PE_d{}_t{}", dimension, tau),
            format!("Tsallis_C_d{}_t{}", dimension, tau),
            format!("Tsallis_PE_d{}_t{}", dimension, tau),
        ]
    }
}

// ============================================================================
// Tests - Rust's built-in testing framework
// ============================================================================

#[cfg(test)]  // Only compile when running tests
mod tests {
    use super::*;
    use approx::assert_relative_eq;  // For floating point comparison

    #[test]  // Mark function as test (like @pytest.mark)
    fn test_permutation_entropy() {
        // Create test signal (sine wave)
        let signal: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.1).sin())
            .collect();

        // Calculate PE
        let pe = permutation_entropy(&signal, 3, 1).unwrap();

        // Check result is in valid range
        assert!(pe >= 0.0 && pe <= 1.0);

        // Sine wave should have moderate entropy
        assert!(pe > 0.3 && pe < 0.7);
    }

    #[test]
    fn test_constant_signal_entropy() {
        // Constant signal should have zero entropy
        let signal = vec![1.0; 1000];

        let pe = permutation_entropy(&signal, 3, 1).unwrap();

        // Should be very close to 0
        assert_relative_eq!(pe, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_random_signal_entropy() {
        // Random signal should have high entropy
        use rand::prelude::*;
        let mut rng = thread_rng();

        let signal: Vec<f64> = (0..1000)
            .map(|_| rng.gen::<f64>())
            .collect();

        let pe = permutation_entropy(&signal, 3, 1).unwrap();

        // Should be close to 1 (maximum entropy)
        assert!(pe > 0.9);
    }

    #[test]
    fn test_all_entropies() {
        // Test that all entropies can be calculated
        let signal: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.3).cos())
            .collect();

        let features = EntropyFeatures::calculate(&signal, 4, 2).unwrap();

        // All values should be finite
        assert!(features.permutation_entropy.is_finite());
        assert!(features.complexity.is_finite());
        assert!(features.fisher_shannon.is_finite());
        assert!(features.fisher_information.is_finite());
        assert!(features.renyi_complexity.is_finite());
        assert!(features.renyi_pe.is_finite());
        assert!(features.tsallis_complexity.is_finite());
        assert!(features.tsallis_pe.is_finite());
    }
}

// Key Rust concepts in this file:
// 1. Module system: Organizing code into logical units
// 2. Public/Private: 'pub' makes things accessible outside module
// 3. Result<T, E>: Error handling without exceptions
// 4. Pattern Matching: Powerful control flow
// 5. Ownership: Functions borrow (&) or take ownership
// 6. Iterators: Functional programming with map/filter/collect
// 7. Traits: impl blocks add methods to structs
// 8. Testing: Built-in test framework with #[test]
// 9. Type Aliases: Making code more readable
// 10. Generics: Functions work with multiple types