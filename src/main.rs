// main.rs - Entry point of the Rust application
// In Rust, the main() function is where execution starts (like Python's if __name__ == "__main__")

// Import statements in Rust use the 'use' keyword (like 'import' in Python or TypeScript)
// The '::' is the namespace separator (like '.' in Python or TypeScript)
use clap::{Parser, Subcommand};  // Command-line argument parsing
use anyhow::Result;              // Result type for error handling (no try/catch!)
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use log::{info, error, warn, debug};  // Logging macros
use rayon::prelude::*;           // Parallel iterators (like multiprocessing)
use std::path::{Path, PathBuf};  // Path handling (like os.path or pathlib)
use std::fs;                     // File system operations
use std::time::Instant;          // For timing operations
use chrono::Local;               // For timestamps

// Module declarations - Rust organizes code in modules (like Python modules)
// Each 'mod' statement tells Rust to look for a file with that name
mod entropy;                     // entropy.rs - Contains entropy calculations
mod signal_processing;           // signal_processing.rs - Signal preprocessing
mod data_loader;                 // data_loader.rs - CSV loading
mod feature_extractor;           // feature_extractor.rs - Main extraction logic
mod types;                       // types.rs - Custom types and structs

// Bring types into scope
use crate::types::{SignalType, Dataset, ExtractionConfig, Features, LongFormatFeatures};
use crate::feature_extractor::FeatureExtractor;

/// AI4Pain Feature Extraction - Ultra-fast Rust implementation
///
/// This is a doc comment (/// or //!) that generates documentation
/// Similar to docstrings in Python or JSDoc in TypeScript
///
/// # Examples (like Python doctest)
/// ```
/// ai4pain extract --dataset train --signal-type bvp
/// ```
#[derive(Parser)]  // This is a "derive macro" - generates code at compile time
#[command(name = "ai4pain")]
#[command(author = "AI4Pain Team")]
#[command(version = "2.0.0")]
#[command(about = "Ultra-fast entropy-based feature extraction", long_about = None)]
struct Cli {
    /// Verbosity level (-v, -vv, -vvv)
    /// Each 'v' increases logging detail
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,  // u8 = unsigned 8-bit integer (0-255)

    /// Number of parallel workers (defaults to all CPUs)
    #[arg(short = 'j', long, default_value_t = 0)]
    workers: usize,  // usize = pointer-sized unsigned integer

    /// Output directory for results
    #[arg(short, long, default_value = "results")]
    output: PathBuf,  // PathBuf is like Path but owned (String vs &str)

    #[command(subcommand)]
    command: Commands,
}

// Enum in Rust is like a discriminated union in TypeScript
// Much more powerful than Python enums - can hold data!
#[derive(Subcommand)]
enum Commands {
    /// Extract features from physiological signals
    Extract {
        /// Dataset(s) to process (space-separated: train validation test)
        #[arg(short, long, value_delimiter = ' ', default_values_t = vec!["train".to_string(), "validation".to_string(), "test".to_string()])]
        dataset: Vec<String>,

        /// Signal type(s) (space-separated: bvp eda resp spo2)
        #[arg(short, long, value_delimiter = ' ', default_values_t = vec!["bvp".to_string(), "eda".to_string(), "resp".to_string(), "spo2".to_string()])]
        signal_type: Vec<String>,

        /// Skip signals with more than this % of NaN values
        #[arg(long, default_value_t = 85.0)]
        nan_threshold: f64,  // f64 = 64-bit float (like Python's float)

        /// Embedding dimensions (comma-separated)
        #[arg(long, default_value = "3,4,5,6,7", value_delimiter = ',')]
        dimensions: Vec<usize>,  // Vec<T> is like Python's List[T]

        /// Time delays (comma-separated)
        #[arg(long, default_value = "1,2,3", value_delimiter = ',')]
        taus: Vec<usize>,
    },

    /// Benchmark Rust vs Python implementation
    Benchmark {
        /// Number of iterations
        #[arg(short = 'n', long, default_value_t = 100)]
        iterations: usize,
    },

    /// Validate extracted features
    Validate {
        /// Path to features CSV file
        #[arg(short, long)]
        file: PathBuf,
    },
}

// The main function - where execution begins
// Result<()> means it returns Ok(()) on success or an error
// This is Rust's approach to error handling without exceptions
fn main() -> Result<()> {
    // Parse command-line arguments
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    // In Rust, 'match' is pattern matching (like switch but more powerful)
    let log_level = match cli.verbose {
        0 => "warn",      // Default level
        1 => "info",      // -v
        2 => "debug",     // -vv
        _ => "trace",     // -vvv or more
    };

    // Initialize the logger
    // The '?' operator is like try/catch - returns early if error
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(log_level)
    ).init();

    // Log startup information
    info!("AI4Pain Rust Feature Extraction v2.0.0");
    info!("Starting at {}", Local::now().format("%Y-%m-%d %H:%M:%S"));

    // Determine number of workers
    // 'if' expressions return values in Rust (like ternary in JS)
    let num_workers = if cli.workers == 0 {
        num_cpus::get()  // Get number of CPU cores
    } else {
        cli.workers
    };

    info!("Using {} parallel workers", num_workers);

    // Configure Rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build_global()?;  // The ? propagates errors up

    // Create output directory if it doesn't exist
    fs::create_dir_all(&cli.output)?;

    // Match on the subcommand (pattern matching again)
    match cli.command {
        Commands::Extract {
            dataset,
            signal_type,
            nan_threshold,
            dimensions,
            taus,
        } => {
            // Call the extraction function with multi-select support
            run_extraction(
                dataset,
                signal_type,
                nan_threshold,
                dimensions,
                taus,
                &cli.output,
            )?;
        },

        Commands::Benchmark { iterations } => {
            run_benchmark(iterations)?;
        },

        Commands::Validate { file } => {
            validate_features(&file)?;
        },
    }

    // Return Ok(()) to indicate success
    // In Rust, the last expression without ';' is the return value
    Ok(())
}

/// Main extraction function
///
/// This function orchestrates the entire feature extraction process
/// Processes multiple datasets and signal types, generating granular output files
fn run_extraction(
    datasets: Vec<String>,
    signal_types_str: Vec<String>,
    nan_threshold: f64,
    dimensions: Vec<usize>,
    taus: Vec<usize>,
    output_dir: &Path,
) -> Result<()> {

    // Record start time
    let start = Instant::now();

    info!("{}", "=".repeat(60));
    info!("Starting AI4Pain feature extraction pipeline");
    info!("{}", "=".repeat(60));

    // Parse datasets
    let mut dataset_enums = Vec::new();
    for ds in &datasets {
        let dataset_enum = match ds.as_str() {
            "train" => Dataset::Train,
            "validation" => Dataset::Validation,
            "test" => Dataset::Test,
            _ => {
                anyhow::bail!("Invalid dataset: {}. Use train/validation/test", ds);
            }
        };
        dataset_enums.push(dataset_enum);
    }

    // Parse signal types
    let mut signal_types = Vec::new();
    for st in &signal_types_str {
        let signal_type = match st.to_lowercase().as_str() {
            "bvp" => SignalType::Bvp,
            "eda" => SignalType::Eda,
            "resp" => SignalType::Resp,
            "spo2" => SignalType::SpO2,
            _ => anyhow::bail!("Invalid signal type: {}", st),
        };
        signal_types.push(signal_type);
    }

    // Process each dataset
    for dataset_enum in dataset_enums {
        info!("Processing {} dataset...", dataset_enum.as_str());

        // Process each signal type for this dataset
        for signal_type in &signal_types {
            info!("  Processing {} signals...", signal_type.as_str());

            // Create extraction configuration for this dataset × signal_type
            let config = ExtractionConfig {
                dataset: dataset_enum,
                signal_types: vec![*signal_type],  // Single signal type per run
                nan_threshold,
                dimensions: dimensions.clone(),
                taus: taus.clone(),
                output_dir: output_dir.to_path_buf(),
            };

            // Create progress bars
            let multi_progress = MultiProgress::new();

            // Create feature extractor
            let mut extractor = FeatureExtractor::new(config, multi_progress)?;

            // Run extraction
            let features = extractor.extract_all()?;

            // Save granular output file: results_<dataset>_<signal_type>.csv
            let output_filename = format!(
                "results_{}_{}.csv",
                dataset_enum.as_str(),
                signal_type.as_str().to_lowercase()
            );
            let output_file = output_dir.join(&output_filename);

            info!("  Saving {} rows to {}", features.len(), output_filename);
            features.save_to_csv(&output_file)?;
        }
    }

    // Print final summary
    let elapsed = start.elapsed();
    info!("{}", "=".repeat(60));
    info!("✅ Extraction complete!");
    info!("  Total time: {:.2} seconds", elapsed.as_secs_f64());
    info!("  Datasets processed: {}", datasets.len());
    info!("  Signal types processed: {}", signal_types_str.len());
    info!("  Output directory: {:?}", output_dir);
    info!("{}", "=".repeat(60));

    Ok(())
}

/// Run benchmark comparing Rust to Python
fn run_benchmark(iterations: usize) -> Result<()> {
    info!("Running benchmark with {} iterations", iterations);

    // Create test data
    let test_signal: Vec<f64> = (0..10000)
        .map(|i| (i as f64).sin())  // Closure (like lambda in Python)
        .collect();  // Collect iterator into Vec

    // Benchmark Rust implementation
    let rust_start = Instant::now();

    // Run iterations in parallel using Rayon
    // par_iter() creates a parallel iterator - automatic parallelization!
    let _results: Vec<_> = (0..iterations)
        .into_par_iter()  // Convert to parallel iterator
        .map(|_| {
            // Calculate entropy for benchmark
            entropy::permutation_entropy(&test_signal, 3, 1)
        })
        .collect();

    let rust_duration = rust_start.elapsed();

    info!("Rust benchmark results:");
    info!("  Total time: {:.3} seconds", rust_duration.as_secs_f64());
    info!("  Per iteration: {:.3} ms", rust_duration.as_secs_f64() * 1000.0 / iterations as f64);
    info!("  Iterations/second: {:.1}", iterations as f64 / rust_duration.as_secs_f64());

    // Note: Python benchmark would require calling Python from Rust
    // or running separately and comparing results

    Ok(())
}

/// Validate feature file (long format)
fn validate_features(file: &Path) -> Result<()> {
    info!("Validating features from {:?}", file);

    // Check file exists
    if !file.exists() {
        anyhow::bail!("File not found: {:?}", file);
    }

    // Load and validate features (long format)
    let features = LongFormatFeatures::load_from_csv(file)?;

    info!("Validation results:");
    info!("  Total rows: {}", features.len());
    info!("  Unique datasets: {:?}", features.datasets());
    info!("  Unique signal types: {:?}", features.signal_types());

    // Check for common issues
    if features.len() == 0 {
        error!("No features found in file!");
        anyhow::bail!("Empty feature file");
    }

    info!("✅ Validation complete!");

    Ok(())
}

// Key Rust concepts demonstrated:
// 1. Ownership: Variables own their data, transferred with move semantics
// 2. Borrowing: & creates references that don't own data
// 3. Pattern Matching: match expressions for control flow
// 4. Error Handling: Result<T, E> instead of exceptions
// 5. Traits: Like interfaces but more powerful
// 6. Macros: Code generation at compile time (vec!, info!, etc.)
// 7. Iterators: Lazy evaluation and functional programming
// 8. Parallel Processing: Rayon for fearless concurrency
// 9. Type Safety: Strong static typing prevents bugs
// 10. Zero-Cost Abstractions: High-level code compiles to optimal assembly
