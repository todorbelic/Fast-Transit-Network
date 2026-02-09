mod graph;
mod algos;

use graph::{CSRGraph, BidirectionalCSRGraph};
use algos::{
    bfs_sequential, bfs_parallel,
    wcc_sequential, wcc_parallel,
    pagerank_sequential, pagerank_parallel,
};
use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::{Write, BufWriter};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "tool")]
#[command(about = "Graph algorithms tool for BFS, WCC, and PageRank", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run Breadth-First Search
    Bfs {
        /// Input edge list file
        #[arg(long)]
        input: String,

        /// Source vertex for BFS
        #[arg(long)]
        source: usize,

        /// Execution mode: seq or par
        #[arg(long)]
        mode: String,

        /// Number of threads (only used in parallel mode)
        #[arg(long, default_value = "8")]
        threads: usize,

        /// Output file for results
        #[arg(long)]
        out: String,
    },

    /// Run Weakly Connected Components
    Wcc {
        /// Input edge list file
        #[arg(long)]
        input: String,

        /// Execution mode: seq or par
        #[arg(long)]
        mode: String,

        /// Number of threads (only used in parallel mode)
        #[arg(long, default_value = "8")]
        threads: usize,

        /// Output file for results
        #[arg(long)]
        out: String,
    },

    /// Run PageRank
    Pagerank {
        /// Input edge list file
        #[arg(long)]
        input: String,

        /// Execution mode: seq or par
        #[arg(long)]
        mode: String,

        /// Number of threads (only used in parallel mode)
        #[arg(long, default_value = "8")]
        threads: usize,

        /// Output file for results
        #[arg(long)]
        out: String,

        /// Damping factor (alpha)
        #[arg(long, default_value = "0.85")]
        alpha: f64,

        /// Maximum iterations
        #[arg(long, default_value = "100")]
        iters: usize,

        /// Convergence tolerance (epsilon)
        #[arg(long, default_value = "1e-6")]
        eps: f64,
    },
}


fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bfs { input, source, mode, threads, out } => {
            run_bfs(&input, source, &mode, threads, &out)?;
        }
        Commands::Wcc { input, mode, threads, out } => {
            run_wcc(&input, &mode, threads, &out)?;
        }
        Commands::Pagerank { input, mode, threads, out, alpha, iters, eps } => {
            run_pagerank(&input, &mode, threads, &out, alpha, iters, eps)?;
        }
    }

    Ok(())
}

fn run_bfs(
    input_file: &str,
    source: usize,
    mode: &str,
    threads: usize,
    output_file: &str,
) -> std::io::Result<()> {
    println!("Running BFS...");
    println!("  Input: {}", input_file);
    println!("  Source: {}", source);
    println!("  Mode: {}", mode);

    // Load graph
    let graph = CSRGraph::from_edge_list(input_file)?;
    println!("  Loaded graph: {} vertices, {} edges", graph.num_vertices(), graph.num_edges());

    // Validate source vertex
    if source >= graph.num_vertices() {
        eprintln!("Error: Source vertex {} is out of range (graph has {} vertices)",
            source, graph.num_vertices());
        std::process::exit(1);
    }

    // Set number of threads for parallel execution
    if mode == "par" {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
        println!("  Threads: {}", threads);
    }

    // Run BFS
    let start = Instant::now();
    let distances = match mode {
        "seq" => bfs_sequential(&graph, source),
        "par" => bfs_parallel(&graph, source),
        _ => {
            eprintln!("Error: Invalid mode '{}'. Use 'seq' or 'par'", mode);
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    println!("  Completed in {:?}", elapsed);

    // Write results to file
    write_bfs_results(output_file, source, &distances)?;
    println!("  Results written to {}", output_file);

    Ok(())
}

fn run_wcc(
    input_file: &str,
    mode: &str,
    threads: usize,
    output_file: &str,
) -> std::io::Result<()> {
    println!("Running WCC...");
    println!("  Input: {}", input_file);
    println!("  Mode: {}", mode);

    // Load graph
    let graph = BidirectionalCSRGraph::from_edge_list(input_file)?;
    println!("  Loaded graph: {} vertices, {} edges", graph.num_vertices(), graph.num_edges());

    // Set number of threads for parallel execution
    if mode == "par" {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
        println!("  Threads: {}", threads);
    }

    // Run WCC
    let start = Instant::now();
    let components = match mode {
        "seq" => wcc_sequential(&graph),
        "par" => wcc_parallel(&graph),
        _ => {
            eprintln!("Error: Invalid mode '{}'. Use 'seq' or 'par'", mode);
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    println!("  Completed in {:?}", elapsed);

    // Count number of components
    use std::collections::HashSet;
    let unique_components: HashSet<_> = components.iter().copied().collect();
    let num_components = unique_components.len();
    println!("  Found {} components", num_components);

    // Write results to file
    write_wcc_results(output_file, &components)?;
    println!("  Results written to {}", output_file);
    println!("  Found {} components", num_components);

    // Write results to file
    write_wcc_results(output_file, &components)?;
    println!("  Results written to {}", output_file);

    Ok(())
}

fn run_pagerank(
    input_file: &str,
    mode: &str,
    threads: usize,
    output_file: &str,
    alpha: f64,
    iters: usize,
    eps: f64,
) -> std::io::Result<()> {
    println!("Running PageRank...");
    println!("  Input: {}", input_file);
    println!("  Mode: {}", mode);
    println!("  Alpha (damping): {}", alpha);
    println!("  Max iterations: {}", iters);
    println!("  Epsilon (tolerance): {}", eps);

    // Load graph
    let graph = BidirectionalCSRGraph::from_edge_list(input_file)?;
    println!("  Loaded graph: {} vertices, {} edges", graph.num_vertices(), graph.num_edges());

    // Set number of threads for parallel execution
    if mode == "par" {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
        println!("  Threads: {}", threads);
    }

    // Run PageRank
    let start = Instant::now();
    let ranks = match mode {
        "seq" => pagerank_sequential(&graph, alpha, iters, eps),
        "par" => pagerank_parallel(&graph, alpha, iters, eps),
        _ => {
            eprintln!("Error: Invalid mode '{}'. Use 'seq' or 'par'", mode);
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    println!("  Completed in {:?}", elapsed);

    // Verify sum
    let sum: f64 = ranks.iter().sum();
    println!("  Sum of ranks: {:.10}", sum);

    // Write results to file
    write_pagerank_results(output_file, &ranks)?;
    println!("  Results written to {}", output_file);

    Ok(())
}

fn write_bfs_results(output_file: &str, source: usize, distances: &[i32]) -> std::io::Result<()> {
    let file = File::create(output_file)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# BFS results from source vertex {}", source)?;
    writeln!(writer, "# Format: vertex distance")?;

    for (vertex, &distance) in distances.iter().enumerate() {
        if distance >= 0 {
            writeln!(writer, "{} {}", vertex, distance)?;
        } else {
            writeln!(writer, "{} -1", vertex)?;
        }
    }

    Ok(())
}

fn write_wcc_results(output_file: &str, components: &[usize]) -> std::io::Result<()> {
    let file = File::create(output_file)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# WCC results")?;
    writeln!(writer, "# Format: vertex component_id")?;

    for (vertex, &component) in components.iter().enumerate() {
        writeln!(writer, "{} {}", vertex, component)?;
    }

    Ok(())
}

fn write_pagerank_results(output_file: &str, ranks: &[f64]) -> std::io::Result<()> {
    let file = File::create(output_file)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# PageRank results")?;
    writeln!(writer, "# Format: vertex rank")?;

    for (vertex, &rank) in ranks.iter().enumerate() {
        writeln!(writer, "{} {:.10}", vertex, rank)?;
    }

    Ok(())
}