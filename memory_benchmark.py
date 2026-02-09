#!/usr/bin/env python3
"""
Memory consumption benchmark for graph algorithms
Measures peak memory usage for sequential and parallel versions
"""

import subprocess
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

# Configuration
TOOL_PATH = "./target/release/tool"
OUTPUT_DIR = "benchmark_results"
GRAPHS_DIR = "test_graphs"

# Ensure directories exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(GRAPHS_DIR).mkdir(exist_ok=True)


def generate_large_graph(filename, num_vertices=50000):
    """Generate a large test graph for memory benchmarking"""
    print(f"Generating large graph: {num_vertices} vertices...")
    edges = []

    # Create a more complex structure
    # Main chain
    for i in range(num_vertices - 1):
        edges.append((i, i + 1))

    # Add cross edges (create shortcuts)
    for i in range(0, num_vertices - 1, 50):
        if i + 100 < num_vertices:
            edges.append((i, i + 100))
        if i + 200 < num_vertices:
            edges.append((i, i + 200))
        if i + 500 < num_vertices:
            edges.append((i, i + 500))

    # Add some random edges for complexity
    import random
    random.seed(42)
    for _ in range(num_vertices // 5):
        src = random.randint(0, num_vertices - 1)
        dst = random.randint(0, num_vertices - 1)
        if src != dst:
            edges.append((src, dst))

    # Add some back edges
    for i in range(100, num_vertices, 100):
        if i - 50 >= 0:
            edges.append((i, i - 50))

    # Write to file
    with open(filename, 'w') as f:
        for src, dst in edges:
            f.write(f"{src} {dst}\n")

    print(f"  Generated {len(edges)} edges")
    return len(edges)


def get_memory_usage_linux(pid):
    """Get memory usage in MB for a process on Linux"""
    try:
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    # VmRSS is resident set size (actual physical memory used)
                    mem_kb = int(line.split()[1])
                    return mem_kb / 1024  # Convert to MB
    except:
        return None
    return None


def get_memory_usage_mac(pid):
    """Get memory usage in MB for a process on macOS"""
    try:
        result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)],
                              capture_output=True, text=True)
        if result.returncode == 0:
            mem_kb = int(result.stdout.strip())
            return mem_kb / 1024  # Convert to MB
    except:
        return None
    return None


def get_memory_usage(pid):
    """Get memory usage in MB (cross-platform)"""
    # Try Linux first
    mem = get_memory_usage_linux(pid)
    if mem is not None:
        return mem

    # Try macOS
    mem = get_memory_usage_mac(pid)
    if mem is not None:
        return mem

    return None


def measure_memory_peak(command):
    """Run command and measure peak memory usage"""
    print(f"  Running: {command}")

    # Start the process
    process = subprocess.Popen(command, shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    pid = process.pid
    peak_memory = 0
    measurements = []

    # Poll memory usage while process is running
    while process.poll() is None:
        mem = get_memory_usage(pid)
        if mem is not None:
            measurements.append(mem)
            peak_memory = max(peak_memory, mem)
        time.sleep(0.01)  # Sample every 10ms

    # Wait for process to complete
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"  Error running command: {stderr.decode()}")
        return None, None

    # Get average memory (excluding first few samples for initialization)
    if len(measurements) > 10:
        avg_memory = np.mean(measurements[5:])
    else:
        avg_memory = np.mean(measurements) if measurements else peak_memory

    return peak_memory, avg_memory


def run_bfs_memory_benchmark(graph_file, source=0, threads=4):
    """Measure memory usage for BFS"""
    print("\n--- BFS Memory Benchmark ---")

    results = {}

    # Sequential
    seq_out = f"{OUTPUT_DIR}/bfs_memory_seq.txt"
    cmd_seq = f"{TOOL_PATH} bfs --input {graph_file} --source {source} --mode seq --out {seq_out}"
    print("Sequential BFS:")
    peak_seq, avg_seq = measure_memory_peak(cmd_seq)

    if peak_seq is not None:
        print(f"  Peak memory: {peak_seq:.2f} MB")
        print(f"  Avg memory:  {avg_seq:.2f} MB")
        results['seq_peak'] = peak_seq
        results['seq_avg'] = avg_seq

    # Parallel
    par_out = f"{OUTPUT_DIR}/bfs_memory_par.txt"
    cmd_par = f"{TOOL_PATH} bfs --input {graph_file} --source {source} --mode par --threads {threads} --out {par_out}"
    print(f"Parallel BFS ({threads} threads):")
    peak_par, avg_par = measure_memory_peak(cmd_par)

    if peak_par is not None:
        print(f"  Peak memory: {peak_par:.2f} MB")
        print(f"  Avg memory:  {avg_par:.2f} MB")
        results['par_peak'] = peak_par
        results['par_avg'] = avg_par

    return results


def run_wcc_memory_benchmark(graph_file, threads=4):
    """Measure memory usage for WCC"""
    print("\n--- WCC Memory Benchmark ---")

    results = {}

    # Sequential
    seq_out = f"{OUTPUT_DIR}/wcc_memory_seq.txt"
    cmd_seq = f"{TOOL_PATH} wcc --input {graph_file} --mode seq --out {seq_out}"
    print("Sequential WCC:")
    peak_seq, avg_seq = measure_memory_peak(cmd_seq)

    if peak_seq is not None:
        print(f"  Peak memory: {peak_seq:.2f} MB")
        print(f"  Avg memory:  {avg_seq:.2f} MB")
        results['seq_peak'] = peak_seq
        results['seq_avg'] = avg_seq

    # Parallel
    par_out = f"{OUTPUT_DIR}/wcc_memory_par.txt"
    cmd_par = f"{TOOL_PATH} wcc --input {graph_file} --mode par --threads {threads} --out {par_out}"
    print(f"Parallel WCC ({threads} threads):")
    peak_par, avg_par = measure_memory_peak(cmd_par)

    if peak_par is not None:
        print(f"  Peak memory: {peak_par:.2f} MB")
        print(f"  Avg memory:  {avg_par:.2f} MB")
        results['par_peak'] = peak_par
        results['par_avg'] = avg_par

    return results


def run_pagerank_memory_benchmark(graph_file, alpha=0.85, iters=50, eps=1e-10, threads=4):
    """Measure memory usage for PageRank"""
    print("\n--- PageRank Memory Benchmark ---")

    results = {}

    # Sequential
    seq_out = f"{OUTPUT_DIR}/pr_memory_seq.txt"
    cmd_seq = f"{TOOL_PATH} pagerank --input {graph_file} --mode seq --out {seq_out} --alpha {alpha} --iters {iters} --eps {eps}"
    print("Sequential PageRank:")
    peak_seq, avg_seq = measure_memory_peak(cmd_seq)

    if peak_seq is not None:
        print(f"  Peak memory: {peak_seq:.2f} MB")
        print(f"  Avg memory:  {avg_seq:.2f} MB")
        results['seq_peak'] = peak_seq
        results['seq_avg'] = avg_seq

    # Parallel
    par_out = f"{OUTPUT_DIR}/pr_memory_par.txt"
    cmd_par = f"{TOOL_PATH} pagerank --input {graph_file} --mode par --threads {threads} --out {par_out} --alpha {alpha} --iters {iters} --eps {eps}"
    print(f"Parallel PageRank ({threads} threads):")
    peak_par, avg_par = measure_memory_peak(cmd_par)

    if peak_par is not None:
        print(f"  Peak memory: {peak_par:.2f} MB")
        print(f"  Avg memory:  {avg_par:.2f} MB")
        results['par_peak'] = peak_par
        results['par_avg'] = avg_par

    return results


def plot_memory_results(results, num_vertices, num_edges):
    """Plot memory consumption results"""
    print("\nGenerating memory consumption plots...")

    algorithms = ['BFS', 'WCC', 'PageRank']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Memory Consumption Analysis\nGraph: {num_vertices:,} vertices, {num_edges:,} edges',
                 fontsize=14, fontweight='bold')

    # Prepare data
    seq_peak = [results['bfs']['seq_peak'], results['wcc']['seq_peak'], results['pagerank']['seq_peak']]
    par_peak = [results['bfs']['par_peak'], results['wcc']['par_peak'], results['pagerank']['par_peak']]
    seq_avg = [results['bfs']['seq_avg'], results['wcc']['seq_avg'], results['pagerank']['seq_avg']]
    par_avg = [results['bfs']['par_avg'], results['wcc']['par_avg'], results['pagerank']['par_avg']]

    x = np.arange(len(algorithms))
    width = 0.35

    # Plot 1: Peak Memory Usage
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, seq_peak, width, label='Sequential', color='#3498db')
    bars2 = ax1.bar(x + width/2, par_peak, width, label='Parallel (16 threads)', color='#e74c3c')

    ax1.set_ylabel('Memory (MB)', fontweight='bold', fontsize=12)
    ax1.set_title('Peak Memory Consumption', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add overhead calculation
    for i, (seq, par) in enumerate(zip(seq_peak, par_peak)):
        overhead = ((par - seq) / seq * 100) if seq > 0 else 0
        ax1.text(i, max(seq, par) * 1.15, f'+{overhead:.1f}%',
                ha='center', fontsize=8, color='darkred' if overhead > 0 else 'darkgreen')

    # Plot 2: Average Memory Usage
    ax2 = axes[1]
    bars1 = ax2.bar(x - width/2, seq_avg, width, label='Sequential', color='#2ecc71')
    bars2 = ax2.bar(x + width/2, par_avg, width, label='Parallel (16 threads)', color='#f39c12')

    ax2.set_ylabel('Memory (MB)', fontweight='bold', fontsize=12)
    ax2.set_title('Average Memory Consumption', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save plot
    plot_file = f"{OUTPUT_DIR}/memory_benchmark.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Memory benchmark plot saved to {plot_file}")

    # Create detailed comparison table plot
    create_memory_table(results, num_vertices, num_edges)

    plt.show()


def create_memory_table(results, num_vertices, num_edges):
    """Create a detailed memory comparison table"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f'Memory Consumption Detailed Comparison\nGraph: {num_vertices:,} vertices, {num_edges:,} edges',
                 fontsize=14, fontweight='bold')

    ax.axis('tight')
    ax.axis('off')

    table_data = []
    table_data.append(['Algorithm', 'Mode', 'Peak (MB)', 'Avg (MB)', 'Overhead', 'Notes'])

    for algo in ['bfs', 'wcc', 'pagerank']:
        algo_name = algo.upper()
        data = results[algo]

        # Sequential row
        table_data.append([
            algo_name,
            'Sequential',
            f"{data['seq_peak']:.2f}",
            f"{data['seq_avg']:.2f}",
            '-',
            'Baseline'
        ])

        # Parallel row
        peak_overhead = ((data['par_peak'] - data['seq_peak']) / data['seq_peak'] * 100)
        avg_overhead = ((data['par_avg'] - data['seq_avg']) / data['seq_avg'] * 100)

        table_data.append([
            '',
            'Parallel (4t)',
            f"{data['par_peak']:.2f}",
            f"{data['par_avg']:.2f}",
            f"+{peak_overhead:.1f}%",
            'Thread overhead' if peak_overhead > 10 else 'Efficient'
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(1, len(table_data)):
        for j in range(6):
            if 'Parallel' in table_data[i][1]:
                table[(i, j)].set_facecolor('#ffe6e6')
            elif 'Sequential' in table_data[i][1]:
                table[(i, j)].set_facecolor('#e6f7ff')

            # Highlight algorithm names
            if j == 0 and table_data[i][0]:
                table[(i, j)].set_text_props(weight='bold')

    plt.tight_layout()

    table_file = f"{OUTPUT_DIR}/memory_table.png"
    plt.savefig(table_file, dpi=300, bbox_inches='tight')
    print(f"Memory table saved to {table_file}")


def print_memory_summary(results, num_vertices, num_edges):
    """Print memory summary to console"""
    print("\n" + "="*80)
    print("MEMORY CONSUMPTION SUMMARY")
    print("="*80)
    print(f"Graph: {num_vertices:,} vertices, {num_edges:,} edges")
    print("-" * 80)

    print(f"\n{'Algorithm':<15} {'Mode':<15} {'Peak (MB)':<12} {'Avg (MB)':<12} {'Overhead':<12}")
    print("-" * 80)

    for algo in ['bfs', 'wcc', 'pagerank']:
        data = results[algo]

        # Sequential
        print(f"{algo.upper():<15} {'Sequential':<15} {data['seq_peak']:>8.2f}    {data['seq_avg']:>8.2f}    {'-':<12}")

        # Parallel
        peak_overhead = ((data['par_peak'] - data['seq_peak']) / data['seq_peak'] * 100)
        print(f"{'':15} {'Parallel (4t)':<15} {data['par_peak']:>8.2f}    {data['par_avg']:>8.2f}    {peak_overhead:>+7.1f}%")
        print()

    print("-" * 80)
    print("\nKey Observations:")

    # Calculate average overhead
    total_overhead = 0
    for algo in ['bfs', 'wcc', 'pagerank']:
        data = results[algo]
        overhead = ((data['par_peak'] - data['seq_peak']) / data['seq_peak'] * 100)
        total_overhead += overhead

    avg_overhead = total_overhead / 3
    print(f"  - Average parallel overhead: {avg_overhead:.1f}%")
    print(f"  - Parallel uses {16} threads")
    print(f"  - Memory overhead mainly from thread stacks and atomic data structures")

    print("\n" + "="*80)


def main():
    """Main memory benchmark execution"""
    print("=" * 80)
    print("Graph Algorithm Memory Benchmark")
    print("=" * 80)

    # Check if tool exists
    if not os.path.exists(TOOL_PATH):
        print(f"Error: Tool not found at {TOOL_PATH}")
        print("Please build the project first with: cargo build --release")
        return

    # Check platform
    if sys.platform not in ['linux', 'darwin']:
        print("Warning: Memory measurement may not work on this platform")
        print("Supported platforms: Linux, macOS")

    # Generate large test graph
    graph_file = f"{GRAPHS_DIR}/large_memory_graph.txt"
    num_edges = generate_large_graph(graph_file, num_vertices=5000000)
    num_vertices = 5000000

    # Run memory benchmarks
    results = {}

    results['bfs'] = run_bfs_memory_benchmark(graph_file, source=0, threads=16)
    results['wcc'] = run_wcc_memory_benchmark(graph_file, threads=16)
    results['pagerank'] = run_pagerank_memory_benchmark(graph_file, threads=16)

    # Print summary
    print_memory_summary(results, num_vertices, num_edges)

    # Plot results
    plot_memory_results(results, num_vertices, num_edges)

    print("\n" + "=" * 80)
    print("Memory benchmark complete!")
    print(f"Results saved to {OUTPUT_DIR}/")
    print("  - memory_benchmark.png: Memory consumption plots")
    print("  - memory_table.png: Detailed comparison table")
    print("=" * 80)


if __name__ == "__main__":
    main()
