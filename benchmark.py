#!/usr/bin/env python3
"""
Benchmark script for graph algorithms (BFS, WCC, PageRank)
Generates test graphs, runs sequential and parallel versions with different thread counts, and plots results
"""

import subprocess
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
TOOL_PATH = "./target/release/tool"
OUTPUT_DIR = "benchmark_results"
GRAPHS_DIR = "test_graphs"
THREAD_COUNTS = [2, 4, 8, 16]

# Ensure directories exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(GRAPHS_DIR).mkdir(exist_ok=True)


def generate_small_graph(filename, num_vertices=100):
    """Generate a small test graph"""
    print(f"Generating small graph: {num_vertices} vertices...")
    edges = []

    # Create a connected graph with some structure
    # Chain
    for i in range(num_vertices - 1):
        edges.append((i, i + 1))

    # Add some cross edges
    for i in range(0, num_vertices - 1, 10):
        if i + 20 < num_vertices:
            edges.append((i, i + 20))

    # Add some back edges
    for i in range(10, num_vertices, 15):
        if i - 5 >= 0:
            edges.append((i, i - 5))

    # Write to file
    with open(filename, 'w') as f:
        for src, dst in edges:
            f.write(f"{src} {dst}\n")

    print(f"  Generated {len(edges)} edges")
    return len(edges)


def generate_large_graph(filename, num_vertices=10000):
    """Generate a larger test graph"""
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

    # Add some random edges for complexity
    import random
    random.seed(42)
    for _ in range(num_vertices // 10):
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


def parse_time_from_output(output):
    """Parse execution time from tool output"""
    for line in output.split('\n'):
        if "Completed in" in line:
            # Extract time like "Completed in 123.45µs" or "Completed in 1.23ms"
            parts = line.split("Completed in")[1].strip()
            time_str = parts.split()[0]

            if 'µs' in time_str or 'μs' in time_str:
                return float(time_str.replace('µs', '').replace('μs', '')) / 1000  # Convert to ms
            elif 'ms' in time_str:
                return float(time_str.replace('ms', ''))
            elif 's' in time_str and 'ms' not in time_str:
                return float(time_str.replace('s', '')) * 1000  # Convert to ms

    return None


def compare_results(file1, file2):
    """Compare two result files to ensure they're equivalent"""
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = [line for line in f1 if not line.startswith('#')]
        lines2 = [line for line in f2 if not line.startswith('#')]

        if len(lines1) != len(lines2):
            return False

        for l1, l2 in zip(lines1, lines2):
            parts1 = l1.strip().split()
            parts2 = l2.strip().split()

            if len(parts1) != len(parts2):
                return False

            # Compare vertex IDs (should be exact)
            if parts1[0] != parts2[0]:
                return False

            # Compare values (allow small floating point differences)
            try:
                val1 = float(parts1[1])
                val2 = float(parts2[1])
                if abs(val1 - val2) > 1e-6:
                    return False
            except ValueError:
                if parts1[1] != parts2[1]:
                    return False

        return True


def run_bfs_benchmark(graph_file, graph_name, source=0):
    """Run BFS benchmark with different thread counts"""
    print(f"\n--- BFS Benchmark: {graph_name} ---")

    results = {'thread_counts': [], 'times': [], 'speedups': []}

    # Sequential
    seq_out = f"{OUTPUT_DIR}/bfs_{graph_name}_seq.txt"
    cmd_seq = f"{TOOL_PATH} bfs --input {graph_file} --source {source} --mode seq --out {seq_out}"
    print("Running sequential BFS...")
    result_seq = subprocess.run(cmd_seq, shell=True, capture_output=True, text=True)
    time_seq = parse_time_from_output(result_seq.stdout)
    print(f"  Sequential: {time_seq:.3f} ms")

    # Parallel with different thread counts
    for threads in THREAD_COUNTS:
        par_out = f"{OUTPUT_DIR}/bfs_{graph_name}_par_t{threads}.txt"
        cmd_par = f"{TOOL_PATH} bfs --input {graph_file} --source {source} --mode par --threads {threads} --out {par_out}"
        print(f"Running parallel BFS with {threads} threads...")
        result_par = subprocess.run(cmd_par, shell=True, capture_output=True, text=True)
        time_par = parse_time_from_output(result_par.stdout)

        speedup = time_seq / time_par if time_par > 0 else 0
        results['thread_counts'].append(threads)
        results['times'].append(time_par)
        results['speedups'].append(speedup)

        print(f"  Parallel ({threads} threads): {time_par:.3f} ms, Speedup: {speedup:.2f}x")

        # Compare with sequential
        match = compare_results(seq_out, par_out)
        if not match:
            print(f"  WARNING: Results don't match for {threads} threads!")

    results['seq_time'] = time_seq
    return results


def run_wcc_benchmark(graph_file, graph_name):
    """Run WCC benchmark with different thread counts"""
    print(f"\n--- WCC Benchmark: {graph_name} ---")

    results = {'thread_counts': [], 'times': [], 'speedups': []}

    # Sequential
    seq_out = f"{OUTPUT_DIR}/wcc_{graph_name}_seq.txt"
    cmd_seq = f"{TOOL_PATH} wcc --input {graph_file} --mode seq --out {seq_out}"
    print("Running sequential WCC...")
    result_seq = subprocess.run(cmd_seq, shell=True, capture_output=True, text=True)
    time_seq = parse_time_from_output(result_seq.stdout)
    print(f"  Sequential: {time_seq:.3f} ms")

    # Parallel with different thread counts
    for threads in THREAD_COUNTS:
        par_out = f"{OUTPUT_DIR}/wcc_{graph_name}_par_t{threads}.txt"
        cmd_par = f"{TOOL_PATH} wcc --input {graph_file} --mode par --threads {threads} --out {par_out}"
        print(f"Running parallel WCC with {threads} threads...")
        result_par = subprocess.run(cmd_par, shell=True, capture_output=True, text=True)
        time_par = parse_time_from_output(result_par.stdout)

        speedup = time_seq / time_par if time_par > 0 else 0
        results['thread_counts'].append(threads)
        results['times'].append(time_par)
        results['speedups'].append(speedup)

        print(f"  Parallel ({threads} threads): {time_par:.3f} ms, Speedup: {speedup:.2f}x")

    results['seq_time'] = time_seq
    return results


def run_pagerank_benchmark(graph_file, graph_name, alpha=0.85, iters=50, eps=1e-10):
    """Run PageRank benchmark with different thread counts"""
    print(f"\n--- PageRank Benchmark: {graph_name} ---")

    results = {'thread_counts': [], 'times': [], 'speedups': []}

    # Sequential
    seq_out = f"{OUTPUT_DIR}/pr_{graph_name}_seq.txt"
    cmd_seq = f"{TOOL_PATH} pagerank --input {graph_file} --mode seq --out {seq_out} --alpha {alpha} --iters {iters} --eps {eps}"
    print("Running sequential PageRank...")
    result_seq = subprocess.run(cmd_seq, shell=True, capture_output=True, text=True)
    time_seq = parse_time_from_output(result_seq.stdout)
    print(f"  Sequential: {time_seq:.3f} ms")

    # Parallel with different thread counts
    for threads in THREAD_COUNTS:
        par_out = f"{OUTPUT_DIR}/pr_{graph_name}_par_t{threads}.txt"
        cmd_par = f"{TOOL_PATH} pagerank --input {graph_file} --mode par --threads {threads} --out {par_out} --alpha {alpha} --iters {iters} --eps {eps}"
        print(f"Running parallel PageRank with {threads} threads...")
        result_par = subprocess.run(cmd_par, shell=True, capture_output=True, text=True)
        time_par = parse_time_from_output(result_par.stdout)

        speedup = time_seq / time_par if time_par > 0 else 0
        results['thread_counts'].append(threads)
        results['times'].append(time_par)
        results['speedups'].append(speedup)

        print(f"  Parallel ({threads} threads): {time_par:.3f} ms, Speedup: {speedup:.2f}x")

        # Compare with sequential
        match = compare_results(seq_out, par_out)
        if not match:
            print(f"  WARNING: Results don't match for {threads} threads!")

    results['seq_time'] = time_seq
    return results


def plot_seq_vs_par_comparison(results, threads_used=4):
    """Plot sequential vs parallel comparison (using specified thread count)"""
    print(f"\nGenerating sequential vs parallel comparison plot (using {threads_used} threads)...")
    
    algorithms = ['BFS', 'WCC', 'PageRank']
    
    # Find index for the specified thread count
    thread_index = THREAD_COUNTS.index(threads_used)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Sequential vs Parallel Performance Comparison ({threads_used} Threads)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Execution Time Comparison (Small Graph)
    ax1 = axes[0, 0]
    small_seq = [results['small']['bfs']['seq_time'], 
                 results['small']['wcc']['seq_time'],
                 results['small']['pagerank']['seq_time']]
    small_par = [results['small']['bfs']['times'][thread_index],  # Use specific thread count
                 results['small']['wcc']['times'][thread_index],
                 results['small']['pagerank']['times'][thread_index]]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, small_seq, width, label='Sequential', color='skyblue')
    bars2 = ax1.bar(x + width/2, small_par, width, label=f'Parallel ({threads_used}t)', color='orange')
    
    ax1.set_ylabel('Time (ms)', fontweight='bold')
    ax1.set_title(f'Execution Time - Small Graph (1000 vertices)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Execution Time Comparison (Large Graph)
    ax2 = axes[0, 1]
    large_seq = [results['large']['bfs']['seq_time'],
                 results['large']['wcc']['seq_time'],
                 results['large']['pagerank']['seq_time']]
    large_par = [results['large']['bfs']['times'][thread_index],  # Use specific thread count
                 results['large']['wcc']['times'][thread_index],
                 results['large']['pagerank']['times'][thread_index]]
    
    bars1 = ax2.bar(x - width/2, large_seq, width, label='Sequential', color='skyblue')
    bars2 = ax2.bar(x + width/2, large_par, width, label=f'Parallel ({threads_used}t)', color='orange')
    
    ax2.set_ylabel('Time (ms)', fontweight='bold')
    ax2.set_title(f'Execution Time - Large Graph (5M vertices)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Speedup Comparison
    ax3 = axes[1, 0]
    small_speedup = [results['small']['bfs']['speedups'][thread_index],  # Use specific thread count
                     results['small']['wcc']['speedups'][thread_index],
                     results['small']['pagerank']['speedups'][thread_index]]
    large_speedup = [results['large']['bfs']['speedups'][thread_index],  # Use specific thread count
                     results['large']['wcc']['speedups'][thread_index],
                     results['large']['pagerank']['speedups'][thread_index]]
    
    bars1 = ax3.bar(x - width/2, small_speedup, width, label='Small Graph', color='lightgreen')
    bars2 = ax3.bar(x + width/2, large_speedup, width, label='Large Graph', color='salmon')
    
    ax3.set_ylabel('Speedup (x)', fontweight='bold')
    ax3.set_title(f'Parallel Speedup ({threads_used} Threads)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms)
    ax3.legend()
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    table_data.append(['Algorithm', 'Graph', 'Seq (ms)', 'Par (ms)', 'Speedup'])
    
    for algo in ['bfs', 'wcc', 'pagerank']:
        for size in ['small', 'large']:
            data = results[size][algo]
            table_data.append([
                algo.upper(),
                size.capitalize(),
                f"{data['seq_time']:.2f}",
                f"{data['times'][thread_index]:.2f}",  # Use specific thread count
                f"{data['speedups'][thread_index]:.2f}x",  # Use specific thread count
            ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.15, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title(f'Summary Results ({threads_used} Threads)', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{OUTPUT_DIR}/benchmark_seq_vs_par.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Sequential vs parallel plot saved to {plot_file}")


def plot_thread_scaling(results):
    """Plot thread scaling analysis"""
    print("\nGenerating thread scaling plot...")

    algorithms = ['BFS', 'WCC', 'PageRank']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Thread Scaling Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    for col, (algo_name, algo_key) in enumerate(zip(algorithms, ['bfs', 'wcc', 'pagerank'])):
        # Plot 1: Small graph execution time
        ax1 = fig.add_subplot(gs[0, col])
        small_data = results['small'][algo_key]

        ax1.plot([0] + small_data['thread_counts'],
                 [small_data['seq_time']] + small_data['times'],
                 marker='o', linewidth=2, markersize=8, color=colors[col], label='Execution Time')
        ax1.axhline(y=small_data['seq_time'], color='gray', linestyle='--', alpha=0.5, label='Sequential')

        ax1.set_xlabel('Number of Threads', fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontweight='bold')
        ax1.set_title(f'{algo_name} - Small Graph (1000 vertices)', fontweight='bold')
        ax1.set_xticks([0] + THREAD_COUNTS)
        ax1.set_xticklabels(['Seq'] + [str(t) for t in THREAD_COUNTS])
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add value labels
        for i, (threads, time_val) in enumerate(zip([0] + small_data['thread_counts'],
                                                     [small_data['seq_time']] + small_data['times'])):
            ax1.annotate(f'{time_val:.2f}',
                        xy=(threads if threads > 0 else 0, time_val),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=8)

        # Plot 2: Large graph execution time
        ax2 = fig.add_subplot(gs[1, col])
        large_data = results['large'][algo_key]

        ax2.plot([0] + large_data['thread_counts'],
                 [large_data['seq_time']] + large_data['times'],
                 marker='s', linewidth=2, markersize=8, color=colors[col], label='Execution Time')
        ax2.axhline(y=large_data['seq_time'], color='gray', linestyle='--', alpha=0.5, label='Sequential')

        ax2.set_xlabel('Number of Threads', fontweight='bold')
        ax2.set_ylabel('Time (ms)', fontweight='bold')
        ax2.set_title(f'{algo_name} - Large Graph (5M vertices)', fontweight='bold')
        ax2.set_xticks([0] + THREAD_COUNTS)
        ax2.set_xticklabels(['Seq'] + [str(t) for t in THREAD_COUNTS])
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add value labels
        for i, (threads, time_val) in enumerate(zip([0] + large_data['thread_counts'],
                                                     [large_data['seq_time']] + large_data['times'])):
            ax2.annotate(f'{time_val:.2f}',
                        xy=(threads if threads > 0 else 0, time_val),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=8)

        # Plot 3: Speedup comparison
        ax3 = fig.add_subplot(gs[2, col])

        ax3.plot(small_data['thread_counts'], small_data['speedups'],
                marker='o', linewidth=2, markersize=8, label='Small Graph', color='#2ECC71')
        ax3.plot(large_data['thread_counts'], large_data['speedups'],
                marker='s', linewidth=2, markersize=8, label='Large Graph', color='#E74C3C')
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.3, label='No Speedup')

        # Ideal speedup line
        ax3.plot(THREAD_COUNTS, THREAD_COUNTS, 'k:', alpha=0.3, label='Ideal (Linear)')

        ax3.set_xlabel('Number of Threads', fontweight='bold')
        ax3.set_ylabel('Speedup (x)', fontweight='bold')
        ax3.set_title(f'{algo_name} - Speedup Analysis', fontweight='bold')
        ax3.set_xticks(THREAD_COUNTS)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Add value labels
        for threads, speedup_small, speedup_large in zip(small_data['thread_counts'],
                                                          small_data['speedups'],
                                                          large_data['speedups']):
            ax3.annotate(f'{speedup_small:.2f}x',
                        xy=(threads, speedup_small),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=7, color='#2ECC71')
            ax3.annotate(f'{speedup_large:.2f}x',
                        xy=(threads, speedup_large),
                        xytext=(0, -12), textcoords='offset points',
                        ha='center', fontsize=7, color='#E74C3C')

    # Save plot
    plot_file = f"{OUTPUT_DIR}/benchmark_thread_scaling.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Thread scaling plot saved to {plot_file}")


def print_summary_table(results):
    """Print a summary table of all results"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for graph_size in ['small', 'large']:
        print(f"\n{graph_size.upper()} GRAPH:")
        print("-" * 80)
        print(f"{'Algorithm':<12} {'Sequential':<12} {'1 Thread':<12} {'2 Threads':<12} {'4 Threads':<12} {'8 Threads':<12}")
        print("-" * 80)

        for algo in ['bfs', 'wcc', 'pagerank']:
            data = results[graph_size][algo]
            row = f"{algo.upper():<12} {data['seq_time']:>8.2f} ms"
            for time_val in data['times']:
                row += f" {time_val:>8.2f} ms"
            print(row)

        print("\nSPEEDUPS:")
        print("-" * 80)
        print(f"{'Algorithm':<12} {'1 Thread':<12} {'2 Threads':<12} {'4 Threads':<12} {'8 Threads':<12}")
        print("-" * 80)

        for algo in ['bfs', 'wcc', 'pagerank']:
            data = results[graph_size][algo]
            row = f"{algo.upper():<12}"
            for speedup in data['speedups']:
                row += f" {speedup:>8.2f}x"
            print(row)

    print("\n" + "="*80)


def main():
    """Main benchmark execution"""
    print("=" * 80)
    print("Graph Algorithm Benchmark Suite - Thread Scaling Analysis")
    print("=" * 80)

    # Check if tool exists
    if not os.path.exists(TOOL_PATH):
        print(f"Error: Tool not found at {TOOL_PATH}")
        print("Please build the project first with: cargo build --release")
        return

    # Generate test graphs
    small_graph = f"{GRAPHS_DIR}/small_graph.txt"
    large_graph = f"{GRAPHS_DIR}/large_graph.txt"

    generate_small_graph(small_graph, num_vertices=1000)
    generate_large_graph(large_graph, num_vertices=5000000)

    # Run benchmarks
    results = {
        'small': {},
        'large': {}
    }

    # Small graph benchmarks
    results['small']['bfs'] = run_bfs_benchmark(small_graph, 'small', source=0)
    results['small']['wcc'] = run_wcc_benchmark(small_graph, 'small')
    results['small']['pagerank'] = run_pagerank_benchmark(small_graph, 'small')

    # Large graph benchmarks
    results['large']['bfs'] = run_bfs_benchmark(large_graph, 'large', source=0)
    results['large']['wcc'] = run_wcc_benchmark(large_graph, 'large')
    results['large']['pagerank'] = run_pagerank_benchmark(large_graph, 'large')

    # Print summary table
    print_summary_table(results)

    # Plot results
    plot_seq_vs_par_comparison(results, threads_used=16)
    plot_thread_scaling(results)

    plt.show()

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print(f"Results saved to {OUTPUT_DIR}/")
    print("  - benchmark_seq_vs_par.png: Sequential vs Parallel comparison")
    print("  - benchmark_thread_scaling.png: Thread scaling analysis")
    print("=" * 80)


if __name__ == "__main__":
    main()
