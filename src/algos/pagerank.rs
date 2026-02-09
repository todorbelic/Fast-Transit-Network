use crate::graph::BidirectionalCSRGraph;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Sequential PageRank implementation
/// Returns a vector of PageRank scores for each vertex
pub fn pagerank_sequential(
    graph: &BidirectionalCSRGraph,
    damping: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Vec<f64> {
    let num_vertices = graph.num_vertices();
    
    // Initialize: all vertices have equal rank
    let mut rank = vec![1.0 / num_vertices as f64; num_vertices];
    let mut new_rank = vec![0.0; num_vertices];
    
    let teleport = (1.0 - damping) / num_vertices as f64;
    
    for _iteration in 0..max_iterations {
        // Initialize all ranks with teleport probability
        for v in 0..num_vertices {
            new_rank[v] = teleport;
        }
        
        // Pull-based: each vertex collects rank from predecessors
        for v in 0..num_vertices {
            // Iterate over all incoming neighbors (predecessors)
            for &u in graph.in_neighbors(v) {
                let out_deg = graph.out_neighbors(u).len();
                if out_deg > 0 {
                    new_rank[v] += damping * rank[u] / out_deg as f64;
                }
            }
        }
        
        // Handle dangling nodes (vertices with no outgoing edges)
        let mut dangling_sum = 0.0;
        for u in 0..num_vertices {
            if graph.out_neighbors(u).is_empty() {
                dangling_sum += rank[u];
            }
        }
        
        // Distribute dangling rank to all vertices
        let dangling_contrib = damping * dangling_sum / num_vertices as f64;
        for v in 0..num_vertices {
            new_rank[v] += dangling_contrib;
        }
        
        // Check convergence (L1 norm)
        let mut diff = 0.0;
        for v in 0..num_vertices {
            diff += (new_rank[v] - rank[v]).abs();
        }
        
        // Swap arrays
        std::mem::swap(&mut rank, &mut new_rank);
        
        if diff < tolerance {
            break;
        }
    }
    
    rank
}

/// Parallel PageRank implementation using pull-based approach
/// Returns a vector of PageRank scores for each vertex
pub fn pagerank_parallel(
    graph: &BidirectionalCSRGraph,
    damping: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Vec<f64> {
    let num_vertices = graph.num_vertices();
    
    // Initialize: all vertices have equal rank
    let mut rank = vec![1.0 / num_vertices as f64; num_vertices];
    let mut new_rank = vec![0.0; num_vertices];
    
    let teleport = (1.0 - damping) / num_vertices as f64;
    
    for _iteration in 0..max_iterations {
        // Pull-based: each vertex independently computes its new rank
        new_rank.par_iter_mut().enumerate().for_each(|(v, new_rank_v)| {
            // Start with teleport probability
            *new_rank_v = teleport;
            
            // Sum contributions from all incoming neighbors
            for &u in graph.in_neighbors(v) {
                let out_deg = graph.out_neighbors(u).len();
                if out_deg > 0 {
                    *new_rank_v += damping * rank[u] / out_deg as f64;
                }
            }
        });
        
        // Handle dangling nodes
        let dangling_sum: f64 = (0..num_vertices)
            .into_par_iter()
            .filter(|&u| graph.out_neighbors(u).is_empty())
            .map(|u| rank[u])
            .sum();
        
        let dangling_contrib = damping * dangling_sum / num_vertices as f64;
        
        // Add dangling contribution to all vertices
        new_rank.par_iter_mut().for_each(|new_rank_v| {
            *new_rank_v += dangling_contrib;
        });
        
        // Check convergence (L1 norm) - parallel reduction
        let diff: f64 = new_rank
            .par_iter()
            .zip(rank.par_iter())
            .map(|(new, old)| (new - old).abs())
            .sum();
        
        // Swap arrays
        std::mem::swap(&mut rank, &mut new_rank);
        
        if diff < tolerance {
            break;
        }
    }
    
    rank
}

/// Helper function to normalize PageRank scores to sum to 1.0
pub fn normalize_pagerank(ranks: &mut [f64]) {
    let sum: f64 = ranks.iter().sum();
    if sum > 0.0 {
        for rank in ranks.iter_mut() {
            *rank /= sum;
        }
    }
}

/// Helper function to check if two PageRank results are approximately equal
pub fn pagerank_approx_equal(rank1: &[f64], rank2: &[f64], epsilon: f64) -> bool {
    if rank1.len() != rank2.len() {
        return false;
    }
    
    rank1.iter()
        .zip(rank2.iter())
        .all(|(r1, r2)| (r1 - r2).abs() < epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::BidirectionalCSRGraph;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_graph(edges: &[(usize, usize)]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for (src, dst) in edges {
            writeln!(file, "{} {}", src, dst).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_pagerank_simple_chain() {
        // Simple chain: 0 → 1 → 2
        let edges = vec![
            (0, 1),
            (1, 2)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let ranks = pagerank_sequential(&graph, 0.85, 100, 1e-6);
        
        // Verify sum is approximately 1.0
        let sum: f64 = ranks.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "PageRank scores should sum to 1.0");
        
        // In a chain, later vertices should have higher rank
        assert!(ranks[2] > ranks[1]);
        assert!(ranks[1] > ranks[0]);
    }

    #[test]
    fn test_pagerank_cycle() {
        // Cycle: 0 → 1 → 2 → 0
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 0)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let ranks = pagerank_sequential(&graph, 0.85, 100, 1e-6);
        
        // In a symmetric cycle, all vertices should have equal rank
        let expected = 1.0 / 3.0;
        for rank in &ranks {
            assert!((rank - expected).abs() < 1e-4, 
                "In a cycle, all vertices should have equal rank");
        }
    }

    #[test]
    fn test_pagerank_parallel_vs_sequential_simple() {
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 0)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let seq_ranks = pagerank_sequential(&graph, 0.85, 100, 1e-6);
        let par_ranks = pagerank_parallel(&graph, 0.85, 100, 1e-6);
        
        assert!(pagerank_approx_equal(&seq_ranks, &par_ranks, 1e-6),
            "Sequential and parallel PageRank should produce nearly identical results");
    }

    #[test]
    fn test_pagerank_parallel_vs_sequential_complex() {
        let edges = vec![
            (0, 1), (0, 2),
            (1, 2), (1, 3),
            (2, 3), (2, 4),
            (3, 4), (3, 0),
            (4, 0), (4, 1)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let seq_ranks = pagerank_sequential(&graph, 0.85, 100, 1e-6);
        let par_ranks = pagerank_parallel(&graph, 0.85, 100, 1e-6);
        
        assert!(pagerank_approx_equal(&seq_ranks, &par_ranks, 1e-6),
            "Sequential and parallel PageRank should match on complex graphs");
    }

}