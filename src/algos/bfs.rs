use crate::graph::CSRGraph;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI32, Ordering};
use rayon::prelude::*;

/// Sequential BFS from a source vertex
/// Returns a vector of distances where distances[v] is the shortest distance from source to v
/// Returns -1 for unreachable vertices
pub fn bfs_sequential(graph: &CSRGraph, source: usize) -> Vec<i32> {
    let num_vertices = graph.num_vertices();
    
    // Initialize distances to -1 (unvisited)
    let mut distances = vec![-1; num_vertices];
    
    // Mark source as visited with distance 0
    distances[source] = 0;
    
    // Create a queue and add source
    let mut queue = VecDeque::new();
    queue.push_back(source);
    
    // Process vertices level by level
    while let Some(current) = queue.pop_front() {
        let current_dist = distances[current];
        
        // Explore all neighbors
        for &neighbor in graph.neighbors(current) {
            // If not visited yet
            if distances[neighbor] == -1 {
                distances[neighbor] = current_dist + 1;
                queue.push_back(neighbor);
            }
        }
    }
    
    distances
}

/// Parallel BFS using level-synchronous approach
/// Returns a vector of distances where distances[v] is the shortest distance from source to v
/// Returns -1 for unreachable vertices
pub fn bfs_parallel(graph: &CSRGraph, source: usize) -> Vec<i32> {
    let num_vertices = graph.num_vertices();
    
    // Use atomic integers for thread-safe distance updates
    let distances: Vec<AtomicI32> = (0..num_vertices)
        .map(|_| AtomicI32::new(-1))
        .collect();
    
    // Initialize source
    distances[source].store(0, Ordering::Relaxed);
    
    // Current frontier (vertices at current level)
    let mut current_frontier = vec![source];
    let mut level = 0;
    
    // Process level by level
    while !current_frontier.is_empty() {
        // Next frontier will hold vertices at next level
        let next_frontier: Vec<usize> = current_frontier
            .par_iter()  // Process current frontier in parallel
            .flat_map(|&vertex| {
                // Collect neighbors that we successfully claim
                let mut local_next = Vec::new();
                
                for &neighbor in graph.neighbors(vertex) {
                    // Try to claim this neighbor atomically
                    // compare_exchange: if current value is -1, set it to level + 1
                    if distances[neighbor]
                        .compare_exchange(
                            -1,
                            level + 1,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        // Successfully claimed this neighbor
                        local_next.push(neighbor);
                    }
                }
                
                local_next
            })
            .collect();
        
        current_frontier = next_frontier;
        level += 1;
    }
    
    // Convert atomic results to regular Vec<i32>
    distances
        .into_iter()
        .map(|d| d.into_inner())
        .collect()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::graph::CSRGraph;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper function to create a temporary graph file
    fn create_test_graph(edges: &[(usize, usize)]) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for (src, dst) in edges {
            writeln!(file, "{} {}", src, dst).unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    pub fn test_bfs_simple_path() {
        // Graph: 0 → 1 → 2 → 3
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let file = create_test_graph(&edges);
        let graph = CSRGraph::from_edge_list(file.path()).unwrap();
        
        let distances = bfs_sequential(&graph, 0);
        
        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 1);
        assert_eq!(distances[2], 2);
        assert_eq!(distances[3], 3);
    }

    #[test]
    fn test_bfs_parallel_vs_sequential_complex() {
        // Complex graph with multiple paths
        //   0 → 1 → 3 → 5
        //   ↓   ↓   ↑   ↑
        //   2 → 4 → 6 → 7
        let edges = vec![
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 4),
            (3, 5),
            (4, 6),
            (6, 3), (6, 7),
            (7, 5)
        ];
        let file = create_test_graph(&edges);
        let graph = CSRGraph::from_edge_list(file.path()).unwrap();
        
        let seq_distances = bfs_sequential(&graph, 0);
        let par_distances = bfs_parallel(&graph, 0);
        
        assert_eq!(seq_distances, par_distances,
            "Sequential and parallel BFS should produce identical results");
    }
}