use crate::graph::BidirectionalCSRGraph;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI32, Ordering};
use rayon::prelude::*;

/// Sequential WCC using BFS
/// Returns a vector where component[v] is the component ID for vertex v
/// Component IDs are assigned in order of discovery (0, 1, 2, ...)
pub fn wcc_sequential(graph: &BidirectionalCSRGraph) -> Vec<usize> {
    let num_vertices = graph.num_vertices();
    let mut component = vec![usize::MAX; num_vertices];  // usize::MAX means unvisited
    let mut component_id = 0;
    
    for start_vertex in 0..num_vertices {
        if component[start_vertex] == usize::MAX {
            // Found a new component, do BFS
            let mut queue = VecDeque::new();
            queue.push_back(start_vertex);
            component[start_vertex] = component_id;
            
            while let Some(current) = queue.pop_front() {
                // Explore both outgoing and incoming neighbors
                for neighbor in graph.all_neighbors(current) {
                    if component[neighbor] == usize::MAX {
                        component[neighbor] = component_id;
                        queue.push_back(neighbor);
                    }
                }
            }
            
            component_id += 1;
        }
    }
    
    component
}

/// Parallel WCC using BFS
/// Multiple threads work on different components simultaneously
pub fn wcc_parallel(graph: &BidirectionalCSRGraph) -> Vec<usize> {
    let num_vertices = graph.num_vertices();
    
    // Use atomic integers for thread-safe component assignment
    let component: Vec<AtomicI32> = (0..num_vertices)
        .map(|_| AtomicI32::new(-1))  // -1 means unvisited
        .collect();
    
    // Process all vertices in parallel, each trying to start a new component
    (0..num_vertices).into_par_iter().for_each(|start_vertex| {
        // Try to claim this vertex as a component root
        if component[start_vertex]
            .compare_exchange(-1, start_vertex as i32, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // Successfully claimed! Do BFS from here
            let comp_id = start_vertex as i32;
            let mut queue = VecDeque::new();
            queue.push_back(start_vertex);
            
            while let Some(current) = queue.pop_front() {
                // Explore all neighbors (both directions)
                for neighbor in graph.all_neighbors(current) {
                    // Try to claim this neighbor for our component
                    if component[neighbor]
                        .compare_exchange(-1, comp_id, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                    {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    });
    
    // Convert atomic results to regular Vec<usize>
    // Note: component IDs in parallel version are vertex IDs, not sequential
    component
        .into_iter()
        .map(|c| c.into_inner() as usize)
        .collect()
}


/// Helper function to check if two component assignments are equivalent
/// (same grouping of vertices, possibly different IDs)
pub fn components_equivalent(comp1: &[usize], comp2: &[usize]) -> bool {
    if comp1.len() != comp2.len() {
        return false;
    }
    
    use std::collections::HashMap;
    
    // Build a mapping from comp1 IDs to comp2 IDs
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    
    for (&c1, &c2) in comp1.iter().zip(comp2.iter()) {
        if let Some(&mapped_c2) = mapping.get(&c1) {
            if mapped_c2 != c2 {
                return false;  // Inconsistent mapping
            }
        } else {
            mapping.insert(c1, c2);
        }
    }
    
    // Check reverse mapping (ensure one-to-one correspondence)
    let mut reverse_mapping: HashMap<usize, usize> = HashMap::new();
    for (&c1, &c2) in mapping.iter() {
        if let Some(&mapped_c1) = reverse_mapping.get(&c2) {
            if mapped_c1 != c1 {
                return false;
            }
        } else {
            reverse_mapping.insert(c2, c1);
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_wcc_single_component() {
        // Single strongly connected component
        // 0 → 1 → 2
        //     ↑___↓
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 1)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let components = wcc_sequential(&graph);
        
        // All vertices should be in the same component
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);
    }

    #[test]
    fn test_wcc_two_components() {
        // Two separate components
        // Component 1: 0 → 1
        // Component 2: 2 → 3
        let edges = vec![
            (0, 1),
            (2, 3)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let components = wcc_sequential(&graph);
        
        // Vertices 0 and 1 should be in same component
        assert_eq!(components[0], components[1]);
        
        // Vertices 2 and 3 should be in same component
        assert_eq!(components[2], components[3]);
        
        // But different from component {0,1}
        assert_ne!(components[0], components[2]);
    }

    #[test]
    fn test_wcc_weakly_connected() {
        // Weakly connected but not strongly connected
        // 0 → 1 → 2
        // (no back edges)
        let edges = vec![
            (0, 1),
            (1, 2)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let components = wcc_sequential(&graph);
        
        // All should be in same component (weakly connected)
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);
    }

    #[test]
    fn test_wcc_three_components() {
        // Three separate components
        // 0 → 1, 2 → 3, 4 → 5
        let edges = vec![
            (0, 1),
            (2, 3),
            (4, 5)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let components = wcc_sequential(&graph);
        
        // Each pair should be in same component
        assert_eq!(components[0], components[1]);
        assert_eq!(components[2], components[3]);
        assert_eq!(components[4], components[5]);
        
        // But all three components should be different
        assert_ne!(components[0], components[2]);
        assert_ne!(components[0], components[4]);
        assert_ne!(components[2], components[4]);
    }

    #[test]
    fn test_wcc_isolated_vertex() {
        // Graph with isolated vertex
        // 0 → 1, 2 (isolated)
        let edges = vec![
            (0, 1),
            (2, 2)  // Self-loop to ensure vertex 2 exists
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let components = wcc_sequential(&graph);
        
        assert_eq!(components[0], components[1]);
        assert_ne!(components[0], components[2]);
    }

    #[test]
    fn test_wcc_complex_graph() {
        // More complex graph
        //   0 → 1 → 2
        //   ↓       ↓
        //   3 ← 4 ← 5
        //
        //   6 → 7
        let edges = vec![
            (0, 1), (1, 2), (2, 5),
            (0, 3), (4, 3), (5, 4),
            (6, 7)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let components = wcc_sequential(&graph);
        
        // First component: 0,1,2,3,4,5
        assert_eq!(components[0], components[1]);
        assert_eq!(components[0], components[2]);
        assert_eq!(components[0], components[3]);
        assert_eq!(components[0], components[4]);
        assert_eq!(components[0], components[5]);
        
        // Second component: 6,7
        assert_eq!(components[6], components[7]);
        
        // Different components
        assert_ne!(components[0], components[6]);
    }

    #[test]
    fn test_wcc_parallel_vs_sequential_simple() {
        // Simple graph
        let edges = vec![
            (0, 1),
            (2, 3)
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let seq_components = wcc_sequential(&graph);
        let par_components = wcc_parallel(&graph);
        
        assert!(components_equivalent(&seq_components, &par_components),
            "Sequential and parallel should produce equivalent components");
    }

    #[test]
    fn test_wcc_parallel_vs_sequential_complex() {
        // Complex graph with multiple components
        let edges = vec![
            (0, 1), (1, 2), (2, 0),  // Component 1
            (3, 4), (4, 5),          // Component 2
            (6, 7), (7, 8), (8, 6),  // Component 3
        ];
        let file = create_test_graph(&edges);
        let graph = BidirectionalCSRGraph::from_edge_list(file.path()).unwrap();
        
        let seq_components = wcc_sequential(&graph);
        let par_components = wcc_parallel(&graph);
        
        assert!(components_equivalent(&seq_components, &par_components),
            "Sequential and parallel should produce equivalent components");
    }
}