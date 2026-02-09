use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use super::CSRGraph;

// Add this new struct for bidirectional CSR
#[derive(Debug, Clone)]
pub struct BidirectionalCSRGraph {
    /// Forward graph (original edges)
    forward: CSRGraph,
    /// Reverse graph (transposed edges)
    reverse: CSRGraph,
}

impl BidirectionalCSRGraph {
    /// Create a bidirectional graph from an edge list file
    pub fn from_edge_list<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        
        let mut edge_list = Vec::new();
        let mut max_vertex = 0;
        
        // Read all edges
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            
            let src: usize = parts[0].parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let dst: usize = parts[1].parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            
            max_vertex = max_vertex.max(src).max(dst);
            edge_list.push((src, dst));
        }
        
        let num_vertices = max_vertex + 1;
        
        // Build forward graph
        let forward = CSRGraph::build_csr(num_vertices, edge_list.clone())?;
        
        // Build reverse graph (swap src and dst)
        let reverse_edges: Vec<(usize, usize)> = edge_list
            .into_iter()
            .map(|(src, dst)| (dst, src))
            .collect();
        let reverse = CSRGraph::build_csr(num_vertices, reverse_edges)?;
        
        Ok(BidirectionalCSRGraph { forward, reverse })
    }
    
    /// Get the number of vertices
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.forward.num_vertices()
    }
    
    /// Get the number of edges
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.forward.num_edges()
    }
    
    /// Get outgoing neighbors of a vertex
    #[inline]
    pub fn out_neighbors(&self, vertex: usize) -> &[usize] {
        self.forward.neighbors(vertex)
    }
    
    /// Get incoming neighbors of a vertex (reverse edges)
    #[inline]
    pub fn in_neighbors(&self, vertex: usize) -> &[usize] {
        self.reverse.neighbors(vertex)
    }
    
    /// Get both incoming and outgoing neighbors
    pub fn all_neighbors(&self, vertex: usize) -> impl Iterator<Item = usize> + '_ {
        self.out_neighbors(vertex)
            .iter()
            .chain(self.in_neighbors(vertex).iter())
            .copied()
    }
    
    /// Get the forward (original) graph
    pub fn forward_graph(&self) -> &CSRGraph {
        &self.forward
    }
    
    /// Get the reverse (transposed) graph
    pub fn reverse_graph(&self) -> &CSRGraph {
        &self.reverse
    }
}

impl std::fmt::Display for BidirectionalCSRGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Bidirectional CSR Graph:")?;
        writeln!(f, "  Vertices: {}", self.num_vertices())?;
        writeln!(f, "  Edges: {}", self.num_edges())?;
        writeln!(f, "  Average out-degree: {:.2}", 
            self.num_edges() as f64 / self.num_vertices() as f64)?;
        Ok(())
    }
}