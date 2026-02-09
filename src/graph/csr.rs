use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CSRGraph {
    /// Number of vertices
    num_vertices: usize,
    /// Number of edges
    num_edges: usize,
    /// Row offsets - offsets[i] points to the start of neighbors for vertex i
    /// Length is num_vertices + 1
    offsets: Vec<usize>,
    /// Column indices - stores the actual neighbor vertices
    /// Length is num_edges
    edges: Vec<usize>,
}

impl CSRGraph {
    /// Create a new CSR graph from an edge list file
    /// File format: each line contains "source target" (0-indexed vertices)
    pub fn from_edge_list<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        
        let mut edge_list = Vec::new();
        let mut max_vertex = 0;
        
        // Read all edges
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            
            // Skip empty lines and comments
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
        let num_edges = edge_list.len();
        
        // Build CSR structure
        Self::build_csr(num_vertices, edge_list)
    }
    
    /// Build CSR structure from edge list
    pub fn build_csr(num_vertices: usize, mut edge_list: Vec<(usize, usize)>) -> std::io::Result<Self> {
        // Sort edges by source vertex
        edge_list.sort_unstable_by_key(|&(src, _)| src);
        
        let num_edges = edge_list.len();
        let mut offsets = vec![0; num_vertices + 1];
        let mut edges = Vec::with_capacity(num_edges);
        
        // Count out-degrees for each vertex
        for &(src, _) in &edge_list {
            offsets[src + 1] += 1;
        }
        
        // Convert counts to offsets (prefix sum)
        for i in 1..=num_vertices {
            offsets[i] += offsets[i - 1];
        }
        
        // Fill the edges array
        for (_, dst) in edge_list {
            edges.push(dst);
        }
        
        Ok(CSRGraph {
            num_vertices,
            num_edges,
            offsets,
            edges,
        })
    }
    
    /// Get the number of vertices
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }
    
    /// Get the number of edges
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }
    
    /// Get neighbors of a vertex
    #[inline]
    pub fn neighbors(&self, vertex: usize) -> &[usize] {
        let start = self.offsets[vertex];
        let end = self.offsets[vertex + 1];
        &self.edges[start..end]
    }
    
    /// Get out-degree of a vertex
    #[inline]
    pub fn out_degree(&self, vertex: usize) -> usize {
        self.offsets[vertex + 1] - self.offsets[vertex]
    }
    
    /// Iterate over all edges as (source, target) pairs
    pub fn iter_edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.num_vertices).flat_map(move |src| {
            self.neighbors(src).iter().map(move |&dst| (src, dst))
        })
    }
    
    /// Get access to raw CSR arrays (useful for parallel algorithms)
    pub fn raw_csr(&self) -> (&[usize], &[usize]) {
        (&self.offsets, &self.edges)
    }
}

// Display implementation for debugging
impl std::fmt::Display for CSRGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CSR Graph:")?;
        writeln!(f, "  Vertices: {}", self.num_vertices)?;
        writeln!(f, "  Edges: {}", self.num_edges)?;
        writeln!(f, "  Average degree: {:.2}", self.num_edges as f64 / self.num_vertices as f64)?;
        Ok(())
    }
}