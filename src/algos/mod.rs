pub mod bfs;
pub mod wcc;
pub mod pagerank;

pub use bfs::{bfs_sequential, bfs_parallel};
pub use wcc::{wcc_sequential, wcc_parallel};
pub use pagerank::{pagerank_sequential, pagerank_parallel};