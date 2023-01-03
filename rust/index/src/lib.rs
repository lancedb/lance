use std::io::Result;

pub mod ann;

/// Generic index traits
pub trait Index {
    fn prefetch(&mut self) -> Result<()>;

    /// Indexed columns
    fn columns(&self) -> &[String];
}
