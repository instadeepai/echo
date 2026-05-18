/// Metadata for a single array in the flattened pytree
#[derive(Debug, Clone)]
pub struct ArraySpec {
    pub shape: Vec<usize>,
    pub dtype_size: usize, // bytes per element
}

impl ArraySpec {
    pub fn new(shape: Vec<usize>, dtype_size: usize) -> Self {
        Self { shape, dtype_size }
    }

    /// Total bytes for one array. Caller is responsible for ensuring the
    /// shape*dtype_size product fits in `usize`; we wrap on overflow.
    pub fn num_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype_size
    }
}
