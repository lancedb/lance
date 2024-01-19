use std::ops::Range;

pub struct RowAddress(u64);

impl RowAddress {
    pub const FRAGMENT_SIZE: u64 = 1 << 32;
    // A fragment id that will never be used
    pub const TOMBSTONE_FRAG: u32 = 0xffffffff;
    // A row id that will never be used
    pub const TOMBSTONE_ROW: u64 = 0xffffffffffffffff;

    pub fn new_from_id(row_id: u64) -> Self {
        Self(row_id)
    }

    pub fn new_from_parts(fragment_id: u32, row_id: u32) -> Self {
        Self(((fragment_id as u64) << 32) | row_id as u64)
    }

    pub fn first_row(fragment_id: u32) -> Self {
        Self::new_from_parts(fragment_id, 0)
    }

    pub fn fragment_range(fragment_id: u32) -> Range<u64> {
        u64::from(Self::first_row(fragment_id))..u64::from(Self::first_row(fragment_id + 1))
    }

    pub fn fragment_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    pub fn row_id(&self) -> u32 {
        self.0 as u32
    }
}

impl From<RowAddress> for u64 {
    fn from(row_id: RowAddress) -> Self {
        row_id.0
    }
}

impl std::fmt::Debug for RowAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self) // use Display
    }
}

impl std::fmt::Display for RowAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.fragment_id(), self.row_id())
    }
}
