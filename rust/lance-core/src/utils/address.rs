// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;

use crate::{Error, Result};

/// A row address encodes a fragment ID (upper 32 bits) and row offset (lower 32 bits).
///
/// ```
/// use lance_core::utils::address::RowAddress;
///
/// let addr = RowAddress::new_from_parts(5, 100);
/// assert_eq!(addr.fragment_id(), 5);
/// assert_eq!(addr.row_offset(), 100);
///
/// // Convert to/from u64
/// let raw: u64 = addr.into();
/// let addr2: RowAddress = raw.into();
/// assert_eq!(addr, addr2);
///
/// // Display format
/// assert_eq!(format!("{}", addr), "(5, 100)");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowAddress(u64);

/// A stable logical row address.
///
/// The upper 32 bits identify the logical fragment where the row was first
/// committed. The lower 32 bits identify the row's immutable slot within that
/// logical fragment. Unlike [`RowAddress`], this value does not identify the
/// row's current physical storage location.
///
/// ```
/// use lance_core::utils::address::LogicalRowAddress;
///
/// # fn main() -> lance_core::Result<()> {
/// let address = LogicalRowAddress::try_new_from_parts(5, 100)?;
/// assert_eq!(address.logical_fragment_id(), 5);
/// assert_eq!(address.immutable_slot(), 100);
/// assert_eq!(address.parts(), (5, 100));
/// assert_eq!(address.raw(), 0x0000_0005_0000_0064);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalRowAddress(u64);

impl LogicalRowAddress {
    /// Logical fragment ID reserved for invalid addresses.
    pub const INVALID_LOGICAL_FRAGMENT_ID: u32 = u32::MAX;
    /// Raw value reserved for an invalid address.
    pub const INVALID_RAW: u64 = u64::MAX;

    /// Creates a logical row address from its logical fragment and immutable slot.
    ///
    /// `u32::MAX` is reserved as the invalid logical fragment ID and is rejected.
    pub fn try_new_from_parts(logical_fragment_id: u32, immutable_slot: u32) -> Result<Self> {
        if logical_fragment_id == Self::INVALID_LOGICAL_FRAGMENT_ID {
            return Err(Error::invalid_input(format!(
                "logical_fragment_id is reserved for invalid logical row addresses: logical_fragment_id={}, immutable_slot={}",
                logical_fragment_id, immutable_slot
            )));
        }

        Ok(Self(
            ((logical_fragment_id as u64) << 32) | immutable_slot as u64,
        ))
    }

    /// Returns the encoded 64-bit wire value.
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Returns the logical fragment ID and immutable slot.
    pub const fn parts(self) -> (u32, u32) {
        (self.logical_fragment_id(), self.immutable_slot())
    }

    /// Returns the logical fragment ID.
    pub const fn logical_fragment_id(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Returns the immutable slot within the logical fragment.
    pub const fn immutable_slot(self) -> u32 {
        self.0 as u32
    }
}

impl TryFrom<u64> for LogicalRowAddress {
    type Error = Error;

    fn try_from(raw: u64) -> Result<Self> {
        let logical_fragment_id = (raw >> 32) as u32;
        if logical_fragment_id == Self::INVALID_LOGICAL_FRAGMENT_ID {
            return Err(Error::invalid_input(format!(
                "raw logical row address uses the reserved logical fragment ID: raw={raw:#018x}, logical_fragment_id={logical_fragment_id}"
            )));
        }

        Ok(Self(raw))
    }
}

impl From<LogicalRowAddress> for u64 {
    fn from(address: LogicalRowAddress) -> Self {
        address.raw()
    }
}

impl std::fmt::Debug for LogicalRowAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("LogicalRowAddress")
            .field(&self.logical_fragment_id())
            .field(&self.immutable_slot())
            .finish()
    }
}

impl std::fmt::Display for LogicalRowAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {})",
            self.logical_fragment_id(),
            self.immutable_slot()
        )
    }
}

impl RowAddress {
    pub const FRAGMENT_SIZE: u64 = 1 << 32;
    /// A fragment id that will never be used.
    pub const TOMBSTONE_FRAG: u32 = 0xffffffff;
    /// A row id that will never be used.
    pub const TOMBSTONE_ROW: u64 = 0xffffffffffffffff;

    pub fn new_from_u64(row_addr: u64) -> Self {
        Self(row_addr)
    }

    pub fn new_from_parts(fragment_id: u32, row_offset: u32) -> Self {
        Self(((fragment_id as u64) << 32) | row_offset as u64)
    }

    /// Returns the address for the first row of a fragment.
    pub fn first_row(fragment_id: u32) -> Self {
        Self::new_from_parts(fragment_id, 0)
    }

    /// Returns the range of u64 addresses for a given fragment.
    ///
    /// ```
    /// use lance_core::utils::address::RowAddress;
    ///
    /// let range = RowAddress::address_range(2);
    /// assert_eq!(range.start, 2 * RowAddress::FRAGMENT_SIZE);
    /// assert_eq!(range.end, 3 * RowAddress::FRAGMENT_SIZE);
    /// ```
    pub fn address_range(fragment_id: u32) -> Range<u64> {
        u64::from(Self::first_row(fragment_id))..u64::from(Self::first_row(fragment_id + 1))
    }

    pub fn fragment_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    pub fn row_offset(&self) -> u32 {
        self.0 as u32
    }
}

impl From<RowAddress> for u64 {
    fn from(row_addr: RowAddress) -> Self {
        row_addr.0
    }
}

impl From<u64> for RowAddress {
    fn from(row_addr: u64) -> Self {
        Self(row_addr)
    }
}

impl std::fmt::Debug for RowAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self) // use Display
    }
}

impl std::fmt::Display for RowAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.fragment_id(), self.row_offset())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_address() {
        // new_from_u64 (not in doctest)
        let addr = RowAddress::new_from_u64(0x0000_0001_0000_0002);
        assert_eq!(addr.fragment_id(), 1);
        assert_eq!(addr.row_offset(), 2);

        // address_range uses first_row internally (coverage)
        let range = RowAddress::address_range(3);
        assert_eq!(range.start, 3 * RowAddress::FRAGMENT_SIZE);

        // From impls with different values than doctest
        let addr2 = RowAddress::new_from_parts(7, 8);
        let raw: u64 = addr2.into();
        let addr3: RowAddress = raw.into();
        assert_eq!(addr2, addr3);

        // Debug format (doctest only tests Display)
        assert_eq!(format!("{:?}", addr), "(1, 2)");
    }

    #[test]
    fn test_logical_row_address_round_trip() {
        let address = LogicalRowAddress::try_new_from_parts(7, 8).unwrap();

        assert_eq!(address.logical_fragment_id(), 7);
        assert_eq!(address.immutable_slot(), 8);
        assert_eq!(address.parts(), (7, 8));
        assert_eq!(address.raw(), 0x0000_0007_0000_0008);

        let raw: u64 = address.into();
        let decoded = LogicalRowAddress::try_from(raw).unwrap();
        assert_eq!(decoded, address);
        assert_eq!(format!("{}", address), "(7, 8)");
        assert_eq!(format!("{:?}", address), "LogicalRowAddress(7, 8)");
    }

    #[test]
    fn test_logical_row_address_allows_maximum_slot() {
        let address = LogicalRowAddress::try_new_from_parts(u32::MAX - 1, u32::MAX).unwrap();

        assert_eq!(address.parts(), (u32::MAX - 1, u32::MAX));
        assert_eq!(LogicalRowAddress::try_from(address.raw()).unwrap(), address);
    }

    #[test]
    fn test_logical_row_address_rejects_reserved_fragment_id() {
        let error = LogicalRowAddress::try_new_from_parts(u32::MAX, 0).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("logical_fragment_id=4294967295"));
    }

    #[test]
    fn test_logical_row_address_rejects_invalid_raw_values() {
        for raw in [0xffff_ffff_0000_0000, LogicalRowAddress::INVALID_RAW] {
            let error = LogicalRowAddress::try_from(raw).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("reserved logical fragment ID"));
        }
    }
}
