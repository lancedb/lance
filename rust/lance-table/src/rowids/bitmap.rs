// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[derive(PartialEq, Eq, Clone)]
pub struct Bitmap {
    pub data: Vec<u8>,
    pub len: usize,
}

impl std::fmt::Debug for Bitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Bitmap {{ data: ")?;
        for i in 0..self.len {
            write!(f, "{}", if self.get(i) { "1" } else { "0" })?;
        }
        write!(f, ", len: {} }}", self.len)
    }
}

impl Bitmap {
    pub fn new(len: usize) -> Self {
        let data = vec![0; (len + 7) / 8];
        Self { data, len }
    }

    pub fn set(&mut self, i: usize) {
        self.data[i / 8] |= 1 << (i % 8);
    }

    pub fn clear(&mut self, i: usize) {
        self.data[i / 8] &= !(1 << (i % 8));
    }

    pub fn get(&self, i: usize) -> bool {
        self.data[i / 8] & (1 << (i % 8)) != 0
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn slice(&self, start: usize, len: usize) -> Self {
        let mut data = vec![0; (len + 7) / 8];
        for i in 0..len {
            if self.get(start + i) {
                data[i / 8] |= 1 << (i % 8);
            }
        }
        Self { data, len }
    }

    pub fn count_ones(&self) -> usize {
        self.data.iter().map(|&x| x.count_ones() as usize).sum()
    }
}

impl From<&[bool]> for Bitmap {
    fn from(slice: &[bool]) -> Self {
        let mut bitmap = Bitmap::new(slice.len());
        for (i, &b) in slice.iter().enumerate() {
            if b {
                bitmap.set(i);
            }
        }
        bitmap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap() {
        let mut bitmap = Bitmap::new(10);
        assert_eq!(bitmap.len(), 10);
        assert_eq!(bitmap.count_ones(), 0);

        bitmap.set(0);
        bitmap.set(1);
        bitmap.set(4);
        bitmap.set(5);
        bitmap.set(9);
        assert_eq!(bitmap.count_ones(), 5);
        assert_eq!(
            format!("{:?}", bitmap),
            "Bitmap { data: 1100100010, len: 10 }"
        );

        bitmap.clear(1);
        bitmap.clear(4);
        assert_eq!(bitmap.count_ones(), 3);
        assert_eq!(
            format!("{:?}", bitmap),
            "Bitmap { data: 1000000010, len: 10 }"
        );

        bitmap.slice(5, 5);
        assert_eq!(bitmap.count_ones(), 1);
        assert_eq!(format!("{:?}", bitmap), "Bitmap { data: 00010, len: 5 }");
    }
}
