// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Hamming distance.

/// Hamming distance between two vectors.
pub fn hamming(x: &[u8], y: &[u8]) -> u32 {
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi ^ yi).count_ones())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming() {
        let x = vec![0b1101_1010, 0b1010_1010, 0b1010_1010];
        let y = vec![0b1101_1010, 0b1010_1010, 0b1010_1010];
        assert_eq!(hamming(&x, &y), 0);

        let y = vec![0b1101_1010, 0b1010_1010, 0b1010_1000];
        assert_eq!(hamming(&x, &y), 1);

        let y = vec![0b1101_1010, 0b1010_1010, 0b1010_1001];
        assert_eq!(hamming(&x, &y), 2);
    }
}
