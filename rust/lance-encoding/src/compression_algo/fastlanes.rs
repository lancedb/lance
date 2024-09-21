// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// This file is a modification of the `fastlanes` crate: https://github.com/spiraldb/fastlanes
// It is modified to allow a rust stable build 

use arrayref::{array_mut_ref, array_ref};
use core::mem::size_of;
use paste::paste;

pub const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

pub trait FastLanes: Sized + Copy {
    const T: usize = size_of::<Self>() * 8;
    const LANES: usize = 1024 / Self::T;
}

// Implement the trait for basic unsigned integer types
impl FastLanes for u8 {}
impl FastLanes for u16 {}
impl FastLanes for u32 {}
impl FastLanes for u64 {}

#[macro_export]
macro_rules! iterate {
    ($T:ty, $lane: expr, | $_1:tt $idx:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $idx:ident ) => ( $($body)* )}
        {
            use $crate::{seq_t, FL_ORDER};
            use paste::paste;

            #[inline(always)]
            fn index(row: usize, lane: usize) -> usize {
                let o = row / 8;
                let s = row % 8;
                (FL_ORDER[o] * 16) + (s * 128) + lane
            }

            paste!(seq_t!(row in $T {
                let idx = index(row, $lane);
                __kernel__!(idx);
            }));
        }
    }
}

#[macro_export]
macro_rules! pack {
    ($T:ty, $W:expr, $packed:expr, $lane:expr, | $_1:tt $idx:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $idx:ident ) => ( $($body)* )}
        {
            use paste::paste;

            // The number of bits of T.
            const T: usize = <$T>::T;

            #[inline(always)]
            fn index(row: usize, lane: usize) -> usize {
                let o = row / 8;
                let s = row % 8;
                (FL_ORDER[o] * 16) + (s * 128) + lane
            }

            if $W == 0 {
                // Nothing to do if W is 0, since the packed array is zero bytes.
            } else if $W == T {
                // Special case for W=T, we can just copy the input value directly to the packed value.
                paste!(seq_t!(row in $T {
                    let idx = index(row, $lane);
                    $packed[<$T>::LANES * row + $lane] = __kernel__!(idx);
                }));
            } else {
                // A mask of W bits.
                let mask: $T = (1 << $W) - 1;

                // First we loop over each lane in the virtual 1024 bit word.
                let mut tmp: $T = 0;

                // Loop over each of the rows of the lane.
                // Inlining this loop means all branches are known at compile time and
                // the code is auto-vectorized for SIMD execution.
                paste!(seq_t!(row in $T {
                    let idx = index(row, $lane);
                    let src = __kernel__!(idx);
                    let src = src & mask;

                    // Shift the src bits into their position in the tmp output variable.
                    if row == 0 {
                        tmp = src;
                    } else {
                        tmp |= src << (row * $W) % T;
                    }

                    // If the next packed position is after our current one, then we have filled
                    // the current output and we can write the packed value.
                    let curr_word: usize = (row * $W) / T;
                    let next_word: usize = ((row + 1) * $W) / T;

                    #[allow(unused_assignments)]
                    if next_word > curr_word {
                        $packed[<$T>::LANES * curr_word + $lane] = tmp;
                        let remaining_bits: usize = ((row + 1) * $W) % T;
                        // Keep the remaining bits for the next packed value.
                        tmp = src >> $W - remaining_bits;
                    }
                }));
            }
        }
    };
}

#[macro_export]
macro_rules! unpack {
    ($T:ty, $W:expr, $packed:expr, $lane:expr, | $_1:tt $idx:ident, $_2:tt $elem:ident | $($body:tt)*) => {
        macro_rules! __kernel__ {( $_1 $idx:ident, $_2 $elem:ident ) => ( $($body)* )}
        {
            use paste::paste;

            // The number of bits of T.
            const T: usize = <$T>::T;

            #[inline(always)]
            fn index(row: usize, lane: usize) -> usize {
                let o = row / 8;
                let s = row % 8;
                (FL_ORDER[o] * 16) + (s * 128) + lane
            }

            if $W == 0 {
                // Special case for W=0, we just need to zero the output.
                // We'll still respect the iteration order in case the kernel has side effects.
                paste!(seq_t!(row in $T {
                    let idx = index(row, $lane);
                    let zero: $T = 0;
                    __kernel__!(idx, zero);
                }));
            } else if $W == T {
                // Special case for W=T, we can just copy the packed value directly to the output.
                paste!(seq_t!(row in $T {
                    let idx = index(row, $lane);
                    let src = $packed[<$T>::LANES * row + $lane];
                    __kernel__!(idx, src);
                }));
            } else {
                #[inline]
                fn mask(width: usize) -> $T {
                    if width == T { <$T>::MAX } else { (1 << (width % T)) - 1 }
                }

                let mut src: $T = $packed[$lane];
                let mut tmp: $T;

                paste!(seq_t!(row in $T {
                    // Figure out the packed positions
                    let curr_word: usize = (row * $W) / T;
                    let next_word = ((row + 1) * $W) / T;

                    let shift = (row * $W) % T;

                    if next_word > curr_word {
                        // Consume some bits from the curr packed input, the remainder are in the next
                        // packed input value
                        let remaining_bits = ((row + 1) * $W) % T;
                        let current_bits = $W - remaining_bits;
                        tmp = (src >> shift) & mask(current_bits);

                        if next_word < $W {
                            // Load the next packed value
                            src = $packed[<$T>::LANES * next_word + $lane];
                            // Consume the remaining bits from the next input value.
                            tmp |= (src & mask(remaining_bits)) << current_bits;
                        }
                    } else {
                        // Otherwise, just grab W bits from the src value
                        tmp = (src >> shift) & mask($W);
                    }

                    // Write out the unpacked value
                    let idx = index(row, $lane);
                    __kernel__!(idx, tmp);
                }));
            }
        }
    };
}

// Macro for repeating a code block bit_size_of::<T> times.
#[macro_export]
macro_rules! seq_t {
    ($ident:ident in u8 $body:tt) => {seq_macro::seq!($ident in 0..8 $body)};
    ($ident:ident in u16 $body:tt) => {seq_macro::seq!($ident in 0..16 $body)};
    ($ident:ident in u32 $body:tt) => {seq_macro::seq!($ident in 0..32 $body)};
    ($ident:ident in u64 $body:tt) => {seq_macro::seq!($ident in 0..64 $body)};
}

/// `BitPack` into a compile-time known bit-width.
pub trait BitPacking: FastLanes {
    /// Packs 1024 elements into `W` bits each, where `W` is runtime-known instead of
    /// compile-time known.
    ///
    /// # Safety
    /// The input slice must be of exactly length 1024. The output slice must be of length
    /// `1024 * W / T`, where `T` is the bit-width of Self and `W` is the packed width.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]);

    /// Unpacks 1024 elements from `W` bits each, where `W` is runtime-known instead of
    /// compile-time known.
    ///
    /// # Safety
    /// The input slice must be of length `1024 * W / T`, where `T` is the bit-width of Self and `W`
    /// is the packed width. The output slice must be of exactly length 1024.
    /// These lengths are checked only with `debug_assert` (i.e., not checked on release builds).
    unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]);
}

impl BitPacking for u8 {
    unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]) {
        let packed_len = 128 * width / size_of::<Self>();
        debug_assert_eq!(output.len(), packed_len, "Output buffer must be of size 1024 * W / T");
        debug_assert_eq!(input.len(), 1024, "Input buffer must be of size 1024");
        debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

        match width {
            1 => pack_8_1(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 1 / u8::T]),
            2 => pack_8_2(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 2 / u8::T]),
            3 => pack_8_3(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 3 / u8::T]),
            4 => pack_8_4(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 4 / u8::T]),
            5 => pack_8_5(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 5 / u8::T]),
            6 => pack_8_6(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 6 / u8::T]),
            7 => pack_8_7(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 7 / u8::T]),
            8 => pack_8_8(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 8 / u8::T]),

            _ => unreachable!("Unsupported width: {}", width)
        }
    }

    unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]) {
        let packed_len = 128 * width / size_of::<Self>();
        debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
        debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
        debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

        match width {
            1 => unpack_8_1(array_ref![input, 0, 1024 * 1 / u8::T], array_mut_ref![output, 0, 1024]),
            2 => unpack_8_2(array_ref![input, 0, 1024 * 2 / u8::T], array_mut_ref![output, 0, 1024]),
            3 => unpack_8_3(array_ref![input, 0, 1024 * 3 / u8::T], array_mut_ref![output, 0, 1024]),
            4 => unpack_8_4(array_ref![input, 0, 1024 * 4 / u8::T], array_mut_ref![output, 0, 1024]),
            5 => unpack_8_5(array_ref![input, 0, 1024 * 5 / u8::T], array_mut_ref![output, 0, 1024]),
            6 => unpack_8_6(array_ref![input, 0, 1024 * 6 / u8::T], array_mut_ref![output, 0, 1024]),
            7 => unpack_8_7(array_ref![input, 0, 1024 * 7 / u8::T], array_mut_ref![output, 0, 1024]),
            8 => unpack_8_8(array_ref![input, 0, 1024 * 8 / u8::T], array_mut_ref![output, 0, 1024]),

            _ => unreachable!("Unsupported width: {}", width)
        }
    }
}

impl BitPacking for u16 {
    unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]) {
        let packed_len = 128 * width / size_of::<Self>();
        debug_assert_eq!(output.len(), packed_len, "Output buffer must be of size 1024 * W / T");
        debug_assert_eq!(input.len(), 1024, "Input buffer must be of size 1024");
        debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

        match width {
            1 => pack_16_1(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 1 / u16::T]),
            2 => pack_16_2(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 2 / u16::T]),
            3 => pack_16_3(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 3 / u16::T]),
            4 => pack_16_4(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 4 / u16::T]),
            5 => pack_16_5(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 5 / u16::T]),
            6 => pack_16_6(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 6 / u16::T]),
            7 => pack_16_7(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 7 / u16::T]),
            8 => pack_16_8(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 8 / u16::T]),
            9 => pack_16_9(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 9 / u16::T]),

            10 => pack_16_10(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 10 / u16::T]),
            11 => pack_16_11(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 11 / u16::T]),
            12 => pack_16_12(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 12 / u16::T]),
            13 => pack_16_13(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 13 / u16::T]),
            14 => pack_16_14(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 14 / u16::T]),
            15 => pack_16_15(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 15 / u16::T]),
            16 => pack_16_16(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 16 / u16::T]),

            _ => unreachable!("Unsupported width: {}", width)
        }
    }

    unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]) {
        let packed_len = 128 * width / size_of::<Self>();
        debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
        debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
        debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

        match width {
            1 => unpack_16_1(array_ref![input, 0, 1024 * 1 / u16::T], array_mut_ref![output, 0, 1024]),
            2 => unpack_16_2(array_ref![input, 0, 1024 * 2 / u16::T], array_mut_ref![output, 0, 1024]),
            3 => unpack_16_3(array_ref![input, 0, 1024 * 3 / u16::T], array_mut_ref![output, 0, 1024]),
            4 => unpack_16_4(array_ref![input, 0, 1024 * 4 / u16::T], array_mut_ref![output, 0, 1024]),
            5 => unpack_16_5(array_ref![input, 0, 1024 * 5 / u16::T], array_mut_ref![output, 0, 1024]),
            6 => unpack_16_6(array_ref![input, 0, 1024 * 6 / u16::T], array_mut_ref![output, 0, 1024]),
            7 => unpack_16_7(array_ref![input, 0, 1024 * 7 / u16::T], array_mut_ref![output, 0, 1024]),
            8 => unpack_16_8(array_ref![input, 0, 1024 * 8 / u16::T], array_mut_ref![output, 0, 1024]),
            9 => unpack_16_9(array_ref![input, 0, 1024 * 9 / u16::T], array_mut_ref![output, 0, 1024]),

            10 => unpack_16_10(array_ref![input, 0, 1024 * 10 / u16::T], array_mut_ref![output, 0, 1024]),
            11 => unpack_16_11(array_ref![input, 0, 1024 * 11 / u16::T], array_mut_ref![output, 0, 1024]),
            12 => unpack_16_12(array_ref![input, 0, 1024 * 12 / u16::T], array_mut_ref![output, 0, 1024]),
            13 => unpack_16_13(array_ref![input, 0, 1024 * 13 / u16::T], array_mut_ref![output, 0, 1024]),
            14 => unpack_16_14(array_ref![input, 0, 1024 * 14 / u16::T], array_mut_ref![output, 0, 1024]),
            15 => unpack_16_15(array_ref![input, 0, 1024 * 15 / u16::T], array_mut_ref![output, 0, 1024]),
            16 => unpack_16_16(array_ref![input, 0, 1024 * 16 / u16::T], array_mut_ref![output, 0, 1024]),

            _ => unreachable!("Unsupported width: {}", width)
        }
    }
}

impl BitPacking for u32 {
    unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]) {
        let packed_len = 128 * width / size_of::<Self>();
        debug_assert_eq!(output.len(), packed_len, "Output buffer must be of size 1024 * W / T");
        debug_assert_eq!(input.len(), 1024, "Input buffer must be of size 1024");
        debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

        match width {
            1 => pack_32_1(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 1 / u32::T]),
            2 => pack_32_2(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 2 / u32::T]),
            3 => pack_32_3(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 3 / u32::T]),
            4 => pack_32_4(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 4 / u32::T]),
            5 => pack_32_5(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 5 / u32::T]),
            6 => pack_32_6(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 6 / u32::T]),
            7 => pack_32_7(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 7 / u32::T]),
            8 => pack_32_8(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 8 / u32::T]),
            9 => pack_32_9(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 9 / u32::T]),

            10 => pack_32_10(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 10 / u32::T]),
            11 => pack_32_11(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 11 / u32::T]),
            12 => pack_32_12(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 12 / u32::T]),
            13 => pack_32_13(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 13 / u32::T]),
            14 => pack_32_14(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 14 / u32::T]),
            15 => pack_32_15(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 15 / u32::T]),
            16 => pack_32_16(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 16 / u32::T]),
            17 => pack_32_17(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 17 / u32::T]),
            18 => pack_32_18(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 18 / u32::T]),
            19 => pack_32_19(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 19 / u32::T]),

            20 => pack_32_20(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 20 / u32::T]),
            21 => pack_32_21(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 21 / u32::T]),
            22 => pack_32_22(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 22 / u32::T]),
            23 => pack_32_23(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 23 / u32::T]),
            24 => pack_32_24(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 24 / u32::T]),
            25 => pack_32_25(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 25 / u32::T]),
            26 => pack_32_26(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 26 / u32::T]),
            27 => pack_32_27(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 27 / u32::T]),
            28 => pack_32_28(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 28 / u32::T]),
            29 => pack_32_29(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 29 / u32::T]),

            30 => pack_32_30(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 30 / u32::T]),
            31 => pack_32_31(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 31 / u32::T]),
            32 => pack_32_32(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 32 / u32::T]),

            _ => unreachable!("Unsupported width: {}", width)
        }
    }

    unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]) {
        let packed_len = 128 * width / size_of::<Self>();
        debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
        debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
        debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

        match width {
            1 => unpack_32_1(array_ref![input, 0, 1024 * 1 / u32::T], array_mut_ref![output, 0, 1024]),
            2 => unpack_32_2(array_ref![input, 0, 1024 * 2 / u32::T], array_mut_ref![output, 0, 1024]),
            3 => unpack_32_3(array_ref![input, 0, 1024 * 3 / u32::T], array_mut_ref![output, 0, 1024]),
            4 => unpack_32_4(array_ref![input, 0, 1024 * 4 / u32::T], array_mut_ref![output, 0, 1024]),
            5 => unpack_32_5(array_ref![input, 0, 1024 * 5 / u32::T], array_mut_ref![output, 0, 1024]),
            6 => unpack_32_6(array_ref![input, 0, 1024 * 6 / u32::T], array_mut_ref![output, 0, 1024]),
            7 => unpack_32_7(array_ref![input, 0, 1024 * 7 / u32::T], array_mut_ref![output, 0, 1024]),
            8 => unpack_32_8(array_ref![input, 0, 1024 * 8 / u32::T], array_mut_ref![output, 0, 1024]),
            9 => unpack_32_9(array_ref![input, 0, 1024 * 9 / u32::T], array_mut_ref![output, 0, 1024]),

            10 => unpack_32_10(array_ref![input, 0, 1024 * 10 / u32::T], array_mut_ref![output, 0, 1024]),
            11 => unpack_32_11(array_ref![input, 0, 1024 * 11 / u32::T], array_mut_ref![output, 0, 1024]),
            12 => unpack_32_12(array_ref![input, 0, 1024 * 12 / u32::T], array_mut_ref![output, 0, 1024]),
            13 => unpack_32_13(array_ref![input, 0, 1024 * 13 / u32::T], array_mut_ref![output, 0, 1024]),
            14 => unpack_32_14(array_ref![input, 0, 1024 * 14 / u32::T], array_mut_ref![output, 0, 1024]),
            15 => unpack_32_15(array_ref![input, 0, 1024 * 15 / u32::T], array_mut_ref![output, 0, 1024]),
            16 => unpack_32_16(array_ref![input, 0, 1024 * 16 / u32::T], array_mut_ref![output, 0, 1024]),
            17 => unpack_32_17(array_ref![input, 0, 1024 * 17 / u32::T], array_mut_ref![output, 0, 1024]),
            18 => unpack_32_18(array_ref![input, 0, 1024 * 18 / u32::T], array_mut_ref![output, 0, 1024]),
            19 => unpack_32_19(array_ref![input, 0, 1024 * 19 / u32::T], array_mut_ref![output, 0, 1024]),

            20 => unpack_32_20(array_ref![input, 0, 1024 * 20 / u32::T], array_mut_ref![output, 0, 1024]),
            21 => unpack_32_21(array_ref![input, 0, 1024 * 21 / u32::T], array_mut_ref![output, 0, 1024]),
            22 => unpack_32_22(array_ref![input, 0, 1024 * 22 / u32::T], array_mut_ref![output, 0, 1024]),
            23 => unpack_32_23(array_ref![input, 0, 1024 * 23 / u32::T], array_mut_ref![output, 0, 1024]),
            24 => unpack_32_24(array_ref![input, 0, 1024 * 24 / u32::T], array_mut_ref![output, 0, 1024]),
            25 => unpack_32_25(array_ref![input, 0, 1024 * 25 / u32::T], array_mut_ref![output, 0, 1024]),
            26 => unpack_32_26(array_ref![input, 0, 1024 * 26 / u32::T], array_mut_ref![output, 0, 1024]),
            27 => unpack_32_27(array_ref![input, 0, 1024 * 27 / u32::T], array_mut_ref![output, 0, 1024]),
            28 => unpack_32_28(array_ref![input, 0, 1024 * 28 / u32::T], array_mut_ref![output, 0, 1024]),
            29 => unpack_32_29(array_ref![input, 0, 1024 * 29 / u32::T], array_mut_ref![output, 0, 1024]),

            30 => unpack_32_30(array_ref![input, 0, 1024 * 30 / u32::T], array_mut_ref![output, 0, 1024]),
            31 => unpack_32_31(array_ref![input, 0, 1024 * 31 / u32::T], array_mut_ref![output, 0, 1024]),
            32 => unpack_32_32(array_ref![input, 0, 1024 * 32 / u32::T], array_mut_ref![output, 0, 1024]),

            _ => unreachable!("Unsupported width: {}", width)
        }
    }
}

impl BitPacking for u64 {
    unsafe fn unchecked_pack(width: usize, input: &[Self], output: &mut [Self]) {
        let packed_len = 128 * width / size_of::<Self>();
        debug_assert_eq!(output.len(), packed_len, "Output buffer must be of size 1024 * W / T");
        debug_assert_eq!(input.len(), 1024, "Input buffer must be of size 1024");
        debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

        match width {
            1 => pack_64_1(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 1 / u64::T]),
            2 => pack_64_2(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 2 / u64::T]),
            3 => pack_64_3(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 3 / u64::T]),
            4 => pack_64_4(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 4 / u64::T]),
            5 => pack_64_5(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 5 / u64::T]),
            6 => pack_64_6(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 6 / u64::T]),
            7 => pack_64_7(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 7 / u64::T]),
            8 => pack_64_8(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 8 / u64::T]),
            9 => pack_64_9(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 9 / u64::T]),

            10 => pack_64_10(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 10 / u64::T]),
            11 => pack_64_11(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 11 / u64::T]),
            12 => pack_64_12(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 12 / u64::T]),
            13 => pack_64_13(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 13 / u64::T]),
            14 => pack_64_14(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 14 / u64::T]),
            15 => pack_64_15(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 15 / u64::T]),
            16 => pack_64_16(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 16 / u64::T]),
            17 => pack_64_17(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 17 / u64::T]),
            18 => pack_64_18(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 18 / u64::T]),
            19 => pack_64_19(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 19 / u64::T]),

            20 => pack_64_20(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 20 / u64::T]),
            21 => pack_64_21(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 21 / u64::T]),
            22 => pack_64_22(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 22 / u64::T]),
            23 => pack_64_23(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 23 / u64::T]),
            24 => pack_64_24(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 24 / u64::T]),
            25 => pack_64_25(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 25 / u64::T]),
            26 => pack_64_26(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 26 / u64::T]),
            27 => pack_64_27(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 27 / u64::T]),
            28 => pack_64_28(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 28 / u64::T]),
            29 => pack_64_29(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 29 / u64::T]),

            30 => pack_64_30(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 30 / u64::T]),
            31 => pack_64_31(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 31 / u64::T]),
            32 => pack_64_32(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 32 / u64::T]),
            33 => pack_64_33(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 33 / u64::T]),
            34 => pack_64_34(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 34 / u64::T]),
            35 => pack_64_35(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 35 / u64::T]),
            36 => pack_64_36(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 36 / u64::T]),
            37 => pack_64_37(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 37 / u64::T]),
            38 => pack_64_38(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 38 / u64::T]),
            39 => pack_64_39(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 39 / u64::T]),

            40 => pack_64_40(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 40 / u64::T]),
            41 => pack_64_41(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 41 / u64::T]),
            42 => pack_64_42(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 42 / u64::T]),
            43 => pack_64_43(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 43 / u64::T]),
            44 => pack_64_44(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 44 / u64::T]),
            45 => pack_64_45(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 45 / u64::T]),
            46 => pack_64_46(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 46 / u64::T]),
            47 => pack_64_47(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 47 / u64::T]),
            48 => pack_64_48(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 48 / u64::T]),
            49 => pack_64_49(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 49 / u64::T]),

            50 => pack_64_50(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 50 / u64::T]),
            51 => pack_64_51(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 51 / u64::T]),
            52 => pack_64_52(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 52 / u64::T]),
            53 => pack_64_53(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 53 / u64::T]),
            54 => pack_64_54(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 54 / u64::T]),
            55 => pack_64_55(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 55 / u64::T]),
            56 => pack_64_56(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 56 / u64::T]),
            57 => pack_64_57(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 57 / u64::T]),
            58 => pack_64_58(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 58 / u64::T]),
            59 => pack_64_59(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 59 / u64::T]),

            60 => pack_64_60(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 60 / u64::T]),
            61 => pack_64_61(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 61 / u64::T]),
            62 => pack_64_62(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 62 / u64::T]),
            63 => pack_64_63(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 63 / u64::T]),
            64 => pack_64_64(array_ref![input, 0, 1024], array_mut_ref![output, 0, 1024 * 64 / u64::T]),

            _ => unreachable!("Unsupported width: {}", width)
        }
    }

    unsafe fn unchecked_unpack(width: usize, input: &[Self], output: &mut [Self]) {
        let packed_len = 128 * width / size_of::<Self>();
        debug_assert_eq!(input.len(), packed_len, "Input buffer must be of size 1024 * W / T");
        debug_assert_eq!(output.len(), 1024, "Output buffer must be of size 1024");
        debug_assert!(width <= Self::T, "Width must be less than or equal to {}", Self::T);

        match width {
            1 => unpack_64_1(array_ref![input, 0, 1024 * 1 / u64::T], array_mut_ref![output, 0, 1024]),
            2 => unpack_64_2(array_ref![input, 0, 1024 * 2 / u64::T], array_mut_ref![output, 0, 1024]),
            3 => unpack_64_3(array_ref![input, 0, 1024 * 3 / u64::T], array_mut_ref![output, 0, 1024]),
            4 => unpack_64_4(array_ref![input, 0, 1024 * 4 / u64::T], array_mut_ref![output, 0, 1024]),
            5 => unpack_64_5(array_ref![input, 0, 1024 * 5 / u64::T], array_mut_ref![output, 0, 1024]),
            6 => unpack_64_6(array_ref![input, 0, 1024 * 6 / u64::T], array_mut_ref![output, 0, 1024]),
            7 => unpack_64_7(array_ref![input, 0, 1024 * 7 / u64::T], array_mut_ref![output, 0, 1024]),
            8 => unpack_64_8(array_ref![input, 0, 1024 * 8 / u64::T], array_mut_ref![output, 0, 1024]),
            9 => unpack_64_9(array_ref![input, 0, 1024 * 9 / u64::T], array_mut_ref![output, 0, 1024]),

            10 => unpack_64_10(array_ref![input, 0, 1024 * 10 / u64::T], array_mut_ref![output, 0, 1024]),
            11 => unpack_64_11(array_ref![input, 0, 1024 * 11 / u64::T], array_mut_ref![output, 0, 1024]),
            12 => unpack_64_12(array_ref![input, 0, 1024 * 12 / u64::T], array_mut_ref![output, 0, 1024]),
            13 => unpack_64_13(array_ref![input, 0, 1024 * 13 / u64::T], array_mut_ref![output, 0, 1024]),
            14 => unpack_64_14(array_ref![input, 0, 1024 * 14 / u64::T], array_mut_ref![output, 0, 1024]),
            15 => unpack_64_15(array_ref![input, 0, 1024 * 15 / u64::T], array_mut_ref![output, 0, 1024]),
            16 => unpack_64_16(array_ref![input, 0, 1024 * 16 / u64::T], array_mut_ref![output, 0, 1024]),
            17 => unpack_64_17(array_ref![input, 0, 1024 * 17 / u64::T], array_mut_ref![output, 0, 1024]),
            18 => unpack_64_18(array_ref![input, 0, 1024 * 18 / u64::T], array_mut_ref![output, 0, 1024]),
            19 => unpack_64_19(array_ref![input, 0, 1024 * 19 / u64::T], array_mut_ref![output, 0, 1024]),

            20 => unpack_64_20(array_ref![input, 0, 1024 * 20 / u64::T], array_mut_ref![output, 0, 1024]),
            21 => unpack_64_21(array_ref![input, 0, 1024 * 21 / u64::T], array_mut_ref![output, 0, 1024]),
            22 => unpack_64_22(array_ref![input, 0, 1024 * 22 / u64::T], array_mut_ref![output, 0, 1024]),
            23 => unpack_64_23(array_ref![input, 0, 1024 * 23 / u64::T], array_mut_ref![output, 0, 1024]),
            24 => unpack_64_24(array_ref![input, 0, 1024 * 24 / u64::T], array_mut_ref![output, 0, 1024]),
            25 => unpack_64_25(array_ref![input, 0, 1024 * 25 / u64::T], array_mut_ref![output, 0, 1024]),
            26 => unpack_64_26(array_ref![input, 0, 1024 * 26 / u64::T], array_mut_ref![output, 0, 1024]),
            27 => unpack_64_27(array_ref![input, 0, 1024 * 27 / u64::T], array_mut_ref![output, 0, 1024]),
            28 => unpack_64_28(array_ref![input, 0, 1024 * 28 / u64::T], array_mut_ref![output, 0, 1024]),
            29 => unpack_64_29(array_ref![input, 0, 1024 * 29 / u64::T], array_mut_ref![output, 0, 1024]),

            30 => unpack_64_30(array_ref![input, 0, 1024 * 30 / u64::T], array_mut_ref![output, 0, 1024]),
            31 => unpack_64_31(array_ref![input, 0, 1024 * 31 / u64::T], array_mut_ref![output, 0, 1024]),
            32 => unpack_64_32(array_ref![input, 0, 1024 * 32 / u64::T], array_mut_ref![output, 0, 1024]),
            33 => unpack_64_33(array_ref![input, 0, 1024 * 33 / u64::T], array_mut_ref![output, 0, 1024]),
            34 => unpack_64_34(array_ref![input, 0, 1024 * 34 / u64::T], array_mut_ref![output, 0, 1024]),
            35 => unpack_64_35(array_ref![input, 0, 1024 * 35 / u64::T], array_mut_ref![output, 0, 1024]),
            36 => unpack_64_36(array_ref![input, 0, 1024 * 36 / u64::T], array_mut_ref![output, 0, 1024]),
            37 => unpack_64_37(array_ref![input, 0, 1024 * 37 / u64::T], array_mut_ref![output, 0, 1024]),
            38 => unpack_64_38(array_ref![input, 0, 1024 * 38 / u64::T], array_mut_ref![output, 0, 1024]),
            39 => unpack_64_39(array_ref![input, 0, 1024 * 39 / u64::T], array_mut_ref![output, 0, 1024]),

            40 => unpack_64_40(array_ref![input, 0, 1024 * 40 / u64::T], array_mut_ref![output, 0, 1024]),
            41 => unpack_64_41(array_ref![input, 0, 1024 * 41 / u64::T], array_mut_ref![output, 0, 1024]),
            42 => unpack_64_42(array_ref![input, 0, 1024 * 42 / u64::T], array_mut_ref![output, 0, 1024]),
            43 => unpack_64_43(array_ref![input, 0, 1024 * 43 / u64::T], array_mut_ref![output, 0, 1024]),
            44 => unpack_64_44(array_ref![input, 0, 1024 * 44 / u64::T], array_mut_ref![output, 0, 1024]),
            45 => unpack_64_45(array_ref![input, 0, 1024 * 45 / u64::T], array_mut_ref![output, 0, 1024]),
            46 => unpack_64_46(array_ref![input, 0, 1024 * 46 / u64::T], array_mut_ref![output, 0, 1024]),
            47 => unpack_64_47(array_ref![input, 0, 1024 * 47 / u64::T], array_mut_ref![output, 0, 1024]),
            48 => unpack_64_48(array_ref![input, 0, 1024 * 48 / u64::T], array_mut_ref![output, 0, 1024]),
            49 => unpack_64_49(array_ref![input, 0, 1024 * 49 / u64::T], array_mut_ref![output, 0, 1024]),

            50 => unpack_64_50(array_ref![input, 0, 1024 * 50 / u64::T], array_mut_ref![output, 0, 1024]),
            51 => unpack_64_51(array_ref![input, 0, 1024 * 51 / u64::T], array_mut_ref![output, 0, 1024]),
            52 => unpack_64_52(array_ref![input, 0, 1024 * 52 / u64::T], array_mut_ref![output, 0, 1024]),
            53 => unpack_64_53(array_ref![input, 0, 1024 * 53 / u64::T], array_mut_ref![output, 0, 1024]),
            54 => unpack_64_54(array_ref![input, 0, 1024 * 54 / u64::T], array_mut_ref![output, 0, 1024]),
            55 => unpack_64_55(array_ref![input, 0, 1024 * 55 / u64::T], array_mut_ref![output, 0, 1024]),
            56 => unpack_64_56(array_ref![input, 0, 1024 * 56 / u64::T], array_mut_ref![output, 0, 1024]),
            57 => unpack_64_57(array_ref![input, 0, 1024 * 57 / u64::T], array_mut_ref![output, 0, 1024]),
            58 => unpack_64_58(array_ref![input, 0, 1024 * 58 / u64::T], array_mut_ref![output, 0, 1024]),
            59 => unpack_64_59(array_ref![input, 0, 1024 * 59 / u64::T], array_mut_ref![output, 0, 1024]),

            60 => unpack_64_60(array_ref![input, 0, 1024 * 60 / u64::T], array_mut_ref![output, 0, 1024]),
            61 => unpack_64_61(array_ref![input, 0, 1024 * 61 / u64::T], array_mut_ref![output, 0, 1024]),
            62 => unpack_64_62(array_ref![input, 0, 1024 * 62 / u64::T], array_mut_ref![output, 0, 1024]),
            63 => unpack_64_63(array_ref![input, 0, 1024 * 63 / u64::T], array_mut_ref![output, 0, 1024]),
            64 => unpack_64_64(array_ref![input, 0, 1024 * 64 / u64::T], array_mut_ref![output, 0, 1024]),

            _ => unreachable!("Unsupported width: {}", width)
        }
    }
}

macro_rules! unpack_8 {
    ($name:ident, $bits:expr) => {
        fn $name(input: &[u8; 1024 * $bits / u8::T], output: &mut [u8; 1024]) {
            for lane in 0..u8::LANES {
                unpack!(u8, $bits, input, lane, |$idx, $elem| {
                    output[$idx] = $elem;
                });
            }
        }
    };
}

unpack_8!(unpack_8_1, 1);
unpack_8!(unpack_8_2, 2);
unpack_8!(unpack_8_3, 3);
unpack_8!(unpack_8_4, 4);
unpack_8!(unpack_8_5, 5);
unpack_8!(unpack_8_6, 6);
unpack_8!(unpack_8_7, 7);
unpack_8!(unpack_8_8, 8);

macro_rules! pack_8 {
    ($name:ident, $bits:expr) => {
        fn $name(input: &[u8; 1024], output: &mut [u8; 1024 * $bits / u8::T]) {
            for lane in 0..u8::LANES {
                pack!(u8, $bits, output, lane, |$idx| {
                    input[$idx]
                });
            }
        }
    };
}
pack_8!(pack_8_1, 1);
pack_8!(pack_8_2, 2);
pack_8!(pack_8_3, 3);
pack_8!(pack_8_4, 4);
pack_8!(pack_8_5, 5);
pack_8!(pack_8_6, 6);
pack_8!(pack_8_7, 7);
pack_8!(pack_8_8, 8);

macro_rules! unpack_16 {
    ($name:ident, $bits:expr) => {
        fn $name(input: &[u16; 1024 * $bits / u16::T], output: &mut [u16; 1024]) {
            for lane in 0..u16::LANES {
                unpack!(u16, $bits, input, lane, |$idx, $elem| {
                    output[$idx] = $elem;
                });
            }
        }
    };
}

unpack_16!(unpack_16_1, 1);
unpack_16!(unpack_16_2, 2);
unpack_16!(unpack_16_3, 3);
unpack_16!(unpack_16_4, 4);
unpack_16!(unpack_16_5, 5);
unpack_16!(unpack_16_6, 6);
unpack_16!(unpack_16_7, 7);
unpack_16!(unpack_16_8, 8);
unpack_16!(unpack_16_9, 9);
unpack_16!(unpack_16_10, 10);
unpack_16!(unpack_16_11, 11);
unpack_16!(unpack_16_12, 12);
unpack_16!(unpack_16_13, 13);
unpack_16!(unpack_16_14, 14);
unpack_16!(unpack_16_15, 15);
unpack_16!(unpack_16_16, 16);

macro_rules! pack_16 {
    ($name:ident, $bits:expr) => {
        fn $name(input: &[u16; 1024], output: &mut [u16; 1024 * $bits / u16::T]) {
            for lane in 0..u16::LANES {
                pack!(u16, $bits, output, lane, |$idx| {
                    input[$idx]
                });
            }
        }
    };
}

pack_16!(pack_16_1, 1);
pack_16!(pack_16_2, 2);
pack_16!(pack_16_3, 3);
pack_16!(pack_16_4, 4);
pack_16!(pack_16_5, 5);
pack_16!(pack_16_6, 6);
pack_16!(pack_16_7, 7);
pack_16!(pack_16_8, 8);
pack_16!(pack_16_9, 9);
pack_16!(pack_16_10, 10);
pack_16!(pack_16_11, 11);
pack_16!(pack_16_12, 12);
pack_16!(pack_16_13, 13);
pack_16!(pack_16_14, 14);
pack_16!(pack_16_15, 15);
pack_16!(pack_16_16, 16);

macro_rules! unpack_32 {
    ($name:ident, $bit_width:expr) => {
        fn $name(input: &[u32; 1024 * $bit_width / u32::T], output: &mut [u32; 1024]) {
            for lane in 0..u32::LANES {
                unpack!(u32, $bit_width, input, lane, |$idx, $elem| {
                    output[$idx] = $elem
                });
            }
        }
    };
}

unpack_32!(unpack_32_1, 1);
unpack_32!(unpack_32_2, 2);
unpack_32!(unpack_32_3, 3);
unpack_32!(unpack_32_4, 4);
unpack_32!(unpack_32_5, 5);
unpack_32!(unpack_32_6, 6);
unpack_32!(unpack_32_7, 7);
unpack_32!(unpack_32_8, 8);
unpack_32!(unpack_32_9, 9);
unpack_32!(unpack_32_10, 10);
unpack_32!(unpack_32_11, 11);
unpack_32!(unpack_32_12, 12);
unpack_32!(unpack_32_13, 13);
unpack_32!(unpack_32_14, 14);
unpack_32!(unpack_32_15, 15);
unpack_32!(unpack_32_16, 16);
unpack_32!(unpack_32_17, 17);
unpack_32!(unpack_32_18, 18);
unpack_32!(unpack_32_19, 19);
unpack_32!(unpack_32_20, 20);
unpack_32!(unpack_32_21, 21);
unpack_32!(unpack_32_22, 22);
unpack_32!(unpack_32_23, 23);
unpack_32!(unpack_32_24, 24);
unpack_32!(unpack_32_25, 25);
unpack_32!(unpack_32_26, 26);
unpack_32!(unpack_32_27, 27);
unpack_32!(unpack_32_28, 28);
unpack_32!(unpack_32_29, 29);
unpack_32!(unpack_32_30, 30);
unpack_32!(unpack_32_31, 31);
unpack_32!(unpack_32_32, 32);

macro_rules! pack_32 {
    ($name:ident, $bits:expr) => {
        fn $name(input: &[u32; 1024], output: &mut [u32; 1024 * $bits / u32::BITS as usize]) {
            for lane in 0..u32::LANES {
                pack!(u32, $bits, output, lane, |$idx| {
                    input[$idx]
                });
            }
        }
    };
}

pack_32!(pack_32_1, 1);
pack_32!(pack_32_2, 2);
pack_32!(pack_32_3, 3);
pack_32!(pack_32_4, 4);
pack_32!(pack_32_5, 5);
pack_32!(pack_32_6, 6);
pack_32!(pack_32_7, 7);
pack_32!(pack_32_8, 8);
pack_32!(pack_32_9, 9);
pack_32!(pack_32_10, 10);
pack_32!(pack_32_11, 11);
pack_32!(pack_32_12, 12);
pack_32!(pack_32_13, 13);
pack_32!(pack_32_14, 14);
pack_32!(pack_32_15, 15);
pack_32!(pack_32_16, 16);
pack_32!(pack_32_17, 17);
pack_32!(pack_32_18, 18);
pack_32!(pack_32_19, 19);
pack_32!(pack_32_20, 20);
pack_32!(pack_32_21, 21);
pack_32!(pack_32_22, 22);
pack_32!(pack_32_23, 23);
pack_32!(pack_32_24, 24);
pack_32!(pack_32_25, 25);
pack_32!(pack_32_26, 26);
pack_32!(pack_32_27, 27);
pack_32!(pack_32_28, 28);
pack_32!(pack_32_29, 29);
pack_32!(pack_32_30, 30);
pack_32!(pack_32_31, 31);
pack_32!(pack_32_32, 32);

macro_rules! generate_unpack_64 {
    ($($n:expr),*) => {
        $(
            paste::item! {
                fn [<unpack_64_ $n>](input: &[u64; 1024 * $n / u64::T], output: &mut [u64; 1024]) {
                    for lane in 0..u64::LANES {
                        unpack!(u64, $n, input, lane, |$idx, $elem| {
                            output[$idx] = $elem
                        });
                    }
                }
            }
        )*
    };
}

generate_unpack_64!(
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
);

macro_rules! generate_pack_64 {
    ($($n:expr),*) => {
        $(
            paste::item! {
                fn [<pack_64_ $n>](input: &[u64; 1024], output: &mut [u64; 1024 * $n / u64::T]) {
                    for lane in 0..u64::LANES {
                        pack!(u64, $n, output, lane, |$idx| {
                            input[$idx]
                        });
                    }
                }
            }
        )*
    };
}

generate_pack_64!(
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
);

#[cfg(test)]
mod test {
    use core::array;
    use super::*;

    #[test]
    fn test_pack() {
        let mut values: [u16; 1024] = [0; 1024];
        for i in 0..1024 {
            values[i] = (i % (1 << 15)) as u16;
        }

        let mut packed: [u16; 960] = [0; 960];
        for lane in 0..u16::LANES {
            // Always loop over lanes first. This is what the compiler vectorizes.
            pack!(u16, 15, packed, lane, |$pos| {
                values[$pos]
            });
        }

        let mut packed_orig: [u16; 960] = [0; 960];
        unsafe {

            BitPacking::unchecked_pack(15, &values, &mut packed_orig);
        }

        let mut unpacked: [u16; 1024] = [0; 1024];
        for lane in 0..u16::LANES {
            // Always loop over lanes first. This is what the compiler vectorizes.
            unpack!(u16, 15, packed, lane, |$idx, $elem| {
                unpacked[$idx] = $elem;
            });
        }

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_unchecked_pack() {
        let input = array::from_fn(|i| i as u32);
        let mut packed = [0; 320];
        unsafe { BitPacking::unchecked_pack(10, &input, &mut packed) };
        let mut output = [0; 1024];
        unsafe { BitPacking::unchecked_unpack(10, &packed, &mut output) };
        assert_eq!(input, output);
    }
}
