fn argmin_scalar(arr: &[f32]) -> u32 {
    arr
    .iter()
    .enumerate()
    .min_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0 as u32
}

pub trait Argmin {
    fn argmin(&self) -> u32;
}

impl Argmin for [f32] {
    #[inline]
    fn argmin(&self) -> u32 {
        #[cfg(target_arch = "aarch64")]
        {
            unimplemented!()
        }

        if self.len() < 8 {
            return argmin_scalar(self);
        }

        #[cfg(target_arch = "x86_64")]
        {
            use x86_64::sse::argmin_f32;
            return argmin_f32(self)
        }
        return argmin_scalar(self);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    pub(crate) mod sse {
        #[inline]
        pub(crate) fn argmin_f32(arr: &[f32]) -> u32 {
            unsafe {
                use std::arch::x86_64::*;

                let mut cur = _mm_setr_epi32(0, 1, 2, 3);
                let mut min = _mm_set1_ps(f32::MAX);
                let mut idx = _mm_set1_epi32(0);
                let len_aligned_to_instruction = arr.len() / 4 * 4;

                for i in (0..len_aligned_to_instruction).step_by(4) {
                    let vals = _mm_loadu_ps(arr.as_ptr().add(i));

                    // Update index
                    // if true, bit 0 to 31 are all set to 1
                    let mask = _mm_castps_si128(
                        _mm_cmp_ps(vals, min, _CMP_LT_OQ)
                    );
                    // blend is const mask
                    // blendv support dynamic mask
                    idx = _mm_blendv_epi8(idx, cur, mask);

                    // Update value
                    min = _mm_min_ps(vals, min);

                    // increment cur
                    let four = _mm_set1_epi32(4);
                    cur = _mm_add_epi32(cur, four);
                }

                let mut val_result: [f32; 4] = [0f32; 4];
                let mut idx_result: [u32; 4] = [0; 4];
                _mm_storeu_ps(val_result.as_mut_ptr(), min);
                _mm_storeu_ps(idx_result.as_mut_ptr() as *mut f32, _mm_castsi128_ps(idx));

                let mut result_idx = idx_result[0];
                let mut result_min = val_result[0];
                for i in 1..4 {
                    if val_result[i] < result_min {
                        result_min = val_result[i];
                        result_idx = idx_result[i];
                    }
                }

                for i in len_aligned_to_instruction..arr.len() {
                    if arr[i] < result_min {
                        result_min = arr[i];
                        result_idx = i as u32;
                    }
                }

                result_idx
            }
        }
    }
}

mod tests {
    use super::*;
    #[test]
    fn test_argmin() {
        let arr = [1f32, 2f32, 3f32, 4f32, 5f32, 6f32, 7f32, 8f32];
        assert_eq!(arr.argmin(), 0);
        let arr = [8f32, 7f32, 6f32, 5f32, 4f32, 3f32, 2f32, 1f32];
        assert_eq!(arr.argmin(), 7);
        let arr = [1f32, 2f32, 3f32, 0f32, 5f32, 6f32, 1f32, 8f32];
        assert_eq!(arr.argmin(), 3);
    }
}
