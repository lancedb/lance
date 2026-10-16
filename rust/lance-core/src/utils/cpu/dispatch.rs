// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::cpu::CpuFeatures;

    type SumFn = unsafe fn(&[f32], &[f32]) -> f32;
    type FillFn = unsafe fn(&[f32], &[f32], usize, &mut [f32]);

    fn portable_sum(x: &[f32], y: &[f32]) -> f32 {
        x.iter().chain(y).sum()
    }

    fn doubled_sum(x: &[f32], y: &[f32]) -> f32 {
        2.0 * portable_sum(x, y)
    }

    crate::target_kernel! {
        #[target_features()]
        #[candidate(PORTABLE_SUM: SumFn)]
        fn macro_sum(x: &[f32], y: &[f32]) -> f32 {
            portable_sum(x, y)
        }
    }

    crate::target_kernel! {
        #[target_features()]
        #[candidate(PORTABLE_FILL: FillFn)]
        fn macro_fill(x: &[f32], batch: &[f32], dimension: usize, out: &mut [f32]) {
            for (o, y) in out.iter_mut().zip(batch.chunks_exact(dimension)) {
                *o = portable_sum(x, y);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    crate::target_kernel! {
        #[target_features("avx2", "fma")]
        #[candidate(ARCH_SUM: SumFn)]
        fn arch_sum(x: &[f32], y: &[f32]) -> f32 {
            portable_sum(x, y)
        }
    }

    #[cfg(target_arch = "aarch64")]
    crate::target_kernel! {
        #[target_features("neon")]
        #[candidate(ARCH_SUM: SumFn)]
        fn arch_sum(x: &[f32], y: &[f32]) -> f32 {
            portable_sum(x, y)
        }
    }

    fn feature_missing_from_host() -> CpuFeatures {
        let host = CpuFeatures::host();
        CpuFeatures::all()
            .find(|feature| !host.contains(*feature))
            .expect("every host lacks the features of the other architectures")
    }

    #[test]
    fn select_takes_the_first_candidate_the_host_supports() {
        let missing = feature_missing_from_host();
        // SAFETY: the kernels are plain Rust functions without target features.
        let candidates: [Candidate<SumFn>; 2] = unsafe {
            [
                Candidate::new("unsupported", missing, doubled_sum),
                Candidate::new("portable", CpuFeatures::empty(), portable_sum),
            ]
        };
        assert_eq!(select(&candidates, CpuFeatures::host()), 1);
        assert_eq!(select(&candidates, missing), 0);
    }

    #[test]
    fn unsupported_candidate_yields_no_kernel() {
        // SAFETY: the kernel is a plain Rust function without target features.
        let candidate: Candidate<SumFn> =
            unsafe { Candidate::new("unsupported", feature_missing_from_host(), doubled_sum) };
        assert!(candidate.kernel().is_none());
    }

    crate::dispatch! {
        static PORTABLE_ONLY: Dispatch<SumFn> = &[PORTABLE_SUM];
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    crate::dispatch! {
        static ARCH_FIRST: Dispatch<SumFn> = &[ARCH_SUM, PORTABLE_SUM];
    }

    crate::dispatch! {
        static FILL: Dispatch<FillFn> = &[PORTABLE_FILL];
    }

    #[test]
    fn portable_only_dispatch_is_static_and_callable() {
        assert!(PORTABLE_ONLY.is_static());
        assert_eq!(PORTABLE_ONLY.get().call(&[1.0, 2.0], &[3.0]), 6.0);
        assert_eq!(PORTABLE_ONLY.selected().name(), "macro_sum");
    }

    #[test]
    fn slice_output_kernel_is_callable() {
        let mut out = [0.0; 2];
        FILL.get().call(&[1.0], &[2.0, 3.0], 1, &mut out);
        assert_eq!(out, [3.0, 4.0]);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn macro_mask_matches_declared_target_features() {
        #[cfg(target_arch = "x86_64")]
        let names: &[&str] = &["avx2", "fma"];
        #[cfg(target_arch = "aarch64")]
        let names: &[&str] = &["neon"];
        assert_eq!(
            ARCH_SUM.features(),
            CpuFeatures::from_target_features(names)
        );
        assert_eq!(ARCH_SUM.name(), "arch_sum");
        assert!(PORTABLE_SUM.features().is_empty());

        let is_supported = CpuFeatures::host().contains(ARCH_SUM.features());
        assert_eq!(ARCH_FIRST.selected().name() == "arch_sum", is_supported);
        assert_eq!(
            ARCH_FIRST.is_static(),
            CpuFeatures::baseline().contains(ARCH_SUM.features())
        );
        for candidate in ARCH_FIRST.candidates() {
            if let Some(kernel) = candidate.kernel() {
                assert_eq!(kernel.call(&[1.0], &[2.0]), 3.0, "{}", candidate.name());
            }
        }
    }
}
