// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build script for lance-c.
//!
//! Optionally generates `include/lance.h` via cbindgen.
//! If cbindgen is not available, the pre-committed header is used.

fn main() {
    // cbindgen header generation is optional.
    // Run `cargo install cbindgen && cbindgen --crate lance-c -o include/lance.h`
    // to regenerate the header manually.
}
