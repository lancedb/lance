// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Protobuf definitions for encodings
pub mod pb {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.encodings.rs"));
}
