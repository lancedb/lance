// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub(crate) mod catalog_trait;
pub(crate) mod dataset_identifier;
pub(crate) mod namespace;

pub use catalog_trait::Catalog;
pub use dataset_identifier::DatasetIdentifier;
pub use namespace::Namespace;
