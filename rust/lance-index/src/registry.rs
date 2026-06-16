// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

#[cfg(feature = "geo")]
use crate::scalar::rtree::RTreeIndexPlugin;
use crate::{
    pb, pbold,
    scalar::{
        bitmap::BitmapIndexPlugin, bloomfilter::BloomFilterIndexPlugin, btree::BTreeIndexPlugin,
        fmindex::FMIndexPlugin, inverted::InvertedIndexPlugin, json::JsonIndexPlugin,
        label_list::LabelListIndexPlugin, ngram::NGramIndexPlugin, zonemap::ZoneMapIndexPlugin,
    },
};

/// Derive a human-readable index type name from a details type URL.
///
/// The display name is the final `.`-separated segment of the type URL with any
/// trailing `IndexDetails` removed. For example, `/lance.index.pb.VectorIndexDetails`
/// yields `Vector`. Used as a best-effort fallback when no plugin is registered
/// for the type URL, so the index type is never reported as opaque "Unknown"
/// while valid index details exist.
pub fn display_type_from_url(type_url: &str) -> &str {
    let segment = type_url.rsplit('.').next().unwrap_or(type_url);
    segment
        .strip_suffix("IndexDetails")
        .filter(|stripped| !stripped.is_empty())
        .unwrap_or(segment)
}

pub use lance_index_core::registry::{IndexPluginRegistry, PluginRegistry};

/// Create a registry populated with all built-in index plugins.
pub fn with_default_plugins() -> Arc<IndexPluginRegistry> {
    let mut registry = IndexPluginRegistry::new();
    registry.add_plugin::<pbold::BTreeIndexDetails, BTreeIndexPlugin>();
    registry.add_plugin::<pbold::BitmapIndexDetails, BitmapIndexPlugin>();
    registry.add_plugin::<pbold::LabelListIndexDetails, LabelListIndexPlugin>();
    registry.add_plugin::<pbold::NGramIndexDetails, NGramIndexPlugin>();
    registry.add_plugin::<pbold::ZoneMapIndexDetails, ZoneMapIndexPlugin>();
    registry.add_plugin::<pb::BloomFilterIndexDetails, BloomFilterIndexPlugin>();
    registry.add_plugin::<pbold::InvertedIndexDetails, InvertedIndexPlugin>();
    registry.add_plugin::<pb::JsonIndexDetails, JsonIndexPlugin>();
    registry.add_plugin::<pb::FmIndexIndexDetails, FMIndexPlugin>();
    #[cfg(feature = "geo")]
    registry.add_plugin::<pb::RTreeIndexDetails, RTreeIndexPlugin>();

    let registry = Arc::new(registry);
    let registry_dyn: Arc<dyn PluginRegistry> = registry.clone();
    registry.for_each_plugin(|p| p.attach_registry(registry_dyn.clone()));

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_type_from_url() {
        assert_eq!(
            display_type_from_url("/lance.index.pb.VectorIndexDetails"),
            "Vector"
        );
        assert_eq!(display_type_from_url("BTreeIndexDetails"), "BTree");
        // Segment without the IndexDetails suffix is returned verbatim.
        assert_eq!(
            display_type_from_url("/lance.pb.SomethingElse"),
            "SomethingElse"
        );
        // A bare "IndexDetails" segment has nothing left after stripping, so it
        // is returned as-is rather than an empty string.
        assert_eq!(display_type_from_url("IndexDetails"), "IndexDetails");
        assert_eq!(display_type_from_url(""), "");
    }

    #[test]
    fn test_get_plugin_by_name_accepts_case_insensitive_builtin_names() {
        let registry = with_default_plugins();

        for (requested_name, expected_name) in [
            ("BTREE", "BTree"),
            ("Bitmap", "Bitmap"),
            ("INVERTED", "Inverted"),
            ("NGRAM", "NGram"),
            ("ZONEMAP", "ZoneMap"),
            ("BLOOMFILTER", "BloomFilter"),
            ("FMINDEX", "Fm"),
            ("JSON", "Json"),
        ] {
            let plugin = registry.get_plugin_by_name(requested_name).unwrap();
            assert_eq!(plugin.name(), expected_name);
        }
    }
}
