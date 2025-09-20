// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Graph configuration for mapping Lance datasets to property graphs

use crate::error::{GraphError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for mapping Lance datasets to property graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Mapping of node labels to their field configurations
    pub node_mappings: HashMap<String, NodeMapping>,
    /// Mapping of relationship types to their field configurations  
    pub relationship_mappings: HashMap<String, RelationshipMapping>,
    /// Default node ID field if not specified in mappings
    pub default_node_id_field: String,
    /// Default relationship type field if not specified in mappings
    pub default_relationship_type_field: String,
}

/// Configuration for mapping node labels to dataset fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMapping {
    /// The node label (e.g., "Person", "Product")
    pub label: String,
    /// Field name that serves as the node identifier
    pub id_field: String,
    /// Optional fields that define node properties
    pub property_fields: Vec<String>,
    /// Optional filter conditions for this node type
    pub filter_conditions: Option<String>,
}

/// Configuration for mapping relationship types to dataset fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipMapping {
    /// The relationship type (e.g., "KNOWS", "PURCHASED")
    pub relationship_type: String,
    /// Field containing the source node ID
    pub source_id_field: String,
    /// Field containing the target node ID
    pub target_id_field: String,
    /// Optional field containing the relationship type
    pub type_field: Option<String>,
    /// Optional fields that define relationship properties
    pub property_fields: Vec<String>,
    /// Optional filter conditions for this relationship type
    pub filter_conditions: Option<String>,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            node_mappings: HashMap::new(),
            relationship_mappings: HashMap::new(),
            default_node_id_field: "id".to_string(),
            default_relationship_type_field: "type".to_string(),
        }
    }
}

impl GraphConfig {
    /// Create a new builder for GraphConfig
    pub fn builder() -> GraphConfigBuilder {
        GraphConfigBuilder::new()
    }

    /// Get node mapping for a given label
    pub fn get_node_mapping(&self, label: &str) -> Option<&NodeMapping> {
        self.node_mappings.get(label)
    }

    /// Get relationship mapping for a given type
    pub fn get_relationship_mapping(&self, rel_type: &str) -> Option<&RelationshipMapping> {
        self.relationship_mappings.get(rel_type)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Check for conflicting field names
        for (label, mapping) in &self.node_mappings {
            if mapping.id_field.is_empty() {
                return Err(GraphError::ConfigError {
                    message: format!("Node mapping for '{}' has empty id_field", label),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        }

        for (rel_type, mapping) in &self.relationship_mappings {
            if mapping.source_id_field.is_empty() || mapping.target_id_field.is_empty() {
                return Err(GraphError::ConfigError {
                    message: format!(
                        "Relationship mapping for '{}' has empty source or target id field",
                        rel_type
                    ),
                    location: snafu::Location::new(file!(), line!(), column!()),
                });
            }
        }

        Ok(())
    }
}

/// Builder for GraphConfig
#[derive(Debug, Default, Clone)]
pub struct GraphConfigBuilder {
    node_mappings: HashMap<String, NodeMapping>,
    relationship_mappings: HashMap<String, RelationshipMapping>,
    default_node_id_field: Option<String>,
    default_relationship_type_field: Option<String>,
}

impl GraphConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node label mapping
    pub fn with_node_label<S: Into<String>>(mut self, label: S, id_field: S) -> Self {
        let label_str = label.into();
        self.node_mappings.insert(
            label_str.clone(),
            NodeMapping {
                label: label_str,
                id_field: id_field.into(),
                property_fields: Vec::new(),
                filter_conditions: None,
            },
        );
        self
    }

    /// Add a node mapping with additional configuration
    pub fn with_node_mapping(mut self, mapping: NodeMapping) -> Self {
        self.node_mappings.insert(mapping.label.clone(), mapping);
        self
    }

    /// Add a relationship type mapping
    pub fn with_relationship<S: Into<String>>(
        mut self,
        rel_type: S,
        source_field: S,
        target_field: S,
    ) -> Self {
        let type_str = rel_type.into();
        self.relationship_mappings.insert(
            type_str.clone(),
            RelationshipMapping {
                relationship_type: type_str,
                source_id_field: source_field.into(),
                target_id_field: target_field.into(),
                type_field: None,
                property_fields: Vec::new(),
                filter_conditions: None,
            },
        );
        self
    }

    /// Add a relationship mapping with additional configuration
    pub fn with_relationship_mapping(mut self, mapping: RelationshipMapping) -> Self {
        self.relationship_mappings
            .insert(mapping.relationship_type.clone(), mapping);
        self
    }

    /// Set the default node ID field
    pub fn with_default_node_id_field<S: Into<String>>(mut self, field: S) -> Self {
        self.default_node_id_field = Some(field.into());
        self
    }

    /// Set the default relationship type field
    pub fn with_default_relationship_type_field<S: Into<String>>(mut self, field: S) -> Self {
        self.default_relationship_type_field = Some(field.into());
        self
    }

    /// Build the GraphConfig
    pub fn build(self) -> Result<GraphConfig> {
        let config = GraphConfig {
            node_mappings: self.node_mappings,
            relationship_mappings: self.relationship_mappings,
            default_node_id_field: self
                .default_node_id_field
                .unwrap_or_else(|| "id".to_string()),
            default_relationship_type_field: self
                .default_relationship_type_field
                .unwrap_or_else(|| "type".to_string()),
        };

        config.validate()?;
        Ok(config)
    }
}

impl NodeMapping {
    /// Create a new node mapping
    pub fn new<S: Into<String>>(label: S, id_field: S) -> Self {
        Self {
            label: label.into(),
            id_field: id_field.into(),
            property_fields: Vec::new(),
            filter_conditions: None,
        }
    }

    /// Add property fields to the mapping
    pub fn with_properties(mut self, fields: Vec<String>) -> Self {
        self.property_fields = fields;
        self
    }

    /// Add filter conditions for this node type
    pub fn with_filter<S: Into<String>>(mut self, filter: S) -> Self {
        self.filter_conditions = Some(filter.into());
        self
    }
}

impl RelationshipMapping {
    /// Create a new relationship mapping
    pub fn new<S: Into<String>>(rel_type: S, source_field: S, target_field: S) -> Self {
        Self {
            relationship_type: rel_type.into(),
            source_id_field: source_field.into(),
            target_id_field: target_field.into(),
            type_field: None,
            property_fields: Vec::new(),
            filter_conditions: None,
        }
    }

    /// Set the type field for this relationship
    pub fn with_type_field<S: Into<String>>(mut self, type_field: S) -> Self {
        self.type_field = Some(type_field.into());
        self
    }

    /// Add property fields to the mapping
    pub fn with_properties(mut self, fields: Vec<String>) -> Self {
        self.property_fields = fields;
        self
    }

    /// Add filter conditions for this relationship type
    pub fn with_filter<S: Into<String>>(mut self, filter: S) -> Self {
        self.filter_conditions = Some(filter.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_config_builder() {
        let config = GraphConfig::builder()
            .with_node_label("Person", "person_id")
            .with_node_label("Company", "company_id")
            .with_relationship("WORKS_FOR", "person_id", "company_id")
            .build()
            .unwrap();

        assert_eq!(config.node_mappings.len(), 2);
        assert_eq!(config.relationship_mappings.len(), 1);

        let person_mapping = config.get_node_mapping("Person").unwrap();
        assert_eq!(person_mapping.id_field, "person_id");

        let works_for_mapping = config.get_relationship_mapping("WORKS_FOR").unwrap();
        assert_eq!(works_for_mapping.source_id_field, "person_id");
        assert_eq!(works_for_mapping.target_id_field, "company_id");
    }

    #[test]
    fn test_validation_empty_id_field() {
        let mut config = GraphConfig::default();
        config.node_mappings.insert(
            "Person".to_string(),
            NodeMapping {
                label: "Person".to_string(),
                id_field: "".to_string(),
                property_fields: Vec::new(),
                filter_conditions: None,
            },
        );

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_node_mapping_with_properties() {
        let mapping = NodeMapping::new("Person", "id")
            .with_properties(vec!["name".to_string(), "age".to_string()])
            .with_filter("age > 18".to_string());

        assert_eq!(mapping.property_fields.len(), 2);
        assert!(mapping.filter_conditions.is_some());
    }
}
