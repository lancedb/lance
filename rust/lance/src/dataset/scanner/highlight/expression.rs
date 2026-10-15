// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_array::{RecordBatch, cast::AsArray, types::UInt32Type};
use arrow_schema::{FieldRef, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::ColumnarValue;
use datafusion_physical_expr::{PhysicalExpr, expressions::col};
use lance_index::scalar::inverted::DOC_INDEX_COL;
use lance_index::scalar::inverted::highlight::{
    HighlightDocument, PreparedHighlightQuery, highlights_array, highlights_field,
};

use crate::index::scalar::inverted::ResolvedFtsField;

#[derive(Debug)]
struct PreparedHighlight {
    query: PreparedHighlightQuery,
    sources: Vec<ResolvedFtsField>,
    schema: SchemaRef,
}

/// Computes ranges from explicit column expressions so projection rewrites can
/// prune and reorder input columns without changing source names or coordinates.
#[derive(Debug, Clone)]
pub(super) struct HighlightExpr {
    prepared: Arc<PreparedHighlight>,
    inputs: Vec<Arc<dyn PhysicalExpr>>,
}

impl HighlightExpr {
    pub(super) fn try_new(
        query: PreparedHighlightQuery,
        sources: Vec<ResolvedFtsField>,
        schema: &Schema,
    ) -> Result<Self> {
        let mut names = sources
            .iter()
            .map(|source| source.root_column.as_str())
            .collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        if schema.field_with_name(DOC_INDEX_COL).is_ok() {
            names.push(DOC_INDEX_COL);
        }
        let inputs = names
            .iter()
            .map(|name| col(name, schema))
            .collect::<Result<Vec<_>>>()?;
        let fields = names
            .iter()
            .map(|name| schema.field_with_name(name).cloned())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            prepared: Arc::new(PreparedHighlight {
                query,
                sources,
                schema: Arc::new(Schema::new(fields)),
            }),
            inputs,
        })
    }
}

impl PartialEq for HighlightExpr {
    fn eq(&self, other: &Self) -> bool {
        // Analyzer trait objects have no semantic equality. Rewrites share the
        // same prepared query, while independently prepared queries stay distinct.
        Arc::ptr_eq(&self.prepared, &other.prepared) && self.inputs == other.inputs
    }
}

impl Eq for HighlightExpr {}

impl Hash for HighlightExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.prepared).hash(state);
        self.inputs.hash(state);
    }
}

impl fmt::Display for HighlightExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "highlight(")?;
        for (i, input) in self.inputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{input}")?;
        }
        write!(f, ")")
    }
}

impl PhysicalExpr for HighlightExpr {
    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }

    fn return_field(&self, _: &Schema) -> Result<FieldRef> {
        Ok(Arc::new(highlights_field()))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let columns = self
            .inputs
            .iter()
            .map(|input| input.evaluate(batch)?.into_array(batch.num_rows()))
            .collect::<Result<Vec<_>>>()?;
        let source_batch = RecordBatch::try_new(self.prepared.schema.clone(), columns)?;
        let mut rows = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            let coordinate = source_batch
                .column_by_name(DOC_INDEX_COL)
                .map(|array| array.as_list::<i32>().value(row));
            let doc_index = coordinate
                .as_ref()
                .map(|array| array.as_primitive::<UInt32Type>().values().as_ref())
                .unwrap_or(&[]);
            let documents = self
                .prepared
                .sources
                .iter()
                .map(|source| {
                    Ok(HighlightDocument {
                        field: source.canonical_path.clone(),
                        values: source.highlight_values(&source_batch, row, doc_index)?,
                    })
                })
                .collect::<lance_core::Result<Vec<_>>>()?;
            rows.push(self.prepared.query.highlight(&documents).map_err(|error| {
                let message = format!("highlight batch row {row}: {error}");
                match error {
                    lance_core::Error::InvalidInput { .. } => {
                        lance_core::Error::invalid_input(message)
                    }
                    _ => lance_core::Error::internal(message),
                }
            })?);
        }
        Ok(ColumnarValue::Array(highlights_array(&rows)?))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.inputs.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if children.len() != self.inputs.len() {
            return Err(DataFusionError::Internal(format!(
                "highlight expected {} input expressions, received {}",
                self.inputs.len(),
                children.len()
            )));
        }
        Ok(Arc::new(Self {
            prepared: self.prepared.clone(),
            inputs: children,
        }))
    }
}
