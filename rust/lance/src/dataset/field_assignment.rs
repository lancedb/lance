// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lazy I/O for snapshot-level field assignment state.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Cursor;
use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{Expr, ScalarUDF, col};
use datafusion::scalar::ScalarValue;
use lance_datafusion::udf::{AssignmentFragment, IS_ASSIGNED_NAME, bound_is_assigned_udf};
use lance_table::format::{
    FieldAssignmentFile, FieldAssignmentFragment, FieldAssignmentFragmentState,
    FieldAssignmentRoot, FieldAssignmentState, pb,
};
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::Dataset;
use super::transaction::{
    FieldAssignmentFragmentState as TransactionFieldAssignmentFragmentState,
    FieldAssignmentFragmentValue, Operation, Transaction, UpdateMode,
};
use crate::{Error, Result};
use lance_core::utils::address::RowAddress;
use lance_io::object_store::ObjectStore;

pub const FIELD_ASSIGNMENTS_DIR: &str = "_field_assignments";
const FIELD_ASSIGNMENT_ROOTS_DIR: &str = "roots";
const FIELD_ASSIGNMENT_BITMAPS_DIR: &str = "bitmaps";

impl Dataset {
    pub(crate) fn tracked_field_ids(&self) -> impl Iterator<Item = i32> + '_ {
        self.manifest
            .field_assignment_states
            .iter()
            .map(|state| state.field_id)
    }

    pub(crate) fn field_assignment_state(&self, field_id: i32) -> Option<&FieldAssignmentState> {
        self.manifest
            .field_assignment_states
            .binary_search_by_key(&field_id, |state| state.field_id)
            .ok()
            .map(|index| &self.manifest.field_assignment_states[index])
    }

    fn field_assignment_path(&self, file: &FieldAssignmentFile) -> Result<Path> {
        let relative = Path::parse(file.path.as_str())?;
        match file.base_id {
            Some(base_id) => {
                let base_path = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Dataset base path with ID {} not found for field assignment file '{}'",
                        base_id, file.path
                    ))
                })?;
                if !base_path.is_dataset_root {
                    return Err(Error::invalid_input(format!(
                        "Dataset base path with ID {} is not a dataset root for field assignment file '{}'",
                        base_id, file.path
                    )));
                }
                let base = base_path.extract_path(self.session.store_registry())?;
                Ok(Path::from_iter(base.parts().chain(relative.parts())))
            }
            None => Ok(Path::from_iter(self.base.parts().chain(relative.parts()))),
        }
    }

    async fn read_field_assignment_file(&self, file: &FieldAssignmentFile) -> Result<Vec<u8>> {
        let store = self.object_store(file.base_id).await?;
        let path = self.field_assignment_path(file)?;
        let bytes = store.read_one_all(&path).await?;
        if bytes.len() as u64 != file.size_bytes {
            return Err(Error::invalid_input(format!(
                "Field assignment file '{}' has size {}, expected {}",
                file.path,
                bytes.len(),
                file.size_bytes
            )));
        }
        Ok(bytes.to_vec())
    }

    /// Load one field's immutable assignment root on demand.
    pub(crate) async fn load_field_assignment_root(
        &self,
        field_id: i32,
    ) -> Result<Option<FieldAssignmentRoot>> {
        let Some(descriptor) = self.field_assignment_state(field_id) else {
            return Ok(None);
        };
        let bytes = self.read_field_assignment_file(&descriptor.root).await?;
        let proto = pb::FieldAssignmentRoot::decode(bytes.as_slice()).map_err(|error| {
            Error::invalid_input(format!(
                "Failed to decode field assignment root '{}' for field ID {}: {}",
                descriptor.root.path, field_id, error
            ))
        })?;
        let mut root = FieldAssignmentRoot::try_from(proto)?;

        for entry in &mut root.fragments {
            let fragment = self
                .manifest
                .fragments
                .iter()
                .find(|fragment| fragment.id == entry.fragment_id)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Field assignment root for field ID {} references unknown fragment {}",
                        field_id, entry.fragment_id
                    ))
                })?;
            let physical_rows = fragment.physical_rows.ok_or_else(|| {
                Error::invalid_input(format!(
                    "Fragment {} has no physical row count for field assignment",
                    entry.fragment_id
                ))
            })?;
            if entry.physical_rows != physical_rows as u64 {
                return Err(Error::invalid_input(format!(
                    "Field assignment root for field ID {} records {} physical rows for fragment {}, manifest records {}",
                    field_id, entry.physical_rows, entry.fragment_id, physical_rows
                )));
            }
            if let FieldAssignmentFragmentState::Partial(file) = &mut entry.state
                && file.base_id.is_none()
            {
                // Bitmap paths are relative to the dataset that owns the root.
                // Materialize that inheritance before a future root reuses the page.
                file.base_id = descriptor.root.base_id;
            }
        }
        Ok(Some(root))
    }

    /// Load and validate a partial assignment bitmap.
    pub(crate) async fn load_field_assignment_bitmap(
        &self,
        field_id: i32,
        fragment: &FieldAssignmentFragment,
    ) -> Result<RoaringBitmap> {
        let FieldAssignmentFragmentState::Partial(file) = &fragment.state else {
            return Err(Error::internal(format!(
                "Field assignment bitmap requested for non-partial fragment {}",
                fragment.fragment_id
            )));
        };
        let bytes = self.read_field_assignment_file(file).await?;
        let bitmap = RoaringBitmap::deserialize_from(&mut Cursor::new(bytes)).map_err(|error| {
            Error::invalid_input(format!(
                "Failed to decode assignment bitmap '{}' for field ID {}, fragment {}: {}",
                file.path, field_id, fragment.fragment_id, error
            ))
        })?;
        if bitmap.is_empty() || bitmap.len() == fragment.physical_rows {
            return Err(Error::invalid_input(format!(
                "Partial assignment bitmap '{}' for field ID {}, fragment {} must be non-empty and non-full",
                file.path, field_id, fragment.fragment_id
            )));
        }
        if bitmap
            .max()
            .is_some_and(|offset| offset as u64 >= fragment.physical_rows)
        {
            return Err(Error::invalid_input(format!(
                "Assignment bitmap '{}' for field ID {}, fragment {} contains an out-of-bounds row offset",
                file.path, field_id, fragment.fragment_id
            )));
        }
        Ok(bitmap)
    }

    pub(crate) async fn write_field_assignment_bitmap(
        &self,
        write_store: &ObjectStore,
        field_id: i32,
        fragment_id: u64,
        physical_rows: u64,
        bitmap: &RoaringBitmap,
    ) -> Result<FieldAssignmentFile> {
        if bitmap.is_empty() || bitmap.len() == physical_rows {
            return Err(Error::internal(format!(
                "Only partial assignment bitmaps may be written for field ID {}, fragment {}",
                field_id, fragment_id
            )));
        }
        if bitmap
            .max()
            .is_some_and(|offset| offset as u64 >= physical_rows)
        {
            return Err(Error::internal(format!(
                "Assignment bitmap for field ID {}, fragment {} contains an out-of-bounds row offset",
                field_id, fragment_id
            )));
        }
        let mut bytes = Vec::with_capacity(bitmap.serialized_size());
        bitmap.serialize_into(&mut bytes)?;
        let relative = Path::from(FIELD_ASSIGNMENTS_DIR)
            .join(FIELD_ASSIGNMENT_BITMAPS_DIR)
            .join(field_id.to_string())
            .join(fragment_id.to_string())
            .join(format!("{}.rbm", Uuid::new_v4()));
        let full_path = Path::from_iter(self.base.parts().chain(relative.parts()));
        write_store.put(&full_path, &bytes).await?;
        Ok(FieldAssignmentFile {
            path: relative.to_string(),
            size_bytes: bytes.len() as u64,
            base_id: None,
        })
    }

    pub(crate) async fn write_field_assignment_root(
        &self,
        write_store: &ObjectStore,
        field_id: i32,
        mut root: FieldAssignmentRoot,
    ) -> Result<FieldAssignmentState> {
        root.fragments.sort_by_key(|entry| entry.fragment_id);
        if root
            .fragments
            .windows(2)
            .any(|entries| entries[0].fragment_id == entries[1].fragment_id)
        {
            return Err(Error::internal(format!(
                "Field assignment root for field ID {} contains duplicate fragment IDs",
                field_id
            )));
        }
        let bytes = pb::FieldAssignmentRoot::from(&root).encode_to_vec();
        let relative = Path::from(FIELD_ASSIGNMENTS_DIR)
            .join(FIELD_ASSIGNMENT_ROOTS_DIR)
            .join(field_id.to_string())
            .join(format!("{}.root", Uuid::new_v4()));
        let full_path = Path::from_iter(self.base.parts().chain(relative.parts()));
        write_store.put(&full_path, &bytes).await?;
        Ok(FieldAssignmentState {
            field_id,
            root: FieldAssignmentFile {
                path: relative.to_string(),
                size_bytes: bytes.len() as u64,
                base_id: None,
            },
        })
    }

    /// Build exact states for newly inserted fragments from the stable field
    /// IDs supplied by the logical write.
    pub(crate) fn field_assignment_states_for_new_fragments(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        supplied_field_ids: &HashSet<i32>,
    ) -> Result<Vec<TransactionFieldAssignmentFragmentState>> {
        let mut states = Vec::new();
        for field_id in self.tracked_field_ids() {
            let state = if supplied_field_ids.contains(&field_id) {
                FieldAssignmentFragmentValue::All
            } else {
                FieldAssignmentFragmentValue::None
            };
            for fragment in new_fragments {
                states.push(TransactionFieldAssignmentFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    field_id,
                    state: state.clone(),
                });
            }
        }
        Ok(states)
    }

    /// Remap assignment membership from rows in the current snapshot to newly
    /// written fragments. Logical writes and invalidations override the source
    /// state for their fields.
    pub(crate) async fn field_assignment_states_for_rewritten_rows(
        &self,
        new_fragments: &[lance_table::format::Fragment],
        source_row_addresses: &[u64],
        assigned_field_ids: &HashSet<i32>,
        invalidated_field_ids: &HashSet<i32>,
    ) -> Result<Vec<TransactionFieldAssignmentFragmentState>> {
        if !assigned_field_ids.is_disjoint(invalidated_field_ids) {
            return Err(Error::invalid_input(
                "A field cannot be assigned and invalidated by the same mutation",
            ));
        }
        let output_rows = new_fragments.iter().try_fold(0usize, |total, fragment| {
            let rows = usize::try_from(fragment_physical_rows(fragment)?).map_err(|_| {
                Error::invalid_input(format!(
                    "Fragment {} has too many rows for this platform",
                    fragment.id
                ))
            })?;
            total
                .checked_add(rows)
                .ok_or_else(|| Error::invalid_input("Rewritten row count overflow"))
        })?;
        if output_rows != source_row_addresses.len() {
            return Err(Error::internal(format!(
                "Rewritten fragments contain {} rows but {} source row addresses were captured",
                output_rows,
                source_row_addresses.len()
            )));
        }

        let source_fragment_ids = source_row_addresses
            .iter()
            .map(|address| RowAddress::from(*address).fragment_id() as u64)
            .collect::<HashSet<_>>();
        let mut states = Vec::new();
        for field_id in self.tracked_field_ids() {
            if assigned_field_ids.contains(&field_id) || invalidated_field_ids.contains(&field_id) {
                let state = if assigned_field_ids.contains(&field_id) {
                    FieldAssignmentFragmentValue::All
                } else {
                    FieldAssignmentFragmentValue::None
                };
                for fragment in new_fragments {
                    states.push(TransactionFieldAssignmentFragmentState {
                        fragment_path: fragment_path(fragment)?.to_string(),
                        field_id,
                        state: state.clone(),
                    });
                }
                continue;
            }

            let root = self
                .load_field_assignment_root(field_id)
                .await?
                .expect("tracked field must have an assignment root");
            let mut source_states = HashMap::new();
            for fragment in root
                .fragments
                .into_iter()
                .filter(|fragment| source_fragment_ids.contains(&fragment.fragment_id))
            {
                let state = match &fragment.state {
                    FieldAssignmentFragmentState::All => None,
                    FieldAssignmentFragmentState::Partial(_) => Some(
                        self.load_field_assignment_bitmap(field_id, &fragment)
                            .await?,
                    ),
                };
                source_states.insert(fragment.fragment_id, state);
            }

            let mut source_cursor = 0usize;
            for fragment in new_fragments {
                let physical_rows =
                    usize::try_from(fragment_physical_rows(fragment)?).map_err(|_| {
                        Error::invalid_input(format!(
                            "Fragment {} has too many rows for this platform",
                            fragment.id
                        ))
                    })?;
                let mut bitmap = RoaringBitmap::new();
                for (new_offset, address) in source_row_addresses
                    [source_cursor..source_cursor + physical_rows]
                    .iter()
                    .enumerate()
                {
                    let address = RowAddress::from(*address);
                    let assigned = match source_states.get(&(address.fragment_id() as u64)) {
                        Some(None) => true,
                        Some(Some(source_bitmap)) => source_bitmap.contains(address.row_offset()),
                        None => false,
                    };
                    if assigned {
                        bitmap.insert(new_offset as u32);
                    }
                }
                source_cursor += physical_rows;
                let state = if bitmap.is_empty() {
                    FieldAssignmentFragmentValue::None
                } else if bitmap.len() == physical_rows as u64 {
                    FieldAssignmentFragmentValue::All
                } else {
                    FieldAssignmentFragmentValue::Partial(bitmap)
                };
                states.push(TransactionFieldAssignmentFragmentState {
                    fragment_path: fragment_path(fragment)?.to_string(),
                    field_id,
                    state,
                });
            }
        }
        Ok(states)
    }
}

fn field_reference_segments(expr: &Expr) -> Result<Vec<String>> {
    match expr {
        Expr::Column(column) if column.relation.is_none() => Ok(vec![column.name.clone()]),
        Expr::ScalarFunction(function) if function.func.name() == "get_field" => {
            if function.args.len() != 2 {
                return Err(Error::invalid_input(format!(
                    "is_assigned field reference contains get_field with {} arguments, expected 2",
                    function.args.len()
                )));
            }
            let mut segments = field_reference_segments(&function.args[0])?;
            let child = match &function.args[1] {
                Expr::Literal(ScalarValue::Utf8(Some(value)), _)
                | Expr::Literal(ScalarValue::LargeUtf8(Some(value)), _)
                | Expr::Literal(ScalarValue::Utf8View(Some(value)), _) => value.clone(),
                _ => {
                    return Err(Error::invalid_input(
                        "is_assigned(field) requires a direct field reference",
                    ));
                }
            };
            segments.push(child);
            Ok(segments)
        }
        _ => Err(Error::invalid_input(
            "is_assigned(field) requires exactly one direct field reference",
        )),
    }
}

fn is_assignment_call(expr: &Expr) -> Result<Option<Vec<String>>> {
    let Expr::ScalarFunction(function) = expr else {
        return Ok(None);
    };
    if !function.func.name().eq_ignore_ascii_case(IS_ASSIGNED_NAME) {
        return Ok(None);
    }
    if function.args.len() != 1 {
        return Err(Error::invalid_input(format!(
            "is_assigned expects exactly one field reference, received {} arguments",
            function.args.len()
        )));
    }
    field_reference_segments(&function.args[0]).map(Some)
}

pub fn expression_references_field_assignment(expression: &Expr) -> Result<bool> {
    let mut references_assignment = false;
    let mut captured_error = None;
    expression
        .apply(|node| {
            match is_assignment_call(node) {
                Ok(Some(_)) => references_assignment = true,
                Ok(None) => {}
                Err(error) => {
                    captured_error = Some(error);
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .map_err(Error::from)?;
    if let Some(error) = captured_error {
        return Err(error);
    }
    Ok(references_assignment)
}

fn resolve_assignment_field(dataset: &Dataset, segments: &[String]) -> Result<i32> {
    let Some(first) = segments.first() else {
        return Err(Error::invalid_input(
            "is_assigned(field) requires a non-empty field reference",
        ));
    };
    let mut field = dataset
        .schema()
        .fields
        .iter()
        .find(|field| field.name == *first)
        .or_else(|| {
            dataset
                .schema()
                .fields
                .iter()
                .find(|field| field.name.eq_ignore_ascii_case(first))
        })
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "is_assigned references unknown field '{}'",
                segments.join(".")
            ))
        })?;
    for segment in &segments[1..] {
        field = field
            .children
            .iter()
            .find(|child| child.name == *segment)
            .or_else(|| {
                field
                    .children
                    .iter()
                    .find(|child| child.name.eq_ignore_ascii_case(segment))
            })
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "is_assigned references unknown field '{}'",
                    segments.join(".")
                ))
            })?;
    }
    Ok(field.id)
}

/// Snapshot bindings shared by every assignment expression in one scanner.
pub struct FieldAssignmentExprBindings {
    functions: HashMap<i32, Arc<ScalarUDF>>,
}

impl FieldAssignmentExprBindings {
    pub(crate) async fn try_new(
        dataset: &Dataset,
        expressions: &[&Expr],
        selected_fragments: Option<&[lance_table::format::Fragment]>,
    ) -> Result<Option<Self>> {
        let mut field_ids = HashSet::new();
        let mut captured_error = None;
        for expression in expressions {
            expression
                .apply(|node| {
                    match is_assignment_call(node) {
                        Ok(Some(segments)) => match resolve_assignment_field(dataset, &segments) {
                            Ok(field_id) => {
                                field_ids.insert(field_id);
                            }
                            Err(error) => {
                                captured_error = Some(error);
                                return Ok(TreeNodeRecursion::Stop);
                            }
                        },
                        Ok(None) => {}
                        Err(error) => {
                            captured_error = Some(error);
                            return Ok(TreeNodeRecursion::Stop);
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
                .map_err(Error::from)?;
            if captured_error.is_some() {
                break;
            }
        }
        if let Some(error) = captured_error {
            return Err(error);
        }
        if field_ids.is_empty() {
            return Ok(None);
        }

        let selected_fragment_ids = selected_fragments.map(|fragments| {
            fragments
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect::<HashSet<_>>()
        });
        let mut functions = HashMap::with_capacity(field_ids.len());
        for field_id in field_ids {
            if dataset.field_assignment_state(field_id).is_none() {
                let name = dataset
                    .schema()
                    .field_by_id(field_id)
                    .map(|field| field.name.as_str())
                    .unwrap_or("<unknown>");
                return Err(Error::invalid_input(format!(
                    "is_assigned references untracked field '{}' (stable field ID {})",
                    name, field_id
                )));
            }
            let root = dataset
                .load_field_assignment_root(field_id)
                .await?
                .expect("descriptor presence was checked");
            let mut fragments = HashMap::with_capacity(root.fragments.len());
            for fragment in root.fragments {
                let fragment_id = u32::try_from(fragment.fragment_id).map_err(|_| {
                    Error::invalid_input(format!(
                        "Field assignment fragment ID {} does not fit in a row address",
                        fragment.fragment_id
                    ))
                })?;
                if selected_fragment_ids
                    .as_ref()
                    .is_some_and(|selected| !selected.contains(&fragment_id))
                {
                    continue;
                }
                let state = match fragment.state {
                    FieldAssignmentFragmentState::All => AssignmentFragment::All,
                    FieldAssignmentFragmentState::Partial(_) => {
                        let bitmap = dataset
                            .load_field_assignment_bitmap(field_id, &fragment)
                            .await?;
                        AssignmentFragment::Partial(Arc::new(bitmap))
                    }
                };
                fragments.insert(fragment_id, state);
            }
            functions.insert(
                field_id,
                Arc::new(bound_is_assigned_udf(field_id, fragments)),
            );
        }
        Ok(Some(Self { functions }))
    }

    pub(crate) fn bind(&self, expression: Expr, dataset: &Dataset) -> Result<Expr> {
        let mut captured_error = None;
        let transformed = expression
            .transform(|node| {
                let Some(segments) = (match is_assignment_call(&node) {
                    Ok(call) => call,
                    Err(error) => {
                        captured_error = Some(error);
                        return Ok(Transformed::no(node));
                    }
                }) else {
                    return Ok(Transformed::no(node));
                };
                let field_id = match resolve_assignment_field(dataset, &segments) {
                    Ok(field_id) => field_id,
                    Err(error) => {
                        captured_error = Some(error);
                        return Ok(Transformed::no(node));
                    }
                };
                let Some(function) = self.functions.get(&field_id) else {
                    captured_error = Some(Error::internal(format!(
                        "Missing is_assigned binding for field ID {}",
                        field_id
                    )));
                    return Ok(Transformed::no(node));
                };
                Ok(Transformed::yes(Expr::ScalarFunction(
                    ScalarFunction::new_udf(function.clone(), vec![col(lance_core::ROW_ADDR)]),
                )))
            })
            .map_err(Error::from)?
            .data;
        if let Some(error) = captured_error {
            Err(error)
        } else {
            Ok(transformed)
        }
    }
}

fn fragment_path(fragment: &lance_table::format::Fragment) -> Result<&str> {
    fragment
        .files
        .first()
        .map(|file| file.path.as_str())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Fragment {} has no data file to identify assignment state",
                fragment.id
            ))
        })
}

fn fragment_supplies_field(fragment: &lance_table::format::Fragment, field_id: i32) -> bool {
    fragment
        .files
        .iter()
        .any(|file| file.fields.contains(&field_id))
}

fn fragment_physical_rows(fragment: &lance_table::format::Fragment) -> Result<u64> {
    fragment
        .physical_rows
        .map(|rows| rows as u64)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "Fragment {} has no physical row count for field assignment",
                fragment.id
            ))
        })
}

async fn materialize_fragment_bitmap(
    current: &Dataset,
    field_id: i32,
    fragment_id: u64,
    entry: Option<&FieldAssignmentFragment>,
) -> Result<RoaringBitmap> {
    let Some(entry) = entry else {
        return Ok(RoaringBitmap::new());
    };
    match &entry.state {
        FieldAssignmentFragmentState::All => {
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert_range(
                0..u32::try_from(entry.physical_rows).map_err(|_| {
                    Error::invalid_input(format!(
                        "Fragment {} has too many rows for field assignment",
                        fragment_id
                    ))
                })?,
            );
            Ok(bitmap)
        }
        FieldAssignmentFragmentState::Partial(_) => {
            current.load_field_assignment_bitmap(field_id, entry).await
        }
    }
}

#[derive(Default)]
struct PendingRowChanges {
    assigned: RoaringBitmap,
    unassigned: RoaringBitmap,
}

fn add_row_change(
    pending: &mut HashMap<(i32, u64), PendingRowChanges>,
    field_id: i32,
    fragment_id: u64,
    offsets: &RoaringBitmap,
    assigned: bool,
) {
    let change = pending.entry((field_id, fragment_id)).or_default();
    if assigned {
        change.assigned |= offsets;
        change.unassigned -= offsets;
    } else {
        change.unassigned |= offsets;
        change.assigned -= offsets;
    }
}

fn validate_fragment_value(
    field_id: i32,
    fragment_id: u64,
    physical_rows: u64,
    value: &FieldAssignmentFragmentValue,
) -> Result<()> {
    if let FieldAssignmentFragmentValue::Partial(bitmap) = value {
        if bitmap.is_empty() || bitmap.len() == physical_rows {
            return Err(Error::invalid_input(format!(
                "Partial assignment state for field ID {}, fragment {} must be non-empty and non-full",
                field_id, fragment_id
            )));
        }
        if bitmap
            .max()
            .is_some_and(|offset| offset as u64 >= physical_rows)
        {
            return Err(Error::invalid_input(format!(
                "Assignment state for field ID {}, fragment {} contains an out-of-bounds row offset",
                field_id, fragment_id
            )));
        }
    }
    Ok(())
}

/// Reconcile one transaction with the current head's assignment roots.
///
/// This runs after conflict resolution and manifest construction on every
/// commit attempt, so compatible concurrent appends and unrelated field writes
/// are merged from the actual head rather than from the stale read snapshot.
pub async fn apply_field_assignment_transaction(
    current: &Dataset,
    write_store: &ObjectStore,
    manifest: &mut lance_table::format::Manifest,
    transaction: &Transaction,
) -> Result<()> {
    if matches!(
        transaction.operation,
        Operation::Restore { .. } | Operation::Clone { .. }
    ) {
        return Ok(());
    }

    let changes = transaction
        .field_assignment_transaction()?
        .unwrap_or_default();
    let mut initializations = HashMap::new();
    for initialization in &changes.initializations {
        if initializations
            .insert(initialization.field_id, initialization.assigned)
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Field ID {} is initialized for assignment tracking more than once",
                initialization.field_id
            )));
        }
        if current
            .manifest
            .field_assignment_states
            .iter()
            .any(|state| state.field_id == initialization.field_id)
        {
            return Err(Error::invalid_input(format!(
                "Field ID {} already has assignment tracking",
                initialization.field_id
            )));
        }
    }

    let final_field_ids: HashSet<i32> = manifest.schema.field_ids().into_iter().collect();
    let mut transfer_sources = HashMap::new();
    for transfer in &changes.transfers {
        if transfer.source_field_id == transfer.target_field_id {
            return Err(Error::invalid_input(format!(
                "Field assignment transfer source and target are both {}",
                transfer.source_field_id
            )));
        }
        if transfer_sources
            .insert(transfer.target_field_id, transfer.source_field_id)
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Field ID {} is the target of more than one assignment transfer",
                transfer.target_field_id
            )));
        }
        if current
            .field_assignment_state(transfer.source_field_id)
            .is_none()
        {
            return Err(Error::invalid_input(format!(
                "Assignment transfer source field ID {} is not tracked",
                transfer.source_field_id
            )));
        }
        if final_field_ids.contains(&transfer.source_field_id)
            || !final_field_ids.contains(&transfer.target_field_id)
        {
            return Err(Error::invalid_input(format!(
                "Assignment transfer must replace removed field ID {} with current field ID {}",
                transfer.source_field_id, transfer.target_field_id
            )));
        }
        if initializations.contains_key(&transfer.target_field_id) {
            return Err(Error::invalid_input(format!(
                "Field ID {} cannot be initialized and receive an assignment transfer",
                transfer.target_field_id
            )));
        }
    }

    for field_id in initializations.keys() {
        if !final_field_ids.contains(field_id) {
            return Err(Error::invalid_input(format!(
                "Cannot initialize assignment tracking for unknown field ID {}",
                field_id
            )));
        }
    }

    let mut tracked_fields: Vec<i32> = current
        .manifest
        .field_assignment_states
        .iter()
        .map(|state| state.field_id)
        .chain(initializations.keys().copied())
        .chain(transfer_sources.keys().copied())
        .filter(|field_id| final_field_ids.contains(field_id))
        .collect();
    tracked_fields.sort_unstable();
    tracked_fields.dedup();
    if tracked_fields.is_empty() {
        manifest.field_assignment_states.clear();
        return Ok(());
    }

    let final_fragments: HashMap<u64, &lance_table::format::Fragment> = manifest
        .fragments
        .iter()
        .map(|fragment| (fragment.id, fragment))
        .collect();
    let current_fragment_ids: HashSet<u64> = current
        .manifest
        .fragments
        .iter()
        .map(|fragment| fragment.id)
        .collect();
    let final_fragment_ids: HashSet<u64> = final_fragments.keys().copied().collect();
    let fragment_ids_unchanged = final_fragment_ids == current_fragment_ids;
    let fragment_by_path: HashMap<&str, &lance_table::format::Fragment> = manifest
        .fragments
        .iter()
        .map(|fragment| Ok((fragment_path(fragment)?, fragment)))
        .collect::<Result<_>>()?;

    let mut overrides: HashMap<(i32, u64), &FieldAssignmentFragmentValue> = HashMap::new();
    for override_state in &changes.fragment_states {
        let fragment = fragment_by_path
            .get(override_state.fragment_path.as_str())
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Field assignment state references unknown new fragment path '{}'",
                    override_state.fragment_path
                ))
            })?;
        if overrides
            .insert(
                (override_state.field_id, fragment.id),
                &override_state.state,
            )
            .is_some()
        {
            return Err(Error::invalid_input(format!(
                "Duplicate field assignment state for field ID {}, fragment path '{}'",
                override_state.field_id, override_state.fragment_path
            )));
        }
    }

    let mut pending_changes: HashMap<(i32, u64), PendingRowChanges> = HashMap::new();
    for change in &changes.row_changes {
        if !tracked_fields.contains(&change.field_id) {
            return Err(Error::invalid_input(format!(
                "Assignment row change references untracked field ID {}",
                change.field_id
            )));
        }
        let mut by_fragment: HashMap<u64, RoaringBitmap> = HashMap::new();
        for raw_address in &change.row_addresses {
            let address = RowAddress::from(raw_address);
            by_fragment
                .entry(address.fragment_id() as u64)
                .or_default()
                .insert(address.row_offset());
        }
        for (fragment_id, offsets) in by_fragment {
            add_row_change(
                &mut pending_changes,
                change.field_id,
                fragment_id,
                &offsets,
                change.assigned,
            );
        }
    }

    if let Operation::Update {
        fields_modified,
        updated_fragments,
        updated_fragment_offsets,
        update_mode: Some(UpdateMode::RewriteColumns),
        ..
    } = &transaction.operation
    {
        for field_id in fields_modified
            .iter()
            .map(|field_id| *field_id as i32)
            .filter(|field_id| tracked_fields.contains(field_id))
        {
            for fragment in updated_fragments {
                let offsets = updated_fragment_offsets
                    .as_ref()
                    .and_then(|all| all.0.get(&fragment.id))
                    .cloned()
                    .unwrap_or_else(|| {
                        let mut all = RoaringBitmap::new();
                        if let Some(rows) = fragment.physical_rows {
                            all.insert_range(0..rows as u32);
                        }
                        all
                    });
                add_row_change(&mut pending_changes, field_id, fragment.id, &offsets, true);
            }
        }
    }

    if let Operation::DataOverlay { groups } = &transaction.operation {
        for group in groups {
            for overlay in &group.overlays {
                for (field_position, field_id) in overlay.data_file.fields.iter().enumerate() {
                    if tracked_fields.contains(field_id) {
                        add_row_change(
                            &mut pending_changes,
                            *field_id,
                            group.fragment_id,
                            overlay.coverage_for_field(field_position)?.as_ref(),
                            true,
                        );
                    }
                }
            }
        }
    }

    if let Operation::DataReplacement { replacements } = &transaction.operation {
        for replacement in replacements {
            let fragment = final_fragments.get(&replacement.0).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Data replacement references fragment {} that is not in the new snapshot",
                    replacement.0
                ))
            })?;
            let physical_rows = u32::try_from(fragment_physical_rows(fragment)?).map_err(|_| {
                Error::invalid_input(format!(
                    "Fragment {} has too many rows for field assignment",
                    replacement.0
                ))
            })?;
            let mut all_rows = RoaringBitmap::new();
            all_rows.insert_range(0..physical_rows);
            for field_id in replacement
                .1
                .fields
                .iter()
                .copied()
                .filter(|field_id| tracked_fields.contains(field_id))
            {
                add_row_change(
                    &mut pending_changes,
                    field_id,
                    replacement.0,
                    &all_rows,
                    true,
                );
            }
        }
    }

    let mut descriptors = Vec::with_capacity(tracked_fields.len());
    for field_id in tracked_fields {
        let source_field_id = transfer_sources.get(&field_id).copied().unwrap_or(field_id);
        let has_explicit_change = initializations.contains_key(&field_id)
            || overrides
                .keys()
                .any(|(override_field_id, _)| *override_field_id == field_id)
            || pending_changes
                .keys()
                .any(|(change_field_id, _)| *change_field_id == field_id);
        if fragment_ids_unchanged
            && !has_explicit_change
            && source_field_id != field_id
            && let Some(descriptor) = current.field_assignment_state(source_field_id)
        {
            descriptors.push(FieldAssignmentState {
                field_id,
                root: descriptor.root.clone(),
            });
            continue;
        }
        if fragment_ids_unchanged
            && !has_explicit_change
            && let Some(descriptor) = current.field_assignment_state(field_id)
        {
            descriptors.push(descriptor.clone());
            continue;
        }

        let mut states: BTreeMap<u64, FieldAssignmentFragment> =
            if let Some(&assigned) = initializations.get(&field_id) {
                if assigned {
                    final_fragments
                        .values()
                        .map(|fragment| {
                            Ok((
                                fragment.id,
                                FieldAssignmentFragment {
                                    fragment_id: fragment.id,
                                    physical_rows: fragment_physical_rows(fragment)?,
                                    state: FieldAssignmentFragmentState::All,
                                },
                            ))
                        })
                        .collect::<Result<_>>()?
                } else {
                    BTreeMap::new()
                }
            } else {
                current
                    .load_field_assignment_root(source_field_id)
                    .await?
                    .map(|root| {
                        root.fragments
                            .into_iter()
                            .filter(|entry| final_fragments.contains_key(&entry.fragment_id))
                            .map(|entry| (entry.fragment_id, entry))
                            .collect()
                    })
                    .unwrap_or_default()
            };

        match &transaction.operation {
            Operation::Append { .. } => {
                for fragment in manifest
                    .fragments
                    .iter()
                    .filter(|fragment| !current_fragment_ids.contains(&fragment.id))
                {
                    if fragment_supplies_field(fragment, field_id) {
                        states.insert(
                            fragment.id,
                            FieldAssignmentFragment {
                                fragment_id: fragment.id,
                                physical_rows: fragment_physical_rows(fragment)?,
                                state: FieldAssignmentFragmentState::All,
                            },
                        );
                    }
                }
            }
            Operation::Overwrite { .. } => {
                states.clear();
                for fragment in manifest.fragments.iter() {
                    if fragment_supplies_field(fragment, field_id) {
                        states.insert(
                            fragment.id,
                            FieldAssignmentFragment {
                                fragment_id: fragment.id,
                                physical_rows: fragment_physical_rows(fragment)?,
                                state: FieldAssignmentFragmentState::All,
                            },
                        );
                    }
                }
            }
            Operation::Rewrite { groups, .. } => {
                for group in groups {
                    for old in &group.old_fragments {
                        states.remove(&old.id);
                    }
                    for new_fragment in &group.new_fragments {
                        let final_fragment = fragment_by_path
                            .get(fragment_path(new_fragment)?)
                            .ok_or_else(|| {
                                Error::internal("Rewritten fragment is missing from final manifest")
                            })?;
                        if !overrides.contains_key(&(field_id, final_fragment.id)) {
                            return Err(Error::invalid_input(format!(
                                "Rewrite must supply assignment state for tracked field ID {}, new fragment '{}'",
                                field_id,
                                fragment_path(new_fragment)?
                            )));
                        }
                    }
                }
            }
            Operation::Update {
                new_fragments,
                update_mode: Some(UpdateMode::RewriteRows),
                ..
            } => {
                for new_fragment in new_fragments {
                    let final_fragment = fragment_by_path
                        .get(fragment_path(new_fragment)?)
                        .ok_or_else(|| {
                            Error::internal("Updated fragment is missing from final manifest")
                        })?;
                    if !overrides.contains_key(&(field_id, final_fragment.id)) {
                        return Err(Error::invalid_input(format!(
                            "Row rewrite must supply assignment state for tracked field ID {}, new fragment '{}'",
                            field_id,
                            fragment_path(new_fragment)?
                        )));
                    }
                }
            }
            _ => {}
        }

        for ((override_field_id, fragment_id), value) in &overrides {
            if *override_field_id != field_id {
                continue;
            }
            let fragment = final_fragments.get(fragment_id).ok_or_else(|| {
                Error::internal(format!("Final fragment {} is missing", fragment_id))
            })?;
            let physical_rows = fragment_physical_rows(fragment)?;
            validate_fragment_value(field_id, *fragment_id, physical_rows, value)?;
            match value {
                FieldAssignmentFragmentValue::None => {
                    states.remove(fragment_id);
                }
                FieldAssignmentFragmentValue::All => {
                    states.insert(
                        *fragment_id,
                        FieldAssignmentFragment {
                            fragment_id: *fragment_id,
                            physical_rows,
                            state: FieldAssignmentFragmentState::All,
                        },
                    );
                }
                FieldAssignmentFragmentValue::Partial(bitmap) => {
                    let file = current
                        .write_field_assignment_bitmap(
                            write_store,
                            field_id,
                            *fragment_id,
                            physical_rows,
                            bitmap,
                        )
                        .await?;
                    states.insert(
                        *fragment_id,
                        FieldAssignmentFragment {
                            fragment_id: *fragment_id,
                            physical_rows,
                            state: FieldAssignmentFragmentState::Partial(file),
                        },
                    );
                }
            }
        }

        for ((change_field_id, fragment_id), change) in &pending_changes {
            if *change_field_id != field_id {
                continue;
            }
            let fragment = final_fragments.get(fragment_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Assignment change for field ID {} references fragment {} that is not in the new snapshot",
                    field_id, fragment_id
                ))
            })?;
            let physical_rows = fragment_physical_rows(fragment)?;
            if change
                .assigned
                .max()
                .into_iter()
                .chain(change.unassigned.max())
                .any(|offset| offset as u64 >= physical_rows)
            {
                return Err(Error::invalid_input(format!(
                    "Assignment change for field ID {}, fragment {} contains an out-of-bounds row offset",
                    field_id, fragment_id
                )));
            }
            let mut bitmap = materialize_fragment_bitmap(
                current,
                source_field_id,
                *fragment_id,
                states.get(fragment_id),
            )
            .await?;
            bitmap |= &change.assigned;
            bitmap -= &change.unassigned;
            if bitmap.is_empty() {
                states.remove(fragment_id);
            } else if bitmap.len() == physical_rows {
                states.insert(
                    *fragment_id,
                    FieldAssignmentFragment {
                        fragment_id: *fragment_id,
                        physical_rows,
                        state: FieldAssignmentFragmentState::All,
                    },
                );
            } else {
                let file = current
                    .write_field_assignment_bitmap(
                        write_store,
                        field_id,
                        *fragment_id,
                        physical_rows,
                        &bitmap,
                    )
                    .await?;
                states.insert(
                    *fragment_id,
                    FieldAssignmentFragment {
                        fragment_id: *fragment_id,
                        physical_rows,
                        state: FieldAssignmentFragmentState::Partial(file),
                    },
                );
            }
        }

        descriptors.push(
            current
                .write_field_assignment_root(
                    write_store,
                    field_id,
                    FieldAssignmentRoot {
                        fragments: states.into_values().collect(),
                    },
                )
                .await?,
        );
    }
    descriptors.sort_by_key(|state| state.field_id);
    manifest.field_assignment_states = descriptors;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, BooleanArray, Int32Array, Int64Array, RecordBatch, RecordBatchIterator,
        StructArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use datafusion::functions_aggregate::count::count;
    use datafusion::logical_expr::{col, lit};
    use lance_core::{ROW_ID, utils::tempfile::TempStrDir};
    use lance_datafusion::udf::is_assigned;
    use lance_file::version::LanceFileVersion;
    use lance_file::writer::FileWriterOptions;
    use lance_io::utils::CachedFileSize;
    use lance_table::feature_flags::{FLAG_FIELD_ASSIGNMENT, FLAG_UNKNOWN};
    use lance_table::format::DataFile;
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};

    use super::*;
    use crate::dataset::cleanup::{CleanupPolicyBuilder, cleanup_old_versions};
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::scanner::{AggregateExpr, ColumnOrdering};
    use crate::dataset::transaction::{DataOverlayGroup, Operation};
    use crate::dataset::write::merge_insert::{MergeInsertBuilder, WhenMatched, WhenNotMatched};
    use crate::dataset::{
        ColumnAlteration, FieldAssignment, FieldAssignmentAlteration, NewColumnTransform,
        UpdateBuilder, WriteDestination, WriteMode, WriteParams,
    };

    async fn tracked_dataset(directory: &TempStrDir, rows_per_file: usize) -> Result<Dataset> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..8))],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            directory,
            Some(WriteParams {
                max_rows_per_file: rows_per_file,
                ..Default::default()
            }),
        )
        .await?;
        dataset
            .add_columns_with_assignment(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "embedding",
                    DataType::Int32,
                    true,
                )]))),
                None,
                None,
                Some(FieldAssignment::Unassigned),
            )
            .await?;
        Ok(dataset)
    }

    async fn assignment_rows(dataset: &Dataset, field: &str) -> Result<Vec<(i32, bool)>> {
        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[
            ("id", "id"),
            ("assigned", &format!("is_assigned({field})")),
        ])?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let mut rows = (0..batch.num_rows())
            .map(|index| (ids.value(index), assigned.value(index)))
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.0);
        Ok(rows)
    }

    async fn commit_assignment_overlay(
        dataset: Dataset,
        fragment_id: u64,
        field_id: i32,
        offsets: RoaringBitmap,
        values: ArrayRef,
    ) -> Result<Dataset> {
        let read_version = dataset.version().version;
        let overlay_schema = dataset.schema().project_by_ids(&[field_id], true);
        let filename = format!("assignment-overlay-{}.lance", Uuid::new_v4());
        let path = dataset.data_dir().join(filename.as_str());
        let object_writer = dataset.object_store.create(&path).await?;
        let file_version = LanceFileVersion::Stable.resolve();
        let mut writer = lance_file::versions::create_writer(
            file_version,
            object_writer,
            overlay_schema,
            FileWriterOptions::default(),
        )?;
        writer.write_column(0, values).await?;
        let summary = writer.finish().await?;
        let mut data_file = DataFile::new_unstarted(filename, file_version);
        data_file.fields = writer
            .field_id_to_column_indices()
            .iter()
            .map(|(field_id, _)| *field_id as i32)
            .collect::<Vec<_>>()
            .into();
        data_file.column_indices = writer
            .field_id_to_column_indices()
            .iter()
            .map(|(_, column_index)| *column_index as i32)
            .collect::<Vec<_>>()
            .into();
        data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

        Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataOverlay {
                groups: vec![DataOverlayGroup {
                    fragment_id,
                    overlays: vec![DataOverlayFile {
                        data_file,
                        coverage: OverlayCoverage::dense(offsets),
                        committed_version: 0,
                    }],
                }],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
    }

    #[tokio::test]
    async fn test_initial_assignment_expression_is_non_null_and_composable() -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..6))],
        )?;
        let directory = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await?;

        let assigned_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "embedding",
            DataType::Int32,
            true,
        )]));
        dataset
            .add_columns_with_assignment(
                NewColumnTransform::AllNulls(assigned_schema),
                None,
                None,
                Some(FieldAssignment::Unassigned),
            )
            .await?;

        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[
            ("assigned", "is_assigned(embedding)"),
            (
                "combined",
                "NOT (is_assigned(embedding) OR id < 0) AND (id >= 0 OR is_assigned(embedding))",
            ),
        ])?;
        let batch = scanner.try_into_batch().await?;
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let combined = batch
            .column_by_name("combined")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(assigned.null_count(), 0);
        assert_eq!(assigned.iter().collect::<Vec<_>>(), vec![Some(false); 6]);
        assert_eq!(combined.iter().collect::<Vec<_>>(), vec![Some(true); 6]);

        let mut scanner = dataset.scan();
        scanner.filter("NOT is_assigned(embedding)")?;
        assert_eq!(scanner.count_rows().await?, 6);

        Ok(())
    }

    #[tokio::test]
    async fn test_nested_field_assignment_and_parent_write_conflict() -> Result<()> {
        let value_field = Arc::new(ArrowField::new("value", DataType::Int32, true));
        let payload = StructArray::from(vec![(
            value_field.clone(),
            Arc::new(Int32Array::from(vec![Some(10), None])) as ArrayRef,
        )]);
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("payload", payload.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..2)),
                Arc::new(payload),
            ],
        )?;
        let directory = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            None,
        )
        .await?;
        dataset
            .alter_columns_with_assignment(
                &[],
                &[FieldAssignmentAlteration::new(
                    "payload.value",
                    FieldAssignment::Unassigned,
                )],
            )
            .await?;

        let appended_payload = StructArray::from(vec![(
            value_field.clone(),
            Arc::new(Int32Array::from(vec![None])) as ArrayRef,
        )]);
        let appended_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("payload", appended_payload.data_type().clone(), false),
        ]));
        let appended = RecordBatch::try_new(
            appended_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2])),
                Arc::new(appended_payload),
            ],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(appended)], appended_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?;
        assert_eq!(
            assignment_rows(&dataset, "payload.value").await?,
            vec![(0, false), (1, false), (2, true)]
        );

        let error = UpdateBuilder::new(Arc::new(dataset))
            .set("payload", "payload")?
            .invalidate_fields(["payload.value"])?
            .build()
            .unwrap_err();
        assert!(error.to_string().contains("both written and invalidated"));

        Ok(())
    }

    #[tokio::test]
    async fn test_append_distinguishes_explicit_null_from_omitted_field() -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..2))],
        )?;
        let directory = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            None,
        )
        .await?;
        dataset
            .add_columns_with_assignment(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "embedding",
                    DataType::Int32,
                    true,
                )]))),
                None,
                None,
                Some(FieldAssignment::Unassigned),
            )
            .await?;

        let explicit_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let explicit_nulls = RecordBatch::try_new(
            explicit_schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(2..4)),
                Arc::new(Int32Array::from(vec![None, None])),
            ],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(explicit_nulls)], explicit_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?;

        let omitted_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let omitted = RecordBatch::try_new(
            omitted_schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(4..6))],
        )?;
        dataset = Dataset::write(
            RecordBatchIterator::new([Ok(omitted)], omitted_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await?;

        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[
            ("id", "id"),
            ("embedding", "embedding"),
            ("assigned", "is_assigned(embedding)"),
        ])?;
        let batch = scanner.try_into_batch().await?;
        let embedding = batch
            .column_by_name("embedding")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(embedding.null_count(), 6);
        assert_eq!(
            assigned.iter().collect::<Vec<_>>(),
            vec![
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_update_invalidation_and_rename_preserve_stored_values() -> Result<()> {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("text", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(Int32Array::from_iter_values(10..14)),
            ],
        )?;
        let directory = TempStrDir::default();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await?;
        dataset
            .add_columns_with_assignment(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                    "embedding",
                    DataType::Int32,
                    true,
                )]))),
                None,
                None,
                Some(FieldAssignment::Unassigned),
            )
            .await?;

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id < 2")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")?
            .set("text", "99")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();

        let tracked_field_id = dataset.schema().field("embedding").unwrap().id;
        dataset
            .alter_columns(&[
                ColumnAlteration::new("embedding".to_string()).rename("vector".to_string())
            ])
            .await?;
        assert_eq!(
            dataset.schema().field("vector").unwrap().id,
            tracked_field_id
        );

        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[
            ("id", "id"),
            ("vector", "vector"),
            ("assigned", "is_assigned(vector)"),
        ])?;
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let vectors = batch
            .column_by_name("vector")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let mut rows = (0..batch.num_rows())
            .map(|index| {
                (
                    ids.value(index),
                    (!vectors.is_null(index)).then(|| vectors.value(index)),
                    assigned.value(index),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.0);
        assert_eq!(
            rows,
            vec![
                (0, Some(0), true),
                (1, Some(10), false),
                (2, None, false),
                (3, None, false),
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_invalidation_only_update_is_metadata_only() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id < 2")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset;
        let field_id = dataset.schema().field("embedding").unwrap().id;
        let fragments_before = dataset.manifest.fragments.clone();
        let root_before = dataset
            .field_assignment_state(field_id)
            .unwrap()
            .root
            .clone();

        let result = UpdateBuilder::new(dataset)
            .update_where("id = 0")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        assert_eq!(result.rows_updated, 1);
        assert_eq!(result.new_dataset.manifest.fragments, fragments_before);
        assert_ne!(
            result
                .new_dataset
                .field_assignment_state(field_id)
                .unwrap()
                .root,
            root_before
        );
        assert_eq!(
            assignment_rows(&result.new_dataset, "embedding").await?,
            vec![
                (0, false),
                (1, true),
                (2, false),
                (3, false),
                (4, false),
                (5, false),
                (6, false),
                (7, false),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_composes_in_ordering_and_aggregation() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 3, 5, 7)")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        let dataset = result.new_dataset;

        let mut invalid_ordering = dataset.scan();
        assert!(
            invalid_ordering
                .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
                    "id + 1".to_string(),
                )]))
                .is_err()
        );

        let mut scanner = dataset.scan();
        scanner.project_with_transform(&[("id", "id"), ("assigned", "is_assigned(embedding)")])?;
        scanner.order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "is_assigned(embedding)".to_string(),
        )]))?;
        let batch = scanner.try_into_batch().await?;
        let assigned = batch
            .column_by_name("assigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(
            assigned.iter().collect::<Vec<_>>(),
            vec![Some(false); 4]
                .into_iter()
                .chain(vec![Some(true); 4])
                .collect::<Vec<_>>()
        );

        let mut scanner = dataset.scan();
        scanner.aggregate(AggregateExpr::datafusion(
            vec![is_assigned(col("embedding"))],
            vec![count(lit(1)).alias("rows")],
        ))?;
        let batch = scanner.try_into_batch().await?;
        let assigned = batch
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        let counts = batch
            .column_by_name("rows")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let mut groups = (0..batch.num_rows())
            .map(|index| (assigned.value(index), counts.value(index)))
            .collect::<Vec<_>>();
        groups.sort_by_key(|group| group.0);
        assert_eq!(groups, vec![(false, 4), (true, 4)]);

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_and_compaction_remap_assignment() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 2, 5, 6)")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        dataset.delete("id IN (2, 4)").await?;
        let before = assignment_rows(&dataset, "embedding").await?;

        compact_files(&mut dataset, CompactionOptions::default(), None).await?;

        let after = assignment_rows(&dataset, "embedding").await?;
        assert_eq!(after, before);
        assert_eq!(
            after,
            vec![
                (0, false),
                (1, true),
                (3, false),
                (5, true),
                (6, true),
                (7, false),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_time_travel_restore_and_cast_transfer_assignment() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let unassigned_version = dataset.version().version;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id < 3")?
            .set("embedding", "id * 10")?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        assert_eq!(
            assignment_rows(&dataset, "embedding").await?,
            vec![
                (0, true),
                (1, true),
                (2, true),
                (3, false),
                (4, false),
                (5, false),
                (6, false),
                (7, false),
            ]
        );

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 7")?
            .set("embedding", "70")?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();
        let old_field_id = dataset.schema().field("embedding").unwrap().id;
        let old_root = dataset
            .field_assignment_state(old_field_id)
            .unwrap()
            .root
            .clone();
        let expected = assignment_rows(&dataset, "embedding").await?;
        dataset
            .alter_columns(&[
                ColumnAlteration::new("embedding".to_string()).cast_to(DataType::Int64)
            ])
            .await?;
        let new_field_id = dataset.schema().field("embedding").unwrap().id;
        assert_ne!(new_field_id, old_field_id);
        assert_eq!(
            dataset.field_assignment_state(new_field_id).unwrap().root,
            old_root
        );
        assert_eq!(assignment_rows(&dataset, "embedding").await?, expected);

        let mut historical = dataset.checkout_version(unassigned_version).await?;
        assert!(
            assignment_rows(&historical, "embedding")
                .await?
                .iter()
                .all(|(_, assigned)| !assigned)
        );
        historical.restore().await?;
        assert!(
            assignment_rows(&historical, "embedding")
                .await?
                .iter()
                .all(|(_, assigned)| !assigned)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_state_is_loaded_only_when_referenced() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let field_id = dataset.schema().field("embedding").unwrap().id;
        let root = dataset
            .field_assignment_state(field_id)
            .unwrap()
            .root
            .clone();
        let store = dataset.object_store(root.base_id).await?;
        store.delete(&dataset.field_assignment_path(&root)?).await?;

        let mut ordinary = dataset.scan();
        ordinary.project(&["id"])?;
        assert_eq!(ordinary.try_into_batch().await?.num_rows(), 8);

        let mut assignment = dataset.scan();
        assignment.project_with_transform(&[("assigned", "is_assigned(embedding)")])?;
        let error = assignment.try_into_batch().await.unwrap_err();
        assert!(
            error.to_string().contains(&root.path),
            "unexpected assignment read error: {error}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_invalid_assignment_expressions_fail_during_planning() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;

        for expression in ["is_assigned(missing)", "is_assigned(id)", "is_assigned(42)"] {
            let mut scanner = dataset.scan();
            let error = match scanner.project_with_transform(&[("assigned", expression)]) {
                Ok(_) => scanner.try_into_batch().await.unwrap_err(),
                Err(error) => error,
            };
            assert!(
                matches!(error, Error::InvalidInput { .. }),
                "unexpected error for {expression}: {error}"
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_appends_reconcile_assignment_from_current_head() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = Arc::new(tracked_dataset(&directory, 2).await?);

        let explicit_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let explicit = RecordBatch::try_new(
            explicit_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![8])),
                Arc::new(Int32Array::from(vec![None])),
            ],
        )?;
        let omitted_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let omitted = RecordBatch::try_new(
            omitted_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![9]))],
        )?;

        let explicit_write = Dataset::write(
            RecordBatchIterator::new([Ok(explicit)], explicit_schema),
            dataset.clone(),
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        );
        let omitted_write = Dataset::write(
            RecordBatchIterator::new([Ok(omitted)], omitted_schema),
            dataset,
            Some(WriteParams {
                mode: WriteMode::Append,
                ..Default::default()
            }),
        );
        let (explicit_result, omitted_result) = tokio::join!(explicit_write, omitted_write);
        explicit_result?;
        omitted_result?;

        let dataset = Dataset::open(&directory).await?;
        let rows = assignment_rows(&dataset, "embedding").await?;
        assert_eq!(rows.len(), 10);
        assert_eq!(rows[8], (8, true));
        assert_eq!(rows[9], (9, false));
        Ok(())
    }

    #[tokio::test]
    async fn test_update_only_merge_insert_on_row_id_assigns_only_matched_rows() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let mut scanner = dataset.scan();
        scanner.with_row_id();
        let batch = scanner.try_into_batch().await?;
        let ids = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let row_ids = batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let selected = (0..batch.num_rows())
            .filter(|index| matches!(ids.value(*index), 1 | 6))
            .map(|index| row_ids.value(index))
            .collect::<Vec<_>>();

        let source_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ID, DataType::UInt64, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let source = RecordBatch::try_new(
            source_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(selected.clone())),
                Arc::new(Int32Array::from(vec![None, Some(60)])),
            ],
        )?;
        let mut builder = MergeInsertBuilder::try_new(Arc::new(dataset), vec![ROW_ID.to_string()])?;
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing);
        let (dataset, stats) = builder
            .try_build()?
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(source)],
                source_schema,
            )))
            .await?;
        assert_eq!(stats.num_updated_rows, 2);
        assert_eq!(stats.num_inserted_rows, 0);
        assert_eq!(
            assignment_rows(&dataset, "embedding").await?,
            vec![
                (0, false),
                (1, true),
                (2, false),
                (3, false),
                (4, false),
                (5, false),
                (6, true),
                (7, false),
            ]
        );

        let invalidation_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ID, DataType::UInt64, false),
            ArrowField::new("id", DataType::Int32, false),
        ]));
        let invalidation = RecordBatch::try_new(
            invalidation_schema.clone(),
            vec![
                Arc::new(UInt64Array::from(selected)),
                Arc::new(Int32Array::from(vec![1, 6])),
            ],
        )?;
        let mut builder = MergeInsertBuilder::try_new(dataset, vec![ROW_ID.to_string()])?;
        builder
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .invalidate_fields(["embedding"])?;
        let (dataset, stats) = builder
            .try_build()?
            .execute_reader(Box::new(RecordBatchIterator::new(
                [Ok(invalidation)],
                invalidation_schema,
            )))
            .await?;
        assert_eq!(stats.num_updated_rows, 2);
        assert!(
            assignment_rows(&dataset, "embedding")
                .await?
                .iter()
                .all(|(_, assigned)| !assigned)
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_data_replacement_creates_assignment() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 8).await?;
        let replacement_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "embedding",
            DataType::Int32,
            true,
        )]));
        let replacement = RecordBatch::try_new(
            replacement_schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(100..108))],
        )?;
        let path = "assignment-replacement.lance";
        let writer = dataset
            .object_store
            .create(&dataset.data_dir().join(path))
            .await?;
        let mut writer = lance_file::versions::v2_1::create_writer(
            writer,
            replacement_schema.as_ref().try_into()?,
            Default::default(),
        )?;
        writer.write_batch(&replacement).await?;
        writer.finish().await?;
        let data_file = dataset.create_data_file(path, None).await?;
        let read_version = dataset.version().version;

        let mut dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataReplacement {
                replacements: vec![crate::dataset::transaction::DataReplacementGroup(
                    0, data_file,
                )],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await?;

        let expected = assignment_rows(&dataset, "embedding").await?;
        assert!(expected.iter().all(|(_, assigned)| *assigned));

        let merge_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("label", DataType::Int32, false),
        ]));
        let merge_batch = RecordBatch::try_new(
            merge_schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from_iter_values(200..208)),
            ],
        )?;
        dataset
            .merge(
                RecordBatchIterator::new([Ok(merge_batch)], merge_schema),
                "id",
                "id",
            )
            .await?;
        assert_eq!(assignment_rows(&dataset, "embedding").await?, expected);
        Ok(())
    }

    #[tokio::test]
    async fn test_overwrite_and_drop_assignment_tracking() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 2).await?;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![20, 21, 22])),
                Arc::new(Int32Array::from(vec![None, Some(210), None])),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await?;
        assert_eq!(
            assignment_rows(&dataset, "embedding").await?,
            vec![(20, true), (21, true), (22, true)]
        );

        dataset.drop_columns(&["embedding"]).await?;
        assert!(dataset.manifest.field_assignment_states.is_empty());
        let mut scanner = dataset.scan();
        let error = match scanner.project_with_transform(&[("assigned", "is_assigned(embedding)")])
        {
            Ok(_) => scanner.try_into_batch().await.unwrap_err(),
            Err(error) => error,
        };
        assert!(matches!(error, Error::InvalidInput { .. }));

        Ok(())
    }

    #[tokio::test]
    async fn test_data_overlay_coverage_creates_assignment() -> Result<()> {
        let directory = TempStrDir::default();
        let dataset = tracked_dataset(&directory, 8).await?;
        let field_id = dataset.schema().field("embedding").unwrap().id;
        let dataset = commit_assignment_overlay(
            dataset,
            0,
            field_id,
            RoaringBitmap::from_iter([1, 4]),
            Arc::new(Int32Array::from(vec![None, Some(40)])),
        )
        .await?;

        assert_eq!(
            assignment_rows(&dataset, "embedding").await?,
            vec![
                (0, false),
                (1, true),
                (2, false),
                (3, false),
                (4, true),
                (5, false),
                (6, false),
                (7, false),
            ]
        );
        let batch = dataset.scan().try_into_batch().await?;
        let embedding = batch
            .column_by_name("embedding")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert!(embedding.is_null(1));
        assert_eq!(embedding.value(4), 40);

        Ok(())
    }

    #[tokio::test]
    async fn test_assignment_clone_storage_and_writer_compatibility() -> Result<()> {
        let source_directory = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from_iter_values(10..18)),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &source_directory,
            None,
        )
        .await?;
        dataset
            .alter_columns_with_assignment(
                &[],
                &[FieldAssignmentAlteration::new(
                    "embedding",
                    FieldAssignment::Assigned,
                )],
            )
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id IN (1, 6)")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        let mut dataset = result.new_dataset.as_ref().clone();
        let expected = assignment_rows(&dataset, "embedding").await?;
        assert_eq!(expected.iter().filter(|(_, assigned)| !assigned).count(), 2);
        assert_eq!(
            dataset.manifest.reader_feature_flags & FLAG_FIELD_ASSIGNMENT,
            0
        );
        assert_ne!(
            dataset.manifest.writer_feature_flags & FLAG_FIELD_ASSIGNMENT,
            0
        );

        let mut unsupported_writer = dataset.clone();
        Arc::make_mut(&mut unsupported_writer.manifest).writer_feature_flags |= FLAG_UNKNOWN;
        let error = unsupported_writer
            .update_config([("unsupported", "writer")])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cannot be written"));

        let shallow_directory = TempStrDir::default();
        let shallow = dataset
            .shallow_clone(shallow_directory.as_ref(), dataset.version().version, None)
            .await?;
        let field_id = shallow.schema().field("embedding").unwrap().id;
        assert!(
            shallow
                .field_assignment_state(field_id)
                .unwrap()
                .root
                .base_id
                .is_some()
        );
        assert_eq!(assignment_rows(&shallow, "embedding").await?, expected);

        let deep_directory = TempStrDir::default();
        let deep = dataset
            .deep_clone(deep_directory.as_ref(), dataset.version().version, None)
            .await?;
        let deep_descriptor = deep.field_assignment_state(field_id).unwrap();
        assert_eq!(deep_descriptor.root.base_id, None);
        let deep_root = deep.load_field_assignment_root(field_id).await?.unwrap();
        assert!(
            deep_root
                .fragments
                .iter()
                .any(|fragment| matches!(fragment.state, FieldAssignmentFragmentState::Partial(_)))
        );
        assert!(
            deep_root
                .fragments
                .iter()
                .all(|fragment| match &fragment.state {
                    FieldAssignmentFragmentState::All => true,
                    FieldAssignmentFragmentState::Partial(bitmap) => bitmap.base_id.is_none(),
                })
        );

        // Prove the deep clone owns its assignment objects instead of falling
        // back to the source dataset after reopening with a fresh session.
        let source_descriptor = dataset.field_assignment_state(field_id).unwrap().clone();
        let source_root = dataset.load_field_assignment_root(field_id).await?.unwrap();
        for fragment in &source_root.fragments {
            if let FieldAssignmentFragmentState::Partial(bitmap) = &fragment.state {
                let store = dataset.object_store(bitmap.base_id).await?;
                store
                    .delete(&dataset.field_assignment_path(bitmap)?)
                    .await?;
            }
        }
        let store = dataset.object_store(source_descriptor.root.base_id).await?;
        store
            .delete(&dataset.field_assignment_path(&source_descriptor.root)?)
            .await?;
        let reopened = Dataset::open(deep_directory.as_ref()).await?;
        assert_eq!(assignment_rows(&reopened, "embedding").await?, expected);

        Ok(())
    }

    #[tokio::test]
    async fn test_cleanup_retains_branch_assignment_objects() -> Result<()> {
        let directory = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("embedding", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..8)),
                Arc::new(Int32Array::from_iter_values(10..18)),
            ],
        )?;
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            &directory,
            None,
        )
        .await?;
        dataset
            .alter_columns_with_assignment(
                &[],
                &[FieldAssignmentAlteration::new(
                    "embedding",
                    FieldAssignment::Assigned,
                )],
            )
            .await?;
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();
        let field_id = dataset.schema().field("embedding").unwrap().id;
        let branch_descriptor = dataset.field_assignment_state(field_id).unwrap().clone();
        let branch_root = dataset.load_field_assignment_root(field_id).await?.unwrap();
        let branch_paths = std::iter::once(branch_descriptor.root.clone())
            .chain(branch_root.fragments.iter().filter_map(|fragment| {
                if let FieldAssignmentFragmentState::Partial(bitmap) = &fragment.state {
                    Some(bitmap.clone())
                } else {
                    None
                }
            }))
            .collect::<Vec<_>>();
        let expected_branch = assignment_rows(&dataset, "embedding").await?;
        dataset
            .create_branch("assignment-cleanup", dataset.version().version, None)
            .await?;

        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 2")?
            .invalidate_fields(["embedding"])?
            .build()?
            .execute()
            .await?;
        dataset = result.new_dataset.as_ref().clone();
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await?
            .delete_unverified(true)
            .build();
        cleanup_old_versions(&dataset, policy).await?;
        for file in &branch_paths {
            let store = dataset.object_store(file.base_id).await?;
            assert!(store.exists(&dataset.field_assignment_path(file)?).await?);
        }
        let branch = dataset.checkout_branch("assignment-cleanup").await?;
        assert_eq!(
            assignment_rows(&branch, "embedding").await?,
            expected_branch
        );

        dataset.force_delete_branch("assignment-cleanup").await?;
        dataset = Dataset::open(directory.as_ref()).await?;
        assert!(dataset.branches().list().await?.is_empty());
        assert_ne!(
            dataset.field_assignment_state(field_id).unwrap().root.path,
            branch_descriptor.root.path
        );
        let policy = CleanupPolicyBuilder::default()
            .retain_n_versions(&dataset, 1)
            .await?
            .delete_unverified(true)
            .build();
        let explanation = dataset.cleanup(policy.clone()).explain().await?;
        assert!(
            explanation.candidate_files.iter().any(|candidate| candidate
                .path
                .ends_with(branch_descriptor.root.path.as_str())),
            "cleanup did not classify old assignment root '{}' as a candidate: {:?}",
            branch_descriptor.root.path,
            explanation.candidate_files
        );
        cleanup_old_versions(&dataset, policy).await?;
        for file in &branch_paths {
            let store = dataset.object_store(file.base_id).await?;
            assert!(
                !store.exists(&dataset.field_assignment_path(file)?).await?,
                "unreferenced branch assignment object '{}' remains after cleanup",
                file.path
            );
        }

        Ok(())
    }
}
