// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_ipc::{DateUnit, IntervalUnit, Precision, TimeUnit, Type, UnionMode};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::{Descriptor, Error, ErrorCode, ImageRoot, PathKind, Result};

type ExtensionValidator = fn(&Field) -> Result<()>;

/// Worker-supported extension types and their storage/metadata validators.
///
/// The default registry supports no extensions. Registering a name alone is not
/// sufficient: each callback must validate its storage type and extension metadata.
#[derive(Debug, Default, Clone)]
pub struct ExtensionTypes {
    validators: HashMap<String, ExtensionValidator>,
}

impl ExtensionTypes {
    /// Add one supported extension; duplicate or empty names are errors.
    ///
    /// ```
    /// use lance_function::{ExtensionTypes, Error};
    /// use arrow_schema::DataType;
    /// let mut extensions = ExtensionTypes::default();
    /// extensions.register("example.identifier", |field| {
    ///     if field.data_type() == &DataType::FixedSizeBinary(16) { Ok(()) }
    ///     else { Err(Error::new(lance_function::ErrorCode::Incompatible, "expected 16 bytes")) }
    /// })?;
    /// # Ok::<(), Error>(())
    /// ```
    pub fn register(
        &mut self,
        name: impl Into<String>,
        validator: ExtensionValidator,
    ) -> Result<()> {
        let name = name.into();
        if name.is_empty() || self.validators.contains_key(&name) {
            return Err(Error::incompatible(format!(
                "empty or duplicate extension registration {name:?}"
            )));
        }
        self.validators.insert(name, validator);
        Ok(())
    }

    fn validate(&self, field: &Field) -> Result<()> {
        let name = field.metadata().get("ARROW:extension:name");
        if let Some(name) = name {
            let validator = self.validators.get(name).ok_or_else(|| {
                Error::incompatible(format!(
                    "field {:?} uses unknown extension type {name:?}",
                    field.name()
                ))
            })?;
            validator(field)?;
        } else if field.metadata().contains_key("ARROW:extension:metadata") {
            return Err(Error::incompatible(format!(
                "field {:?} has extension metadata without an extension name",
                field.name()
            )));
        }
        self.validate_children(field.data_type())
    }

    fn validate_children(&self, data_type: &DataType) -> Result<()> {
        match data_type {
            DataType::Struct(fields) => {
                for field in fields {
                    self.validate(field)?;
                }
            }
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::ListView(field)
            | DataType::LargeListView(field)
            | DataType::FixedSizeList(field, _)
            | DataType::Map(field, _) => self.validate(field)?,
            DataType::Union(fields, _) => {
                for (_, field) in fields.iter() {
                    self.validate(field)?;
                }
            }
            DataType::RunEndEncoded(run_ends, values) => {
                self.validate(run_ends)?;
                self.validate(values)?;
            }
            DataType::Dictionary(_, values) => self.validate_children(values)?,
            _ => {}
        }
        Ok(())
    }
}

/// Read exactly one encapsulated V5 Arrow IPC Schema message, with no body or EOS.
///
/// Field order, names, types, nullability, and schema/field metadata are preserved.
/// Raw FlatBuffer metadata is checked before conversion to Arrow maps so duplicate
/// keys cannot disappear. Invalid physical type descriptions and unknown extension
/// types fail before any user module can be imported.
pub fn read_schema(bytes: &[u8], extensions: &ExtensionTypes) -> Result<SchemaRef> {
    let (prefix, length_bytes) = if bytes.starts_with(&[0xff; 4]) {
        (8_usize, bytes.get(4..8))
    } else {
        (4_usize, bytes.get(..4))
    };
    let length_bytes: [u8; 4] = length_bytes
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| Error::incompatible("Arrow schema has a truncated IPC prefix"))?;
    let length = i32::from_le_bytes(length_bytes);
    let length = usize::try_from(length)
        .ok()
        .filter(|length| *length > 0)
        .ok_or_else(|| {
            Error::incompatible("Arrow schema requires a positive IPC metadata length")
        })?;
    if prefix.checked_add(length) != Some(bytes.len()) {
        return Err(Error::incompatible(
            "Arrow schema must contain exactly one complete IPC message",
        ));
    }
    let message = arrow_ipc::root_as_message(&bytes[prefix..]).map_err(|error| {
        Error::incompatible(format!("invalid Arrow IPC Schema message: {error}"))
    })?;
    if message.version() != arrow_ipc::MetadataVersion::V5 || message.bodyLength() != 0 {
        return Err(Error::incompatible(
            "Arrow schema requires metadata version V5 and no body",
        ));
    }
    let schema = message
        .header_as_schema()
        .ok_or_else(|| Error::incompatible("Arrow IPC message is not a Schema"))?;
    if !matches!(
        schema.endianness(),
        arrow_ipc::Endianness::Little | arrow_ipc::Endianness::Big
    ) {
        return Err(Error::incompatible("Arrow schema has unknown endianness"));
    }
    if schema.features().is_some_and(|features| {
        features.iter().any(|feature| {
            !matches!(
                feature,
                arrow_ipc::Feature::UNUSED
                    | arrow_ipc::Feature::DICTIONARY_REPLACEMENT
                    | arrow_ipc::Feature::COMPRESSED_BODY
            )
        })
    }) {
        return Err(Error::incompatible(
            "Arrow schema requires an unknown IPC feature",
        ));
    }
    let mut metadata = HashMap::new();
    if let Some(entries) = schema.custom_metadata() {
        for entry in entries {
            let (key, value) = metadata_entry(entry, "schema")?;
            if metadata.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(Error::incompatible(format!(
                    "duplicate Arrow schema metadata key {key:?}"
                )));
            }
        }
    }
    let mut fields = Vec::new();
    if let Some(raw_fields) = schema.fields() {
        fields.reserve(raw_fields.len());
        for raw_field in raw_fields {
            validate_ipc_field(raw_field)?;
            // arrow-ipc's Field conversion is infallible and assumes semantic
            // validity beyond the FlatBuffer verifier. Validate those conditions
            // first, then reuse its complete type/metadata conversion.
            let field = Field::from(raw_field);
            extensions.validate(&field)?;
            fields.push(field);
        }
    }
    Ok(Arc::new(Schema::new_with_metadata(fields, metadata)))
}

fn metadata_entry<'a>(entry: arrow_ipc::KeyValue<'a>, context: &str) -> Result<(&'a str, &'a str)> {
    let key = entry
        .key()
        .ok_or_else(|| Error::incompatible(format!("Arrow {context} metadata entry has no key")))?;
    let value = entry.value().ok_or_else(|| {
        Error::incompatible(format!("Arrow {context} metadata {key:?} has no value"))
    })?;
    Ok((key, value))
}

fn validate_ipc_field(field: arrow_ipc::Field<'_>) -> Result<()> {
    let name = field
        .name()
        .ok_or_else(|| Error::incompatible("Arrow field is missing its name"))?;
    let invalid = |reason: &str| Error::incompatible(format!("Arrow field {name:?}: {reason}"));
    if field.type_().is_none() {
        return Err(invalid("missing type payload"));
    }
    if let Some(metadata) = field.custom_metadata() {
        let mut keys = HashSet::new();
        for entry in metadata {
            let (key, _) = metadata_entry(entry, name)?;
            if !keys.insert(key) {
                return Err(invalid(&format!("duplicate metadata key {key:?}")));
            }
        }
    }
    if let Some(dictionary) = field.dictionary() {
        let index = dictionary
            .indexType()
            .ok_or_else(|| invalid("dictionary index type is missing"))?;
        if !matches!(index.bitWidth(), 8 | 16 | 32 | 64) {
            return Err(invalid("dictionary index width must be 8, 16, 32, or 64"));
        }
        if dictionary.dictionaryKind() != arrow_ipc::DictionaryKind::DenseArray {
            return Err(invalid("unknown dictionary kind"));
        }
    }
    let children = field.children();
    let child_count = children.map_or(0, |children| children.len());
    let required_children = match field.type_type() {
        Type::List
        | Type::LargeList
        | Type::ListView
        | Type::LargeListView
        | Type::FixedSizeList
        | Type::Map => Some(1),
        Type::RunEndEncoded => Some(2),
        Type::Struct_ | Type::Union => None,
        _ => Some(0),
    };
    if required_children.is_some_and(|expected| expected != child_count) {
        return Err(invalid(&format!(
            "type {:?} has {child_count} children, expected {required_children:?}",
            field.type_type()
        )));
    }
    let valid_type = match field.type_type() {
        Type::Null
        | Type::Bool
        | Type::Binary
        | Type::LargeBinary
        | Type::BinaryView
        | Type::Utf8
        | Type::LargeUtf8
        | Type::Utf8View
        | Type::List
        | Type::LargeList
        | Type::ListView
        | Type::LargeListView
        | Type::Struct_ => true,
        Type::Int => field
            .type_as_int()
            .is_some_and(|value| matches!(value.bitWidth(), 8 | 16 | 32 | 64)),
        Type::FloatingPoint => field.type_as_floating_point().is_some_and(|value| {
            matches!(
                value.precision(),
                Precision::HALF | Precision::SINGLE | Precision::DOUBLE
            )
        }),
        Type::FixedSizeBinary => field
            .type_as_fixed_size_binary()
            .is_some_and(|value| value.byteWidth() >= 0),
        Type::FixedSizeList => field
            .type_as_fixed_size_list()
            .is_some_and(|value| value.listSize() >= 0),
        Type::Date => field
            .type_as_date()
            .is_some_and(|value| matches!(value.unit(), DateUnit::DAY | DateUnit::MILLISECOND)),
        Type::Time => field.type_as_time().is_some_and(|value| {
            matches!(
                (value.bitWidth(), value.unit()),
                (32, TimeUnit::SECOND | TimeUnit::MILLISECOND)
                    | (64, TimeUnit::MICROSECOND | TimeUnit::NANOSECOND)
            )
        }),
        Type::Timestamp => field
            .type_as_timestamp()
            .is_some_and(|value| valid_time_unit(value.unit())),
        Type::Duration => field
            .type_as_duration()
            .is_some_and(|value| valid_time_unit(value.unit())),
        Type::Interval => field.type_as_interval().is_some_and(|value| {
            matches!(
                value.unit(),
                IntervalUnit::YEAR_MONTH | IntervalUnit::DAY_TIME | IntervalUnit::MONTH_DAY_NANO
            )
        }),
        Type::Decimal => field.type_as_decimal().is_some_and(|value| {
            let max_precision = match value.bitWidth() {
                32 => 9,
                64 => 18,
                128 => 38,
                256 => 76,
                _ => 0,
            };
            value.precision() > 0
                && value.precision() <= max_precision
                && i8::try_from(value.scale()).is_ok()
                && value.scale() <= value.precision()
        }),
        Type::Union => field.type_as_union().is_some_and(|value| {
            if !matches!(value.mode(), UnionMode::Dense | UnionMode::Sparse) {
                return false;
            }
            match value.typeIds() {
                None => child_count <= 128,
                Some(ids) => {
                    let mut unique = HashSet::new();
                    ids.len() == child_count
                        && ids
                            .iter()
                            .all(|id| (0..=127).contains(&id) && unique.insert(id))
                }
            }
        }),
        Type::Map => children.is_some_and(|children| {
            let entries = children.get(0);
            !entries.nullable()
                && entries.dictionary().is_none()
                && entries.type_type() == Type::Struct_
                && entries
                    .children()
                    .is_some_and(|fields| fields.len() == 2 && !fields.get(0).nullable())
        }),
        Type::RunEndEncoded => children.is_some_and(|children| {
            let run_ends = children.get(0);
            !run_ends.nullable()
                && run_ends.dictionary().is_none()
                && run_ends.type_as_int().is_some_and(|value| {
                    value.is_signed() && matches!(value.bitWidth(), 16 | 32 | 64)
                })
        }),
        _ => false,
    };
    if !valid_type {
        return Err(invalid(&format!(
            "invalid or unsupported {:?} type parameters",
            field.type_type()
        )));
    }
    if let Some(children) = children {
        for child in children {
            validate_ipc_field(child)?;
        }
    }
    Ok(())
}

fn valid_time_unit(unit: TimeUnit) -> bool {
    matches!(
        unit,
        TimeUnit::SECOND | TimeUnit::MILLISECOND | TimeUnit::MICROSECOND | TimeUnit::NANOSECOND
    )
}

/// The complete scalar signature, preserving Arrow metadata and complex types.
#[derive(Debug, Clone)]
pub struct Schemas {
    input: SchemaRef,
    output: SchemaRef,
    initialization: SchemaRef,
}

impl Schemas {
    /// Validate three independently serialized Schema messages.
    ///
    /// The output must contain exactly one field. Input and initialization schemas
    /// may have zero fields; the initialization *batch* must still have one row.
    pub fn from_ipc(
        input: &[u8],
        output: &[u8],
        initialization: &[u8],
        extensions: &ExtensionTypes,
    ) -> Result<Self> {
        let input = read_schema(input, extensions)?;
        let output = read_schema(output, extensions)?;
        let initialization = read_schema(initialization, extensions)?;
        Self::new(input, output, initialization)
    }

    fn new(input: SchemaRef, output: SchemaRef, initialization: SchemaRef) -> Result<Self> {
        if output.fields().len() != 1 {
            return Err(Error::incompatible(format!(
                "scalar output schema has {} fields; expected exactly one",
                output.fields().len()
            )));
        }
        Ok(Self {
            input,
            output,
            initialization,
        })
    }

    /// Load declared schemas from a complete immutable image, resolving image symlinks.
    ///
    /// `max_schema_bytes` limits each serialized file, including its IPC prefix.
    /// Files larger than this caller-selected budget are rejected before reading
    /// their contents. Each file is parsed and its byte buffer released before
    /// loading the next. This is not a limit on decoded Arrow schema allocations.
    pub fn from_image(
        descriptor: &Descriptor,
        root: &ImageRoot,
        extensions: &ExtensionTypes,
        max_schema_bytes: u64,
    ) -> Result<Self> {
        let paths = descriptor.schemas();
        let load = |path| {
            let resolved = root.resolve(path, PathKind::File)?;
            let io_error =
                |error| Error::incompatible(format!("schema {}: {error}", path.as_str()));
            let file = fs::File::open(resolved).map_err(io_error)?;
            let size = file.metadata().map_err(io_error)?.len();
            if size > max_schema_bytes {
                return Err(Error::incompatible(format!(
                    "schema {} has {size} bytes; limit is {max_schema_bytes} bytes",
                    path.as_str()
                )));
            }
            let mut bytes = Vec::new();
            file.take(max_schema_bytes.saturating_add(1))
                .read_to_end(&mut bytes)
                .map_err(io_error)?;
            if bytes.len() as u64 > max_schema_bytes {
                return Err(Error::incompatible(format!(
                    "schema {} exceeds limit of {max_schema_bytes} bytes while reading",
                    path.as_str()
                )));
            }
            read_schema(&bytes, extensions)
        };
        Self::new(
            load(&paths.input)?,
            load(&paths.output)?,
            load(&paths.initialization)?,
        )
    }

    /// Exact input signature, including schema and nested field metadata.
    pub fn input(&self) -> &SchemaRef {
        &self.input
    }

    /// Exactly one output field, including its name, metadata, and nullability.
    pub fn output(&self) -> &SchemaRef {
        &self.output
    }

    /// Signature of the instance's fixed initialization row.
    pub fn initialization(&self) -> &SchemaRef {
        &self.initialization
    }

    /// Validate one initialization row without conversions or user-code execution.
    ///
    /// For no parameters, construct a zero-field batch with an explicit row count
    /// of one using [`arrow_array::RecordBatchOptions::with_row_count`].
    pub fn validate_initialization(&self, batch: &RecordBatch) -> Result<()> {
        let invalid = |message: String| Error::new(ErrorCode::InitializationFailed, message);
        if batch.num_rows() != 1 {
            return Err(invalid(format!(
                "initialization requires exactly one row, got {}",
                batch.num_rows()
            )));
        }
        if batch.schema_ref() != &self.initialization {
            return Err(invalid(
                "initialization schema differs from the declared Arrow schema".into(),
            ));
        }
        for (field, array) in self.initialization.fields().iter().zip(batch.columns()) {
            // RecordBatchOptions can explicitly ignore nested field names when
            // constructing a batch. Function signatures require exact types.
            if array.data_type() != field.data_type() {
                return Err(invalid(format!(
                    "initialization field {:?} array type differs from its declared type",
                    field.name()
                )));
            }
            if !field.is_nullable() && array.logical_null_count() != 0 {
                return Err(invalid(format!(
                    "non-nullable initialization field {:?} contains NULL",
                    field.name()
                )));
            }
            array.to_data().validate_full().map_err(|error| {
                invalid(format!("initialization field {:?}: {error}", field.name()))
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Int8Type;
    use arrow_array::{
        Array, ArrayRef, DictionaryArray, Float64Array, Int8Array, Int32Array, RecordBatchOptions,
        StringArray, StructArray,
    };
    use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
    use flatbuffers::FlatBufferBuilder;
    use rstest::rstest;

    fn encapsulate(metadata: &[u8]) -> Vec<u8> {
        let length = metadata.len().next_multiple_of(8);
        let mut bytes = vec![0xff; 4];
        bytes.extend_from_slice(&(length as i32).to_le_bytes());
        bytes.extend_from_slice(metadata);
        bytes.resize(8 + length, 0);
        bytes
    }

    fn encode(schema: &Schema) -> Vec<u8> {
        let data = IpcDataGenerator {}.schema_to_bytes_with_dictionary_tracker(
            schema,
            &mut DictionaryTracker::new(false),
            &IpcWriteOptions::default(),
        );
        encapsulate(&data.ipc_message)
    }

    fn empty() -> Vec<u8> {
        encode(&Schema::empty())
    }

    fn output() -> Vec<u8> {
        encode(&Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]))
    }

    #[test]
    fn preserves_order_nested_types_nullability_and_metadata() {
        let element = Field::new("item", DataType::Float32, false)
            .with_metadata(HashMap::from([("model".into(), "v1".into())]));
        let schema = Schema::new_with_metadata(
            vec![
                Field::new(
                    "z",
                    DataType::Struct(
                        vec![
                            Field::new(
                                "vector",
                                DataType::FixedSizeList(Arc::new(element), 3),
                                true,
                            ),
                            Field::new(
                                "labels",
                                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                                false,
                            ),
                        ]
                        .into(),
                    ),
                    true,
                )
                .with_metadata(HashMap::from([("field-note".into(), "保留".into())])),
                Field::new("a", DataType::Float64, false),
            ],
            HashMap::from([("schema-note".into(), "ordered".into())]),
        );
        assert_eq!(
            read_schema(&encode(&schema), &ExtensionTypes::default())
                .unwrap()
                .as_ref(),
            &schema
        );
    }

    #[rstest]
    #[case::decimal(DataType::Decimal256(76, -10))]
    #[case::time(DataType::Time64(arrow_schema::TimeUnit::Nanosecond))]
    #[case::timestamp(DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("Asia/Shanghai".into())))]
    #[case::dictionary(DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)))]
    #[case::empty_struct(DataType::Struct(Vec::<Field>::new().into()))]
    #[case::list_view(DataType::ListView(Arc::new(Field::new(
        "item",
        DataType::BinaryView,
        true
    ))))]
    #[case::run_end_encoded(DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", DataType::Int32, false)),
        Arc::new(Field::new("values", DataType::Utf8, true))
    ))]
    fn preserves_arrow_types(#[case] data_type: DataType) {
        let schema = Schema::new(vec![Field::new("value", data_type, true)]);
        assert_eq!(
            read_schema(&encode(&schema), &ExtensionTypes::default())
                .unwrap()
                .as_ref(),
            &schema
        );
    }

    #[test]
    fn rejects_extra_messages_truncation_and_missing_prefix() {
        let valid = output();
        for length in 0..valid.len() {
            let error = read_schema(&valid[..length], &ExtensionTypes::default()).unwrap_err();
            assert_eq!(error.code, ErrorCode::Incompatible);
            assert!(error.message.contains("Arrow"));
        }
        let mut extra = valid.clone();
        extra.extend_from_slice(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]);
        assert!(
            read_schema(&extra, &ExtensionTypes::default())
                .unwrap_err()
                .message
                .contains("exactly one")
        );
        assert!(read_schema(&valid[8..], &ExtensionTypes::default()).is_err());
    }

    #[derive(Clone, Copy)]
    enum RawCase {
        DuplicateSchema,
        DuplicateField,
        DuplicateNestedField,
        InvalidInt,
        OldVersion,
        Body,
        MissingType,
        ListWithoutChild,
    }

    fn raw_message(case: RawCase) -> Vec<u8> {
        let mut builder = FlatBufferBuilder::new();
        let key = builder.create_string("same");
        let value = builder.create_string("value");
        let entry = arrow_ipc::KeyValue::create(
            &mut builder,
            &arrow_ipc::KeyValueArgs {
                key: Some(key),
                value: Some(value),
            },
        );
        let duplicate = builder.create_vector(&[entry, entry]);
        let int = arrow_ipc::Int::create(
            &mut builder,
            &arrow_ipc::IntArgs {
                bitWidth: if matches!(case, RawCase::InvalidInt) {
                    7
                } else {
                    32
                },
                is_signed: true,
            },
        );
        let name = builder.create_string("field");
        let mut args = arrow_ipc::FieldArgs {
            name: Some(name),
            type_type: Type::Int,
            type_: Some(int.as_union_value()),
            ..Default::default()
        };
        if matches!(
            case,
            RawCase::DuplicateField | RawCase::DuplicateNestedField
        ) {
            args.custom_metadata = Some(duplicate);
        }
        if matches!(case, RawCase::MissingType) {
            args.type_ = None;
        }
        if matches!(case, RawCase::ListWithoutChild) {
            let list = arrow_ipc::List::create(&mut builder, &arrow_ipc::ListArgs::default());
            args.type_type = Type::List;
            args.type_ = Some(list.as_union_value());
        }
        let mut field = arrow_ipc::Field::create(&mut builder, &args);
        if matches!(case, RawCase::DuplicateNestedField) {
            let nested = builder.create_vector(&[field]);
            let structure =
                arrow_ipc::Struct_::create(&mut builder, &arrow_ipc::Struct_Args::default());
            field = arrow_ipc::Field::create(
                &mut builder,
                &arrow_ipc::FieldArgs {
                    name: Some(name),
                    type_type: Type::Struct_,
                    type_: Some(structure.as_union_value()),
                    children: Some(nested),
                    ..Default::default()
                },
            );
        }
        let fields = builder.create_vector(&[field]);
        let schema = arrow_ipc::Schema::create(
            &mut builder,
            &arrow_ipc::SchemaArgs {
                fields: Some(fields),
                custom_metadata: if matches!(case, RawCase::DuplicateSchema) {
                    Some(duplicate)
                } else {
                    None
                },
                ..Default::default()
            },
        );
        let message = arrow_ipc::Message::create(
            &mut builder,
            &arrow_ipc::MessageArgs {
                version: if matches!(case, RawCase::OldVersion) {
                    arrow_ipc::MetadataVersion::V4
                } else {
                    arrow_ipc::MetadataVersion::V5
                },
                header_type: arrow_ipc::MessageHeader::Schema,
                header: Some(schema.as_union_value()),
                bodyLength: if matches!(case, RawCase::Body) { 1 } else { 0 },
                ..Default::default()
            },
        );
        builder.finish(message, None);
        encapsulate(builder.finished_data())
    }

    #[rstest]
    #[case::schema_duplicate(RawCase::DuplicateSchema, "duplicate Arrow schema")]
    #[case::field_duplicate(RawCase::DuplicateField, "duplicate metadata")]
    #[case::nested_duplicate(RawCase::DuplicateNestedField, "duplicate metadata")]
    #[case::invalid_integer(RawCase::InvalidInt, "type parameters")]
    #[case::v4(RawCase::OldVersion, "V5")]
    #[case::body(RawCase::Body, "no body")]
    #[case::missing_type(RawCase::MissingType, "type")]
    #[case::list_child(RawCase::ListWithoutChild, "children")]
    fn rejects_malformed_raw_schema(#[case] case: RawCase, #[case] reason: &str) {
        let error = read_schema(&raw_message(case), &ExtensionTypes::default()).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert!(error.message.contains(reason), "{error}");
    }

    #[test]
    fn unknown_nested_extensions_require_a_storage_validator() {
        let extension = |data_type| {
            Field::new("identifier", data_type, true).with_metadata(HashMap::from([
                ("ARROW:extension:name".into(), "example.identifier".into()),
                ("ARROW:extension:metadata".into(), "{}".into()),
            ]))
        };
        let schema = |data_type| {
            Schema::new(vec![Field::new(
                "nested",
                DataType::Struct(vec![extension(data_type)].into()),
                true,
            )])
        };
        let encoded = encode(&schema(DataType::FixedSizeBinary(16)));
        let mut extensions = ExtensionTypes::default();
        let error = read_schema(&encoded, &extensions).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert!(error.message.contains("unknown extension"));
        extensions
            .register("example.identifier", |field| {
                if field.data_type() == &DataType::FixedSizeBinary(16)
                    && field.metadata()["ARROW:extension:metadata"] == "{}"
                {
                    Ok(())
                } else {
                    Err(Error::incompatible(
                        "identifier storage or metadata is invalid",
                    ))
                }
            })
            .unwrap();
        assert_eq!(
            read_schema(&encoded, &extensions).unwrap().as_ref(),
            &schema(DataType::FixedSizeBinary(16))
        );
        assert!(
            read_schema(&encode(&schema(DataType::Int32)), &extensions)
                .unwrap_err()
                .message
                .contains("storage")
        );
    }

    #[test]
    fn output_requires_one_field_and_zero_field_initialization_requires_one_row() {
        let extensions = ExtensionTypes::default();
        let error = Schemas::from_ipc(&empty(), &empty(), &empty(), &extensions).unwrap_err();
        assert_eq!(error.code, ErrorCode::Incompatible);
        assert!(error.message.contains("exactly one"));
        let two = encode(&Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        assert!(
            Schemas::from_ipc(&empty(), &two, &empty(), &extensions)
                .unwrap_err()
                .message
                .contains("2 fields")
        );
        let schemas = Schemas::from_ipc(&empty(), &output(), &empty(), &extensions).unwrap();
        for rows in [0, 1, 2] {
            let batch = RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .unwrap();
            if rows == 1 {
                schemas.validate_initialization(&batch).unwrap();
            } else {
                let error = schemas.validate_initialization(&batch).unwrap_err();
                assert_eq!(error.code, ErrorCode::InitializationFailed);
                assert!(error.message.contains("exactly one row"));
            }
        }
    }

    #[test]
    fn initialization_matches_full_schema_without_coercion() {
        let schema = Arc::new(Schema::new_with_metadata(
            vec![Field::new("factor", DataType::Float64, false)],
            HashMap::from([("version".into(), "1".into())]),
        ));
        let schemas = Schemas::from_ipc(
            &empty(),
            &output(),
            &encode(&schema),
            &ExtensionTypes::default(),
        )
        .unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![2.0]))],
        )
        .unwrap();
        schemas.validate_initialization(&batch).unwrap();
        let no_metadata = RecordBatch::try_new(
            Arc::new(Schema::new(schema.fields().clone())),
            batch.columns().to_vec(),
        )
        .unwrap();
        let error = schemas.validate_initialization(&no_metadata).unwrap_err();
        assert_eq!(error.code, ErrorCode::InitializationFailed);
        assert!(error.message.contains("schema differs"));
    }

    #[test]
    fn initialization_rejects_logical_dictionary_nulls() {
        let array = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![0]),
            Arc::new(StringArray::from(vec![None::<&str>])),
        )
        .unwrap();
        assert_eq!(array.null_count(), 0);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "parameter",
            array.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        let schemas = Schemas::from_ipc(
            &empty(),
            &output(),
            &encode(&schema),
            &ExtensionTypes::default(),
        )
        .unwrap();
        let error = schemas.validate_initialization(&batch).unwrap_err();
        assert_eq!(error.code, ErrorCode::InitializationFailed);
        assert!(error.message.contains("contains NULL"));
    }

    #[test]
    fn initialization_checks_array_types_even_if_batch_construction_ignored_names() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "parameter",
            DataType::Struct(vec![Field::new("declared", DataType::Int32, true)].into()),
            true,
        )]));
        let array = StructArray::from(vec![(
            Arc::new(Field::new("actual", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
        )]);
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            vec![Arc::new(array)],
            &RecordBatchOptions::new().with_match_field_names(false),
        )
        .unwrap();
        let schemas = Schemas::from_ipc(
            &empty(),
            &output(),
            &encode(&schema),
            &ExtensionTypes::default(),
        )
        .unwrap();
        let error = schemas.validate_initialization(&batch).unwrap_err();
        assert_eq!(error.code, ErrorCode::InitializationFailed);
        assert!(error.message.contains("array type differs"));
    }
}
