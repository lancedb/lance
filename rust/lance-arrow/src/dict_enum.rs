use std::sync::Arc;

use arrow_array::{cast::AsArray, Array, StringArray};
use arrow_schema::{ArrowError, Field as ArrowField};
use serde::{Deserialize, Serialize};

use crate::{
    bfloat16::{ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY},
    DataTypeExt, Result,
};

const ENUM_TYPE: &str = "polars.enum";

// TODO: Could be slightly more efficient to use custom JSON serialization
// to go straight from JSON to StringArray without the Vec<String> intermediate
// but this is fine for now
#[derive(Deserialize, Serialize)]
struct DictionaryEnumMetadata {
    categories: Vec<String>,
}

pub struct DictionaryEnumType {
    pub categories: Arc<dyn Array>,
}

impl DictionaryEnumType {
    /// Adds extension type metadata to the given field
    ///
    /// Fails if the field is already an extension type of some kind
    pub fn wrap_field(&self, field: &ArrowField) -> Result<ArrowField> {
        let mut metadata = field.metadata().clone();
        if metadata.contains_key(ARROW_EXT_NAME_KEY) {
            return Err(ArrowError::InvalidArgumentError(
                "Field already has extension metadata".to_string(),
            ));
        }
        metadata.insert(ARROW_EXT_NAME_KEY.to_string(), ENUM_TYPE.to_string());
        metadata.insert(
            ARROW_EXT_META_KEY.to_string(),
            serde_json::to_string(&DictionaryEnumMetadata {
                categories: self
                    .categories
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .values()
                    .iter()
                    .map(|x| x.to_string())
                    .collect(),
            })
            .unwrap(),
        );
        Ok(field.clone().with_metadata(metadata))
    }

    /// Creates a new enum type from the given dictionary array
    ///
    /// # Arguments
    ///
    /// * `arr` - The dictionary array to create the enum type from
    ///
    /// # Errors
    ///
    /// An error is returned if the array is not a dictionary array or if the dictionary
    /// array does not have string values
    pub fn from_dict_array(arr: &dyn Array) -> Result<Self> {
        let arr = arr.as_any_dictionary_opt().ok_or_else(|| {
            ArrowError::InvalidArgumentError(
                "Expected a dictionary array for enum type".to_string(),
            )
        })?;
        if !arr.values().data_type().is_binary_like() {
            Err(ArrowError::InvalidArgumentError(
                "Expected a dictionary array with string values for enum type".to_string(),
            ))
        } else {
            Ok(Self {
                categories: Arc::new(arr.values().clone()),
            })
        }
    }

    /// Attempts to parse the type from the given field
    ///
    /// If the field is not an enum type then None is returned
    ///
    /// Errors can occur if the field is an enum type but the metadata
    /// is not correctly formatted
    ///
    /// # Arguments
    ///
    /// * `field` - The field to parse
    /// * `sample_arr` - An optional sample array.  If provided then categories will be extracted
    /// from this array, avoiding the need to parse the metadata.  This array should be a dictionary
    /// array where the dictionary items are the categories.
    ///
    /// The sample_arr is only used if the field is an enum type.  E.g. it is safe to do something
    /// like:
    ///
    /// ```ignore
    /// let arr = batch.column(0);
    /// let field = batch.schema().field(0);
    /// let enum_type = DictionaryEnumType::from_field(field, Some(arr));
    /// ```
    pub fn from_field(
        field: &ArrowField,
        sample_arr: Option<&Arc<dyn Array>>,
    ) -> Result<Option<Self>> {
        if field
            .metadata()
            .get(ARROW_EXT_NAME_KEY)
            .map(|k| k.eq_ignore_ascii_case(ENUM_TYPE))
            .unwrap_or(false)
        {
            // Prefer extracting values from the first array if possible as it's cheaper
            if let Some(arr) = sample_arr {
                let dict_arr = arr.as_any_dictionary_opt().ok_or_else(|| {
                    ArrowError::InvalidArgumentError(
                        "Expected a dictionary array for enum type".to_string(),
                    )
                })?;
                Ok(Some(Self {
                    categories: dict_arr.values().clone(),
                }))
            } else {
                // No arrays, need to use the field metadata
                let meta = field.metadata().get(ARROW_EXT_META_KEY).ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!(
                        "Field {} is missing extension metadata",
                        field.name()
                    ))
                })?;
                let meta: DictionaryEnumMetadata = serde_json::from_str(meta).map_err(|e| {
                    ArrowError::InvalidArgumentError(format!(
                        "Arrow extension metadata for enum was not correctly formed: {}",
                        e
                    ))
                })?;
                let categories = Arc::new(StringArray::from_iter_values(meta.categories));
                Ok(Some(Self { categories }))
            }
        } else {
            Ok(None)
        }
    }
}
