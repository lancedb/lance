// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading, checking and caching a PCA pre-transform at index load time.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::cache::{
    CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheKey, CacheKeySchema, KeyBuilder,
    LanceCache,
};
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::deepsize::DeepSizeOf;
use lance_file::reader::FileReader;
use lance_index::pb::PcaParameters;
use lance_index::scalar::IndexReader;
use lance_index::vector::VectorIndex;
use lance_index::vector::pca::{
    PCA_METADATA_KEY, PcaModel, decode_pca_model, encode_pca_model, read_pca_model,
};
use lance_index::vector::pretransform::PreTransformIndex;
use lance_index::vector::quantizer::Quantizer;
use lance_index::{INDEX_FILE_NAME, IndexType};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use super::ivf::v2::IvfStateEntryBox;

use super::details::{derive_vector_index_type, pca_params_from_details};
use super::utils::infer_vector_dim;
use super::{PcaComponentMismatch, pca_component_mismatch};
use crate::session::index_caches::write_index_identity;
use crate::{Error, Result};

struct PcaFileReaderAdapter<'a> {
    reader: &'a FileReader,
    schema: LanceSchema,
}

impl<'a> PcaFileReaderAdapter<'a> {
    fn new(reader: &'a FileReader) -> Result<Self> {
        Ok(Self {
            reader,
            schema: reader.schema().as_ref().try_into()?,
        })
    }
}

#[async_trait]
impl IndexReader for PcaFileReaderAdapter<'_> {
    async fn read_record_batch(&self, _n: u64, _batch_size: u64) -> Result<RecordBatch> {
        Err(Error::not_supported(
            "PCA file-reader adapter does not read record batches",
        ))
    }

    async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
        self.reader.read_global_buffer(index).await
    }

    async fn read_range(
        &self,
        _range: std::ops::Range<usize>,
        _projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        Err(Error::not_supported(
            "PCA file-reader adapter does not read row ranges",
        ))
    }

    async fn num_batches(&self, _batch_size: u64) -> u32 {
        0
    }

    fn num_rows(&self) -> usize {
        self.reader.metadata().num_rows as usize
    }

    fn schema(&self) -> &LanceSchema {
        &self.schema
    }
}

/// Reads the optional pre-transform declared by an index file's schema metadata.
pub async fn read_pre_transform(reader: &FileReader) -> Result<Option<Arc<PcaModel>>> {
    let Some(buffer_index) = reader.schema().metadata.get(PCA_METADATA_KEY) else {
        return Ok(None);
    };
    let adapter = PcaFileReaderAdapter::new(reader)?;
    read_pca_model(&adapter, buffer_index)
        .await
        .map(Arc::new)
        .map(Some)
}

pub fn format_pca_fingerprint(fingerprint: &[u8; 32]) -> String {
    fingerprint
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn manifest_pca(index_meta: &IndexMetadata) -> Result<Option<(usize, [u8; 32])>> {
    let (pca, details_index_type) = match index_meta.index_details.as_deref() {
        Some(details) => (
            pca_params_from_details(details)?
                .map(|(params, fingerprint)| (params.dimension, fingerprint)),
            Some(derive_vector_index_type(details)),
        ),
        None => (None, None),
    };
    match (index_meta.index_version, pca) {
        (3, Some(pca)) if details_index_type.as_deref() == Some("IVF_RQ") => Ok(Some(pca)),
        (3, Some(_)) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "manifest index_version=3 details index_type actual={details_index_type:?} expected=IVF_RQ for index {}",
                index_meta.uuid
            ),
        )),
        (3, None) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "manifest index_version=3 requires index_details with PCA parameters and a 32-byte fingerprint for index {}",
                index_meta.uuid
            ),
        )),
        (version, Some(_)) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "manifest index_version={version} must not declare PCA details for index {}",
                index_meta.uuid
            ),
        )),
        (_, None) => Ok(None),
    }
}

pub fn validate_pca_manifest_and_model(
    index_meta: &IndexMetadata,
    physical_index_type: Option<&str>,
    model: Option<&Arc<PcaModel>>,
) -> Result<()> {
    let manifest = manifest_pca(index_meta)?;
    if let Some(index_type) = physical_index_type
        && (index_meta.index_version == 3 || model.is_some())
        && index_type != "IVF_RQ"
    {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={} physical index_type={index_type} contains lance:pca; expected index_type=IVF_RQ",
                index_meta.index_version
            ),
        ));
    }
    match (index_meta.index_version, manifest, model) {
        (3, Some((manifest_dimension, expected_fingerprint)), Some(model)) => {
            let actual_fingerprint = model.fingerprint();
            if actual_fingerprint != expected_fingerprint {
                return Err(Error::corrupt_file_named(
                    INDEX_FILE_NAME,
                    format!(
                        "index_version=3 lance:pca fingerprint mismatch: expected={}, actual={}",
                        format_pca_fingerprint(&expected_fingerprint),
                        format_pca_fingerprint(&actual_fingerprint)
                    ),
                ));
            }
            if model.output_dimension() != manifest_dimension {
                return Err(Error::corrupt_file_named(
                    INDEX_FILE_NAME,
                    format!(
                        "index_version=3 PCA output_dimension actual={} expected={} from manifest details",
                        model.output_dimension(),
                        manifest_dimension
                    ),
                ));
            }
            Ok(())
        }
        (3, Some(_), None) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            "index_version=3 requires physical lance:pca metadata and model buffer",
        )),
        (version, None, Some(_)) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={version} must not contain physical lance:pca metadata or model buffer"
            ),
        )),
        (_, None, None) => Ok(()),
        (version, Some(_), _) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "manifest index_version={version} has an invalid PCA declaration for index {}",
                index_meta.uuid
            ),
        )),
    }
}

pub fn validate_pca_index_dimensions(
    index_meta: &IndexMetadata,
    model: Option<&Arc<PcaModel>>,
    index: &Arc<dyn VectorIndex>,
    expected_input_dimension: usize,
) -> Result<()> {
    let Some(model) = model else {
        return Ok(());
    };
    if index.index_type() != IndexType::IvfRq {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={} PCA index type actual={:?} expected=IvfRq",
                index_meta.index_version,
                index.index_type()
            ),
        ));
    }
    let ivf = index.ivf_model();
    let centroids = ivf.centroids_array().ok_or_else(|| {
        Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={} PCA index is missing IVF centroids",
                index_meta.index_version
            ),
        )
    })?;
    let quantizer = index.quantizer();
    let Quantizer::Rabit(rq) = quantizer else {
        return Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={} PCA quantizer actual={} expected=Rabit",
                index_meta.index_version,
                quantizer.quantization_type()
            ),
        ));
    };
    validate_pca_shape(
        index_meta.index_version,
        model,
        expected_input_dimension,
        ivf.dimension(),
        &centroids.value_type(),
        rq.metadata_ref().rotated_dim(),
    )
}

pub fn validate_pca_shape(
    index_version: i32,
    model: &PcaModel,
    expected_input_dimension: usize,
    ivf_dimension: usize,
    centroid_type: &DataType,
    rq_rotated_dimension: usize,
) -> Result<()> {
    match pca_component_mismatch(
        model,
        expected_input_dimension,
        ivf_dimension,
        centroid_type,
        rq_rotated_dimension,
    ) {
        None => Ok(()),
        Some(PcaComponentMismatch::InputDimension { actual, expected }) => {
            Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "index_version={index_version} PCA input_dimension actual={actual} expected={expected} from dataset field"
                ),
            ))
        }
        Some(PcaComponentMismatch::CentroidType { actual }) => Err(Error::corrupt_file_named(
            INDEX_FILE_NAME,
            format!(
                "index_version={index_version} IVF centroid dtype actual={actual} expected=Float32"
            ),
        )),
        Some(PcaComponentMismatch::OutputDimension { actual, expected }) => {
            Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "index_version={index_version} PCA output_dimension actual={actual} expected={expected} from IVF centroids"
                ),
            ))
        }
        Some(PcaComponentMismatch::RqDimension { actual, expected }) => {
            Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "index_version={index_version} RQ metadata rotated_dim actual={actual} expected={expected} from PCA output_dimension"
                ),
            ))
        }
    }
}

pub fn validate_logical_pca_fingerprints<'a>(
    index_name: &str,
    segments: impl IntoIterator<Item = &'a IndexMetadata>,
) -> Result<()> {
    let mut expected = None;
    for (position, metadata) in segments.into_iter().enumerate() {
        let actual = manifest_pca(metadata)?.map(|(_, fingerprint)| fingerprint);
        if position == 0 {
            expected = actual;
        } else if actual != expected {
            return Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "logical index {index_name} has inconsistent PCA fingerprints across segments: expected={:?}, segment={} actual={:?}",
                    expected.map(|fingerprint| format_pca_fingerprint(&fingerprint)),
                    metadata.uuid,
                    actual.map(|fingerprint| format_pca_fingerprint(&fingerprint))
                ),
            ));
        }
    }
    Ok(())
}

/// Wraps a loaded IVF index with its pre-transform and checks the pair.
fn attach_loaded_pre_transform(
    index: Arc<dyn VectorIndex>,
    model: Option<&Arc<PcaModel>>,
    index_meta: &IndexMetadata,
    input_dimension: usize,
) -> Result<Arc<dyn VectorIndex>> {
    let index = match model {
        Some(model) => {
            Arc::new(PreTransformIndex::new_pca(index, (*model).clone())) as Arc<dyn VectorIndex>
        }
        None => index,
    };
    validate_pca_index_dimensions(index_meta, model, &index, input_dimension)?;
    Ok(index)
}

pub struct PreTransformIndexStateCacheKey<'a> {
    uuid: &'a Uuid,
    fri_uuid: Option<&'a Uuid>,
}

impl<'a> PreTransformIndexStateCacheKey<'a> {
    pub(crate) fn new(uuid: &'a Uuid, fri_uuid: Option<&'a Uuid>) -> Self {
        Self { uuid, fri_uuid }
    }
}

impl CacheKey for PreTransformIndexStateCacheKey<'_> {
    type ValueType = PreTransformIndexState;

    fn type_name() -> &'static str {
        "PreTransformIndexState"
    }

    fn key(&self) -> std::borrow::Cow<'_, str> {
        if let Some(fri_uuid) = self.fri_uuid {
            format!("pca-{}-{}", self.uuid, fri_uuid).into()
        } else {
            format!("pca-{}", self.uuid).into()
        }
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.pca-state-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        write_index_identity(builder, self.uuid, self.fri_uuid);
    }

    fn codec() -> Option<lance_core::cache::CacheCodec> {
        Some(lance_core::cache::CacheCodec::from_impl::<
            PreTransformIndexState,
        >())
    }
}

/// The pre-transform an index segment declares in its manifest.
///
/// This is the only type the vector index open path consults about
/// pre-transforms: cached state, file container checks, reading the stored
/// model and wrapping the opened index all go through it.
pub struct DeclaredPreTransform<'a> {
    index_meta: &'a IndexMetadata,
    frag_reuse_uuid: Option<&'a Uuid>,
    declared: bool,
}

impl<'a> DeclaredPreTransform<'a> {
    pub(crate) fn from_manifest(
        index_meta: &'a IndexMetadata,
        frag_reuse_uuid: Option<&'a Uuid>,
    ) -> Result<Self> {
        Ok(Self {
            index_meta,
            frag_reuse_uuid,
            declared: manifest_pca(index_meta)?.is_some(),
        })
    }

    pub(crate) fn is_declared(&self) -> bool {
        self.declared
    }

    fn state_key(&self) -> PreTransformIndexStateCacheKey<'_> {
        PreTransformIndexStateCacheKey::new(&self.index_meta.uuid, self.frag_reuse_uuid)
    }

    /// The cached state of a declared pre-transformed index, checked against
    /// the manifest.
    pub(crate) async fn cached_state(
        &self,
        cache: &LanceCache,
    ) -> Result<Option<Arc<PreTransformIndexState>>> {
        if !self.declared {
            return Ok(None);
        }
        let Some(entry) = cache.get_with_key(&self.state_key()).await else {
            return Ok(None);
        };
        validate_pca_manifest_and_model(self.index_meta, None, Some(&entry.model))?;
        Ok(Some(entry))
    }

    /// Rejects index file containers that cannot hold a pre-transform.
    pub(crate) fn check_container_version(&self, major: u16, minor: u16) -> Result<()> {
        if self.declared && !matches!((major, minor), (0, 3) | (2, _)) {
            return Err(Error::corrupt_file_named(
                INDEX_FILE_NAME,
                format!(
                    "manifest index_version={} declares PCA for index {}, but file container version=({major},{minor}) cannot store lance:pca global-buffer metadata",
                    self.index_meta.index_version, self.index_meta.uuid
                ),
            ));
        }
        Ok(())
    }

    /// Reads the pre-transform stored in an index file and checks it against
    /// the manifest.
    pub(crate) async fn read(
        &self,
        reader: &FileReader,
        physical_index_type: &str,
    ) -> Result<Option<Arc<PcaModel>>> {
        let model = read_pre_transform(reader).await?;
        validate_pca_manifest_and_model(
            self.index_meta,
            Some(physical_index_type),
            model.as_ref(),
        )?;
        Ok(model)
    }

    /// Wraps an index rebuilt from [`Self::cached_state`].
    pub(crate) fn attach_cached(
        &self,
        index: Arc<dyn VectorIndex>,
        entry: &PreTransformIndexState,
        vector_type: &DataType,
    ) -> Result<Arc<dyn VectorIndex>> {
        attach_loaded_pre_transform(
            index,
            Some(&entry.model),
            self.index_meta,
            infer_vector_dim(vector_type)?,
        )
    }

    /// Wraps a freshly opened IVF index with the model from [`Self::read`]
    /// and caches the pair. Without a model the index and its IVF state pass
    /// through unchanged.
    pub(crate) async fn attach_opened(
        &self,
        cache: &LanceCache,
        index: Arc<dyn VectorIndex>,
        ivf_entry: Option<IvfStateEntryBox>,
        model: Option<Arc<PcaModel>>,
        vector_type: &DataType,
    ) -> Result<(Arc<dyn VectorIndex>, Option<IvfStateEntryBox>)> {
        let Some(model) = model else {
            return Ok((index, ivf_entry));
        };
        let index = attach_loaded_pre_transform(
            index,
            Some(&model),
            self.index_meta,
            infer_vector_dim(vector_type)?,
        )?;
        if let Some(inner) = ivf_entry {
            cache
                .insert_with_key(
                    &self.state_key(),
                    Arc::new(PreTransformIndexState { inner, model }),
                )
                .await;
        }
        Ok((index, None))
    }
}

/// Cached IVF state plus the pre-transform that must be reattached on open.
///
/// The IVF cache entry itself stays free of pre-transform bytes.
pub struct PreTransformIndexState {
    pub(crate) inner: IvfStateEntryBox,
    pub(crate) model: Arc<PcaModel>,
}

impl DeepSizeOf for PreTransformIndexState {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.inner.deep_size_of_children(context) + self.model.deep_size_of_children(context)
    }
}

impl CacheCodecImpl for PreTransformIndexState {
    const TYPE_ID: &'static str = "lance.vector.pretransform.PcaState";
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        let header = PcaParameters {
            dimension: u32::try_from(self.model.output_dimension()).map_err(|_| {
                Error::io(format!(
                    "PCA cache output_dimension={} does not fit in u32",
                    self.model.output_dimension()
                ))
            })?,
            model_fingerprint: self.model.fingerprint().to_vec(),
            ..Default::default()
        };
        let model_bytes = encode_pca_model(self.model.as_ref())?;
        let mut inner_bytes = Vec::new();
        self.inner
            .serialize(&mut CacheEntryWriter::new(&mut inner_bytes))?;
        w.write_header(&header)?;
        w.write_raw(&model_bytes)?;
        w.write_raw(&inner_bytes)?;
        Ok(())
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self> {
        let version = r.version();
        let header: PcaParameters = r.read_header()?;
        let model_bytes = r.read_raw().map_err(|error| {
            Error::io(format!(
                "PCA cache model blob read failed: version={version}, blob_len=unavailable, fingerprint={:?}: {error}",
                header.model_fingerprint
            ))
        })?;
        let blob_len = model_bytes.len();
        let fingerprint = header.model_fingerprint.as_slice();
        if fingerprint.is_empty() {
            return Err(Error::io(format!(
                "PCA cache metadata mismatch: version={version}, blob_len={blob_len}, fingerprint=None: non-empty blob is missing fingerprint"
            )));
        }
        if fingerprint.len() != 32 {
            return Err(Error::io(format!(
                "PCA cache fingerprint length mismatch: version={version}, blob_len={blob_len}, fingerprint_len={}, fingerprint={fingerprint:?}, expected=32",
                fingerprint.len()
            )));
        }
        if model_bytes.is_empty() {
            return Err(Error::io(format!(
                "PCA cache metadata mismatch: version={version}, blob_len={blob_len}, fingerprint_len={}, fingerprint={fingerprint:?}: empty blob has fingerprint",
                fingerprint.len()
            )));
        }
        let model = decode_pca_model(&model_bytes).map_err(|error| {
            Error::io(format!(
                "PCA cache PCA decode failed: version={version}, blob_len={blob_len}, fingerprint_len={}, fingerprint={fingerprint:?}: {error}",
                fingerprint.len()
            ))
        })?;
        let actual_fingerprint = model.fingerprint();
        if fingerprint != actual_fingerprint.as_slice() {
            return Err(Error::io(format!(
                "PCA cache fingerprint mismatch: version={version}, blob_len={blob_len}, fingerprint_len={}, expected={fingerprint:?}, actual={actual_fingerprint:?}",
                fingerprint.len()
            )));
        }
        let inner_bytes = r.read_raw().map_err(|error| {
            Error::io(format!(
                "PCA cache inner IVF blob read failed: version={version}, fingerprint={fingerprint:?}: {error}"
            ))
        })?;
        let inner = IvfStateEntryBox::deserialize(&mut CacheEntryReader::new(
            &inner_bytes,
            0,
            IvfStateEntryBox::CURRENT_VERSION,
        ))?;
        Ok(Self {
            inner,
            model: Arc::new(model),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{FixedSizeListArray, Float32Array};
    use arrow_schema::DataType;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::pca::PcaModel;
    use lance_table::format::IndexMetadata;
    use rstest::rstest;
    use uuid::Uuid;

    use super::*;
    use crate::Error;

    fn test_pca_model() -> Arc<PcaModel> {
        let vectors = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0,
            ]),
            3,
        )
        .unwrap();
        Arc::new(PcaModel::train(&vectors, &lance_index::vector::pca::PcaParams::new(2)).unwrap())
    }

    fn test_pca_details(fingerprint: [u8; 32]) -> Arc<prost_types::Any> {
        use lance_index::pb::vector_index_details::{Compression, PreTransform, RabitQuantization};
        use lance_index::pb::{PcaParameters, VectorIndexDetails, VectorMetricType};

        Arc::new(
            prost_types::Any::from_msg(&VectorIndexDetails {
                metric_type: VectorMetricType::L2.into(),
                target_partition_size: 0,
                hnsw_index_config: None,
                compression: Some(Compression::Rq(RabitQuantization {
                    num_bits: 1,
                    rotation_type: 0,
                })),
                runtime_hints: Default::default(),
                pre_transform: Some(PreTransform::Pca(PcaParameters {
                    dimension: 2,
                    random_rotation: false,
                    whitening: false,
                    epsilon: 1.0e-5,
                    sample_size: 4,
                    max_training_memory_bytes: 1024,
                    sampling_seed: 42,
                    rotation_seed: 5,
                    model_fingerprint: fingerprint.to_vec(),
                })),
            })
            .unwrap(),
        )
    }

    fn test_pca_index_metadata(
        index_version: i32,
        details: Option<Arc<prost_types::Any>>,
    ) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            fields: vec![0],
            covering_fields: Vec::new(),
            name: "pca_idx".to_string(),
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: details,
            index_version,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    #[test]
    fn test_pca_manifest_version_and_physical_presence_validation() {
        let model = test_pca_model();
        let v3_without_details = test_pca_index_metadata(3, None);
        let error =
            validate_pca_manifest_and_model(&v3_without_details, Some("IVF_RQ"), Some(&model))
                .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("index_version=3"));
        assert!(error.to_string().contains("index_details"));

        let v2_with_details =
            test_pca_index_metadata(2, Some(test_pca_details(model.fingerprint())));
        let error = validate_pca_manifest_and_model(&v2_with_details, Some("IVF_RQ"), Some(&model))
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("index_version=2"));
        assert!(error.to_string().contains("PCA details"));

        let v2_with_physical_model = test_pca_index_metadata(2, None);
        let error =
            validate_pca_manifest_and_model(&v2_with_physical_model, Some("IVF_RQ"), Some(&model))
                .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("index_version=2"));
        assert!(error.to_string().contains("physical lance:pca"));

        let v3 = test_pca_index_metadata(3, Some(test_pca_details(model.fingerprint())));
        validate_pca_manifest_and_model(&v3, Some("IVF_RQ"), Some(&model)).unwrap();
        let error = validate_pca_manifest_and_model(&v3, Some("IVF_RQ"), None).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("lance:pca"));

        let error = validate_pca_manifest_and_model(&v3, Some("IVF_PQ"), Some(&model)).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("physical index_type=IVF_PQ"));
    }

    #[test]
    fn test_pca_manifest_rejects_fingerprint_and_segment_mismatch() {
        let model = test_pca_model();
        let mismatched = test_pca_index_metadata(3, Some(test_pca_details([0x5a; 32])));
        let error =
            validate_pca_manifest_and_model(&mismatched, Some("IVF_RQ"), Some(&model)).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("fingerprint mismatch"));
        assert!(error.to_string().contains("expected="));
        assert!(error.to_string().contains("actual="));

        let first = test_pca_index_metadata(3, Some(test_pca_details(model.fingerprint())));
        let second = test_pca_index_metadata(3, Some(test_pca_details([0xa5; 32])));
        let error = validate_logical_pca_fingerprints("pca_idx", [&first, &second]).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("inconsistent PCA fingerprints"));
        assert!(error.to_string().contains(&second.uuid.to_string()));
    }

    #[rstest]
    #[case::input_dimension(4, 2, DataType::Float32, 2, "input_dimension")]
    #[case::output_dimension(3, 1, DataType::Float32, 2, "output_dimension")]
    #[case::centroid_dtype(3, 2, DataType::Float64, 2, "centroid dtype")]
    #[case::rq_dimension(3, 2, DataType::Float32, 1, "rotated_dim")]
    fn test_pca_shape_validation(
        #[case] input_dimension: usize,
        #[case] ivf_dimension: usize,
        #[case] centroid_type: DataType,
        #[case] rq_rotated_dimension: usize,
        #[case] expected_message: &str,
    ) {
        let model = test_pca_model();
        let error = validate_pca_shape(
            3,
            model.as_ref(),
            input_dimension,
            ivf_dimension,
            &centroid_type,
            rq_rotated_dimension,
        )
        .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains(expected_message), "{error}");
        assert!(error.to_string().contains("actual="), "{error}");
        assert!(error.to_string().contains("expected="), "{error}");
    }
}
