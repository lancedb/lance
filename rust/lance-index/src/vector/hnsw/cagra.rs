// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Runtime cuVS CAGRA integration.
//!
//! cuVS remains an optional runtime dependency. The Python layer supplies the
//! exact `libcuvs_c` path, and this module resolves only the stable C symbols
//! needed to build and copy a CAGRA graph.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr;

use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;

use super::builder::{HNSW, HnswBuildParams};
use crate::vector::sq::storage::ScalarQuantizationStorage;
use crate::vector::storage::VectorStore;

const CUVS_SUCCESS: c_int = 1;
const DL_CPU: c_int = 1;
const DL_UINT: u8 = 1;
const DL_FLOAT: u8 = 2;
const CAGRA_HNSW_SIMILAR_SEARCH_PERFORMANCE: c_int = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: c_int,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Self)>,
}

type CuvsStatus = c_int;
type CuvsResources = usize;
type CuvsDataset = *mut c_void;
type CuvsCagraParams = *mut c_void;
type CuvsCagraIndex = *mut c_void;

type GetLastErrorText = unsafe extern "C" fn() -> *const c_char;
type ResourcesCreate = unsafe extern "C" fn(*mut CuvsResources) -> CuvsStatus;
type ResourcesDestroy = unsafe extern "C" fn(CuvsResources) -> CuvsStatus;
type StreamSync = unsafe extern "C" fn(CuvsResources) -> CuvsStatus;
type MatrixCopy =
    unsafe extern "C" fn(CuvsResources, *mut DLManagedTensor, *mut DLManagedTensor) -> CuvsStatus;
type DatasetMakeStandardView =
    unsafe extern "C" fn(CuvsResources, *mut DLManagedTensor, *mut CuvsDataset) -> CuvsStatus;
type DatasetDestroy = unsafe extern "C" fn(CuvsDataset) -> CuvsStatus;
type ParamsCreate = unsafe extern "C" fn(*mut CuvsCagraParams) -> CuvsStatus;
type ParamsDestroy = unsafe extern "C" fn(CuvsCagraParams) -> CuvsStatus;
type ParamsFromHnsw =
    unsafe extern "C" fn(CuvsCagraParams, i64, i64, c_int, c_int, c_int, c_int) -> CuvsStatus;
type IndexCreate = unsafe extern "C" fn(*mut CuvsCagraIndex) -> CuvsStatus;
type IndexDestroy = unsafe extern "C" fn(CuvsCagraIndex) -> CuvsStatus;
type CagraBuild =
    unsafe extern "C" fn(CuvsResources, CuvsCagraParams, CuvsDataset, CuvsCagraIndex) -> CuvsStatus;
type IndexGetGraph = unsafe extern "C" fn(CuvsCagraIndex, *mut DLManagedTensor) -> CuvsStatus;

#[cfg(unix)]
struct DynamicLibrary(*mut c_void);

#[cfg(unix)]
impl DynamicLibrary {
    fn open(path: &Path) -> Result<Self> {
        use std::os::unix::ffi::OsStrExt;

        let path_display = path.display().to_string();
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            Error::invalid_input(format!(
                "cuVS library path contains an interior NUL byte: {}",
                path_display
            ))
        })?;
        // SAFETY: `path` is a valid NUL-terminated string. The returned handle
        // is retained until every resolved function pointer is no longer used.
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(Error::io(format!(
                "failed to load cuVS library {path_display}: {}",
                dl_error_message()
            )));
        }
        Ok(Self(handle))
    }

    fn symbol<T: Copy>(&self, name: &'static [u8]) -> Result<T> {
        debug_assert_eq!(
            name.last(),
            Some(&0),
            "cuVS dynamic symbol names must be NUL-terminated"
        );
        // SAFETY: the symbol name is NUL-terminated and the library handle is
        // live. Each caller supplies the signature from the cuVS C header.
        let symbol = unsafe { libc::dlsym(self.0, name.as_ptr().cast()) };
        if symbol.is_null() {
            let name = match name.strip_suffix(&[0]) {
                Some(name) => String::from_utf8_lossy(name),
                None => String::from_utf8_lossy(name),
            };
            return Err(Error::io(format!(
                "cuVS library is missing required symbol {name}: {}",
                dl_error_message()
            )));
        }
        // SAFETY: `symbol` was resolved by name for the exact C function type
        // requested at each call site. Function pointers are pointer-sized and
        // `T: Copy`, so copying the representation is valid.
        Ok(unsafe { std::mem::transmute_copy(&symbol) })
    }
}

#[cfg(unix)]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        // SAFETY: this is the live handle returned by `dlopen`, closed once.
        let status = unsafe { libc::dlclose(self.0) };
        if status != 0 {
            log::warn!("failed to unload cuVS library: {}", dl_error_message());
        }
    }
}

#[cfg(unix)]
fn dl_error_message() -> String {
    // SAFETY: `dlerror` returns either NULL or a process-owned NUL-terminated
    // diagnostic string that remains valid until the next loader call.
    let error = unsafe { libc::dlerror() };
    if error.is_null() {
        "unknown dynamic loader error".to_string()
    } else {
        // SAFETY: non-NULL `dlerror` results are valid C strings.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(unix)]
struct CuvsApi {
    _library: DynamicLibrary,
    get_last_error_text: GetLastErrorText,
    resources_create: ResourcesCreate,
    resources_destroy: ResourcesDestroy,
    stream_sync: StreamSync,
    matrix_copy: MatrixCopy,
    dataset_make_standard_view: DatasetMakeStandardView,
    dataset_destroy: DatasetDestroy,
    params_create: ParamsCreate,
    params_destroy: ParamsDestroy,
    params_from_hnsw: ParamsFromHnsw,
    index_create: IndexCreate,
    index_destroy: IndexDestroy,
    cagra_build: CagraBuild,
    index_get_graph: IndexGetGraph,
}

#[cfg(unix)]
impl CuvsApi {
    fn load(path: &Path) -> Result<Self> {
        let library = DynamicLibrary::open(path)?;
        Ok(Self {
            get_last_error_text: library.symbol(b"cuvsGetLastErrorText\0")?,
            resources_create: library.symbol(b"cuvsResourcesCreate\0")?,
            resources_destroy: library.symbol(b"cuvsResourcesDestroy\0")?,
            stream_sync: library.symbol(b"cuvsStreamSync\0")?,
            matrix_copy: library.symbol(b"cuvsMatrixCopy\0")?,
            dataset_make_standard_view: library.symbol(b"cuvsDatasetMakeStandardView\0")?,
            dataset_destroy: library.symbol(b"cuvsDatasetDestroy\0")?,
            params_create: library.symbol(b"cuvsCagraIndexParamsCreate\0")?,
            params_destroy: library.symbol(b"cuvsCagraIndexParamsDestroy\0")?,
            params_from_hnsw: library.symbol(b"cuvsCagraIndexParamsFromHnswParams\0")?,
            index_create: library.symbol(b"cuvsCagraIndexCreate\0")?,
            index_destroy: library.symbol(b"cuvsCagraIndexDestroy\0")?,
            cagra_build: library.symbol(b"cuvsCagraBuild\0")?,
            index_get_graph: library.symbol(b"cuvsCagraIndexGetGraph\0")?,
            _library: library,
        })
    }

    fn check(&self, status: CuvsStatus, operation: &str) -> Result<()> {
        if status == CUVS_SUCCESS {
            return Ok(());
        }
        // SAFETY: the function pointer was resolved with its C header
        // signature and returns either NULL or a NUL-terminated error string.
        let error = unsafe { (self.get_last_error_text)() };
        let detail = if error.is_null() {
            format!("status {status}")
        } else {
            // SAFETY: non-NULL cuVS error text is a valid C string.
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        };
        Err(Error::io(format!("cuVS failed to {operation}: {detail}")))
    }
}

#[cfg(unix)]
struct Resources<'a> {
    api: &'a CuvsApi,
    handle: CuvsResources,
}

#[cfg(unix)]
impl<'a> Resources<'a> {
    fn create(api: &'a CuvsApi) -> Result<Self> {
        let mut handle = 0;
        // SAFETY: the output pointer is valid and the function signature was
        // resolved from the cuVS C API.
        api.check(
            unsafe { (api.resources_create)(&mut handle) },
            "create resources",
        )?;
        Ok(Self { api, handle })
    }
}

#[cfg(unix)]
impl Drop for Resources<'_> {
    fn drop(&mut self) {
        // SAFETY: this handle was created by the matching API and is dropped once.
        let status = unsafe { (self.api.resources_destroy)(self.handle) };
        if let Err(error) = self.api.check(status, "destroy resources") {
            log::warn!("{error}");
        }
    }
}

#[cfg(unix)]
struct DatasetView<'a> {
    api: &'a CuvsApi,
    handle: CuvsDataset,
}

#[cfg(unix)]
impl Drop for DatasetView<'_> {
    fn drop(&mut self) {
        // SAFETY: this handle was created by the matching API and is dropped once.
        let status = unsafe { (self.api.dataset_destroy)(self.handle) };
        if let Err(error) = self.api.check(status, "destroy the dataset view") {
            log::warn!("{error}");
        }
    }
}

#[cfg(unix)]
struct CagraParams<'a> {
    api: &'a CuvsApi,
    handle: CuvsCagraParams,
}

#[cfg(unix)]
impl Drop for CagraParams<'_> {
    fn drop(&mut self) {
        // SAFETY: this handle was created by the matching API and is dropped once.
        let status = unsafe { (self.api.params_destroy)(self.handle) };
        if let Err(error) = self.api.check(status, "destroy CAGRA parameters") {
            log::warn!("{error}");
        }
    }
}

#[cfg(unix)]
struct CagraIndex<'a> {
    api: &'a CuvsApi,
    handle: CuvsCagraIndex,
}

#[cfg(unix)]
impl Drop for CagraIndex<'_> {
    fn drop(&mut self) {
        // SAFETY: this handle was created by the matching API and is dropped once.
        let status = unsafe { (self.api.index_destroy)(self.handle) };
        if let Err(error) = self.api.check(status, "destroy the CAGRA index") {
            log::warn!("{error}");
        }
    }
}

struct ManagedGraph(DLManagedTensor);

impl Drop for ManagedGraph {
    fn drop(&mut self) {
        if let Some(deleter) = self.0.deleter {
            // SAFETY: cuVS installed this deleter for this exact managed tensor;
            // it releases only the view metadata, not the index-owned graph.
            unsafe { deleter(&mut self.0) };
        }
    }
}

fn host_tensor<T>(values: &mut [T], shape: &mut [i64], dtype: DLDataType) -> DLManagedTensor {
    DLManagedTensor {
        dl_tensor: DLTensor {
            data: values.as_mut_ptr().cast(),
            device: DLDevice {
                device_type: DL_CPU,
                device_id: 0,
            },
            ndim: shape.len() as i32,
            dtype,
            shape: shape.as_mut_ptr(),
            strides: ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ptr::null_mut(),
        deleter: None,
    }
}

fn cuvs_distance_type(distance_type: DistanceType) -> Result<c_int> {
    match distance_type {
        DistanceType::L2 => Ok(0),
        DistanceType::Cosine => Ok(2),
        DistanceType::Dot => Ok(6),
        DistanceType::Hamming => Err(Error::invalid_input(
            "CAGRA does not support Lance bitwise Hamming vectors".to_string(),
        )),
    }
}

/// Whether a partition is large enough for cuVS's HNSW-compatible graph.
///
/// The similar-search-performance heuristic derives an intermediate graph
/// degree of `M + M * ef_construction / 256`. A graph needs at least one more
/// row than its degree because a node cannot be its own neighbor. Tiny IVF
/// partitions are cheaper and safer to build directly on the CPU.
pub(super) fn supports_partition(num_rows: usize, params: &HnswBuildParams) -> bool {
    let intermediate_graph_degree = params.m.saturating_add(
        params
            .m
            .saturating_mul(params.ef_construction)
            .saturating_div(256),
    );
    num_rows > intermediate_graph_degree
}

/// Build a one-level Lance HNSW graph with cuVS CAGRA.
pub(super) fn build(
    storage: &impl VectorStore,
    params: HnswBuildParams,
    library_path: &str,
) -> Result<HNSW> {
    #[cfg(not(unix))]
    {
        let _ = (storage, params, library_path);
        return Err(Error::io(
            "CAGRA HNSW acceleration is currently supported only on Unix".to_string(),
        ));
    }

    #[cfg(unix)]
    {
        params.validate()?;
        let sq_storage = storage
            .as_any()
            .downcast_ref::<ScalarQuantizationStorage>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "CAGRA HNSW acceleration currently requires scalar-quantized storage"
                        .to_string(),
                )
            })?;
        let (mut vectors, dim) = sq_storage.to_f32_matrix()?;
        let num_rows = storage.len();
        if num_rows == 0 {
            return Ok(HNSW::empty());
        }
        let num_rows_i64 = i64::try_from(num_rows)
            .map_err(|_| Error::invalid_input("CAGRA row count exceeds i64::MAX".to_string()))?;
        let dim_i64 = i64::try_from(dim)
            .map_err(|_| Error::invalid_input("CAGRA dimension exceeds i64::MAX".to_string()))?;
        let m = c_int::try_from(params.m)
            .map_err(|_| Error::invalid_input("HNSW m exceeds C int range".to_string()))?;
        let ef_construction = c_int::try_from(params.ef_construction).map_err(|_| {
            Error::invalid_input("HNSW ef_construction exceeds C int range".to_string())
        })?;

        let api = CuvsApi::load(Path::new(library_path))?;
        let resources = Resources::create(&api)?;
        let mut vector_shape = [num_rows_i64, dim_i64];
        let mut vector_tensor = host_tensor(
            &mut vectors,
            &mut vector_shape,
            DLDataType {
                code: DL_FLOAT,
                bits: 32,
                lanes: 1,
            },
        );

        let mut dataset_handle = ptr::null_mut();
        // SAFETY: all pointers refer to live stack metadata and initialized
        // row-major host storage for the duration of the call.
        api.check(
            unsafe {
                (api.dataset_make_standard_view)(
                    resources.handle,
                    &mut vector_tensor,
                    &mut dataset_handle,
                )
            },
            "create a dataset view",
        )?;
        let dataset = DatasetView {
            api: &api,
            handle: dataset_handle,
        };

        let mut params_handle = ptr::null_mut();
        // SAFETY: the output pointer is valid for the matching create call.
        api.check(
            unsafe { (api.params_create)(&mut params_handle) },
            "create CAGRA parameters",
        )?;
        let cagra_params = CagraParams {
            api: &api,
            handle: params_handle,
        };
        // SAFETY: handles are live and scalar arguments were range-checked.
        api.check(
            unsafe {
                (api.params_from_hnsw)(
                    cagra_params.handle,
                    num_rows_i64,
                    dim_i64,
                    m,
                    ef_construction,
                    CAGRA_HNSW_SIMILAR_SEARCH_PERFORMANCE,
                    cuvs_distance_type(storage.distance_type())?,
                )
            },
            "derive CAGRA parameters from HNSW parameters",
        )?;

        let mut index_handle = ptr::null_mut();
        // SAFETY: the output pointer is valid for the matching create call.
        api.check(
            unsafe { (api.index_create)(&mut index_handle) },
            "create a CAGRA index",
        )?;
        let index = CagraIndex {
            api: &api,
            handle: index_handle,
        };
        // SAFETY: every handle is live and the borrowed dataset outlives the
        // index build and graph copy.
        api.check(
            unsafe {
                (api.cagra_build)(
                    resources.handle,
                    cagra_params.handle,
                    dataset.handle,
                    index.handle,
                )
            },
            "build a CAGRA graph",
        )?;

        let mut graph_tensor = MaybeUninit::<DLManagedTensor>::zeroed();
        // SAFETY: cuVS initializes the complete managed tensor on success.
        api.check(
            unsafe { (api.index_get_graph)(index.handle, graph_tensor.as_mut_ptr()) },
            "get the CAGRA graph",
        )?;
        // SAFETY: the preceding successful C call initialized the value.
        let mut graph = ManagedGraph(unsafe { graph_tensor.assume_init() });
        let tensor = &graph.0.dl_tensor;
        if tensor.ndim != 2 || tensor.shape.is_null() || tensor.data.is_null() {
            return Err(Error::io(format!(
                "CAGRA returned invalid graph metadata: rank={}, shape_null={}, data_null={}",
                tensor.ndim,
                tensor.shape.is_null(),
                tensor.data.is_null()
            )));
        }
        if tensor.dtype.code != DL_UINT || tensor.dtype.bits != 32 || tensor.dtype.lanes != 1 {
            return Err(Error::io(format!(
                "CAGRA returned graph dtype code={}, bits={}, lanes={}; expected uint32",
                tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes
            )));
        }
        // SAFETY: a rank-2 successful DLPack result has two shape elements.
        let graph_shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
        if graph_shape[0] != num_rows_i64 || graph_shape[1] <= 0 {
            return Err(Error::io(format!(
                "CAGRA returned graph shape {:?}; expected ({num_rows}, degree)",
                graph_shape
            )));
        }
        let graph_degree = usize::try_from(graph_shape[1])
            .map_err(|_| Error::io("CAGRA graph degree exceeds usize::MAX".to_string()))?;
        let graph_len = num_rows
            .checked_mul(graph_degree)
            .ok_or_else(|| Error::io("CAGRA graph size overflow".to_string()))?;
        let mut neighbors = vec![0_u32; graph_len];
        let mut host_shape = [num_rows_i64, graph_shape[1]];
        let mut host_graph = host_tensor(
            &mut neighbors,
            &mut host_shape,
            DLDataType {
                code: DL_UINT,
                bits: 32,
                lanes: 1,
            },
        );
        // SAFETY: source and destination tensors have matching shapes and
        // dtypes, and both buffers remain live through synchronization.
        api.check(
            unsafe { (api.matrix_copy)(resources.handle, &mut graph.0, &mut host_graph) },
            "copy the CAGRA graph to host memory",
        )?;
        // SAFETY: this live resources handle owns the stream used by the copy.
        api.check(
            unsafe { (api.stream_sync)(resources.handle) },
            "synchronize the CAGRA graph copy",
        )?;

        log::info!(
            "Built HNSW graph with cuVS CAGRA: num={}, degree={}",
            num_rows,
            graph_degree
        );
        HNSW::from_neighbor_graph(params, neighbors, graph_degree)
    }
}
