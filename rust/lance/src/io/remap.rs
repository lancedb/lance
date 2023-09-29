use bytes::Buf;
use object_store::path::Path;
use roaring::RoaringTreemap;
use snafu::ResultExt;
use uuid::Uuid;

use super::ObjectStore;
use crate::dataset::REMAP_DIR;
use crate::error::{box_error, CorruptFileSnafu};
use crate::format::remap::RemapVector;
use crate::Result;

/// Get the file path for a deletion file. This is relative to the dataset root.
pub fn remap_file_path(base: &Path, uuid: &Uuid) -> Path {
    base.child(REMAP_DIR).child(format!("{uuid}.remap"))
}

pub async fn write_remap_vector(
    base: &Path,
    remap: RemapVector,
    object_store: &ObjectStore,
) -> Result<Uuid> {
    let uuid = Uuid::new_v4();
    let file_path = remap_file_path(base, &uuid);
    let mut out: Vec<u8> = Vec::new();
    RoaringTreemap::from(remap).serialize_into(&mut out)?;
    object_store.inner.put(&file_path, out.into()).await?;
    Ok(uuid)
}

#[allow(dead_code)]
pub async fn read_remap_vector(
    base: &Path,
    uuid: &Uuid,
    object_store: &ObjectStore,
) -> Result<RemapVector> {
    let path = remap_file_path(base, uuid);
    let data = object_store.inner.get(&path).await?.bytes().await?;
    let reader = data.reader();
    let bitmap = RoaringTreemap::deserialize_from(reader)
        .map_err(box_error)
        .context(CorruptFileSnafu { path })?;

    Ok(bitmap.into())
}
