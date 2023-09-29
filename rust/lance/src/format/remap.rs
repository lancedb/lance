use roaring::RoaringTreemap;

/// When a fragment is remapped to a new location the remap vector
/// contains the old row ids.  This can be used to map to the new
/// row ids.
pub struct RemapVector(
    // Currently, a RemapVector is just a RoaringTreemap.  However, we
    // may change this in the future.
    RoaringTreemap,
);

impl RemapVector {}

impl From<RoaringTreemap> for RemapVector {
    fn from(treemap: RoaringTreemap) -> Self {
        Self(treemap)
    }
}

impl From<RemapVector> for RoaringTreemap {
    fn from(remap: RemapVector) -> Self {
        remap.0
    }
}
