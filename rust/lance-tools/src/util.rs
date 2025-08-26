// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::Result;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use std::sync::Arc;

/// Get an object store and a path from a source string.
pub(crate) async fn get_object_store_and_path(source: &String) -> Result<(Arc<ObjectStore>, Path)> {
    let path = Path::parse(source)?;
    return Ok((Arc::new(ObjectStore::local()), path));
}
