// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Extend [object_store::ObjectStore] functionalities

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::{future, stream::BoxStream, StreamExt, TryStreamExt};
use object_store::{path::Path, ObjectMeta, ObjectStore};

use crate::Result;

#[async_trait]
pub trait ObjectStoreExt {
    /// Returns true if the file exists.
    async fn exists(&self, path: &Path) -> Result<bool>;

    /// Read all files (start from base directory) recursively
    ///
    /// unmodified_since can be specified to only return files that have not been modified since the given time.
    async fn read_dir_all(
        &self,
        dir_path: impl Into<&Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> Result<BoxStream<Result<ObjectMeta>>>;
}

#[async_trait]
impl<O: ObjectStore + ?Sized> ObjectStoreExt for O {
    async fn read_dir_all(
        &self,
        dir_path: impl Into<&Path> + Send,
        unmodified_since: Option<DateTime<Utc>>,
    ) -> Result<BoxStream<Result<ObjectMeta>>> {
        let mut output = self.list(Some(dir_path.into())).await?;
        if let Some(unmodified_since_val) = unmodified_since {
            output = output
                .try_filter(move |file| future::ready(file.last_modified < unmodified_since_val))
                .boxed();
        }
        Ok(output.map_err(|e| e.into()).boxed())
    }

    async fn exists(&self, path: &Path) -> Result<bool> {
        match self.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { path: _, source: _ }) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}
