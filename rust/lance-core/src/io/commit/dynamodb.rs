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

//! DynamoDB based external manifest store
//!

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use aws_sdk_dynamodb::operation::{
    get_item::builders::GetItemFluentBuilder, put_item::builders::PutItemFluentBuilder,
    query::builders::QueryFluentBuilder,
};
use aws_sdk_dynamodb::types::{AttributeValue, KeyType};
use aws_sdk_dynamodb::Client;
use snafu::OptionExt;
use snafu::{location, Location};
use tokio::sync::RwLock;

use crate::error::{IOSnafu, NotFoundSnafu};
use crate::io::commit::external_manifest::ExternalManifestStore;
use crate::{Error, Result};

// TODO: re-enable after migration is done.
//
// impl<E> From<SdkError<E>> for Error
// where
//     E: std::error::Error + Send + Sync + 'static,
// {
//     fn from(e: SdkError<E>) -> Self {
//         let error_type = format!("{}", e);
//         Self::IO {
//             message: format!(
//                 "dynamodb error: {}, source: {:?}",
//                 error_type,
//                 e.into_source()
//             ),
//             location: location!(),
//         }
//     }
// }

/// An external manifest store backed by DynamoDB
///
/// When calling DynamoDBExternalManifestStore::new_external_store()
/// the key schema, (PK, SK), is checked. If the table does not exist,
/// or the key schema is not as expected, an error is returned.
///
/// The table schema is expected as follows:
/// PK: base_uri -- string
/// SK: version -- number
/// path -- string
/// commiter -- string
///
/// Consistency: This store is expected to have read-after-write consistency
/// consistent_read should always be set to true
///
/// Transaction Safty: This store uses DynamoDB conditional write to ensure
/// only one writer can win per version.
#[derive(Debug)]
pub struct DynamoDBExternalManifestStore {
    client: Arc<Client>,
    table_name: String,
    commiter_name: String,
}

// these are in macro because I want to use them in a match statement
macro_rules! base_uri {
    () => {
        "base_uri"
    };
}
macro_rules! version {
    () => {
        "version"
    };
}
macro_rules! path {
    () => {
        "path"
    };
}
macro_rules! commiter {
    () => {
        "commiter"
    };
}

impl DynamoDBExternalManifestStore {
    pub async fn new_external_store(
        client: Arc<Client>,
        table_name: &str,
        commiter_name: &str,
    ) -> Result<Arc<dyn ExternalManifestStore>> {
        lazy_static::lazy_static! {
            static ref SANITY_CHECK_CACHE: RwLock<HashSet<String>> = RwLock::new(HashSet::new());
        }

        let store = Arc::new(Self {
            client: client.clone(),
            table_name: table_name.to_string(),
            commiter_name: commiter_name.to_string(),
        });

        // already checked this table before, skip
        // this is to avoid checking the table schema every time
        // because it's expensive to call DescribeTable
        if SANITY_CHECK_CACHE.read().await.contains(table_name) {
            return Ok(store);
        }

        // Check if the table schema is correct
        let describe_result = client
            .describe_table()
            .table_name(table_name)
            .send()
            .await
            .map_err(|e| Error::IO {
                message: format!("dynamodb error: {}", e,),
                location: location!(),
            })?;
        let table = describe_result.table.context(IOSnafu {
            message: format!("dynamodb table: {table_name} does not exist"),
        })?;
        let mut schema = table.key_schema.context(IOSnafu {
            message: format!("dynamodb table: {table_name} does not have a key schema"),
        })?;

        let mut has_hask_key = false;
        let mut has_range_key = false;

        // there should be two keys, HASH(base_uri) and RANGE(version)
        for _ in 0..2 {
            let key = schema.pop().context(IOSnafu {
                message: format!("dynamodb table: {table_name} must have HASH and RANGE keys"),
            })?;
            let key_type = key.key_type.context(IOSnafu {
                message: format!("dynamodb table: {table_name} key types must be defined"),
            })?;
            let name = key.attribute_name.context(IOSnafu {
                message: format!("dynamodb table: {table_name} key must have an attribute name"),
            })?;
            match (key_type, name.as_str()) {
                (KeyType::Hash, base_uri!()) => {
                    has_hask_key = true;
                }
                (KeyType::Range, version!()) => {
                    has_range_key = true;
                }
                _ => {
                    return Err(Error::IO {
                        message: format!(
                            "dynamodb table: {table_name} unknown key type encountered name:{name}",
                        ),
                        location: location!(),
                    });
                }
            }
        }

        // Both keys must be present
        if !(has_hask_key && has_range_key) {
            return Err(
                Error::IO {
                    message: format!("dynamodb table: {} must have HASH and RANGE keys, named `{}` and `{}` respectively", table_name, base_uri!(), version!()),
                    location: location!(),
                }
            );
        }

        SANITY_CHECK_CACHE
            .write()
            .await
            .insert(table_name.to_string());

        Ok(store)
    }

    fn ddb_put(&self) -> PutItemFluentBuilder {
        self.client.put_item().table_name(&self.table_name)
    }

    fn ddb_get(&self) -> GetItemFluentBuilder {
        self.client
            .get_item()
            .table_name(&self.table_name)
            .consistent_read(true)
    }

    fn ddb_query(&self) -> QueryFluentBuilder {
        self.client
            .query()
            .table_name(&self.table_name)
            .consistent_read(true)
    }
}

#[async_trait]
impl ExternalManifestStore for DynamoDBExternalManifestStore {
    /// Get the manifest path for a given base_uri and version
    async fn get(&self, base_uri: &str, version: u64) -> Result<String> {
        let get_item_result = self
            .ddb_get()
            .key(base_uri!(), AttributeValue::S(base_uri.into()))
            .key(version!(), AttributeValue::N(version.to_string()))
            .send()
            .await
            .map_err(|e| Error::IO {
                message: format!("dynamodb error: {}", e,),
                location: location!(),
            })?;

        let item = get_item_result.item.context(NotFoundSnafu {
            uri: format!(
                "dynamodb not found: base_uri: {}; version: {}",
                base_uri, version
            ),
        })?;

        let path = item.get(path!()).context(IOSnafu {
            message: format!("key {} is not present", path!()),
        })?;

        match path {
            AttributeValue::S(path) => Ok(path.clone()),
            _ => Err(Error::IO {
                message: format!("key {} is not a string", path!()),
                location: location!(),
            }),
        }
    }

    /// Get the latest version of a dataset at the base_uri
    async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>> {
        let query_result = self
            .ddb_query()
            .key_condition_expression(format!("{} = :{}", base_uri!(), base_uri!()))
            .expression_attribute_values(
                format!(":{}", base_uri!()),
                AttributeValue::S(base_uri.into()),
            )
            .scan_index_forward(false)
            .limit(1)
            .send()
            .await
            .map_err(|e| Error::IO {
                message: format!("dynamodb error: {}", e,),
                location: location!(),
            })?;

        match query_result.items {
            Some(mut items) => {
                if items.is_empty() {
                    return Ok(None);
                }
                if items.len() > 1 {
                    return Err(Error::IO {
                        message: format!(
                            "dynamodb table: {} return unexpect number of items",
                            self.table_name
                        ),
                        location: location!(),
                    });
                }

                let item = items.pop().expect("length checked");
                let version_attibute = item
                .get(version!())
                .context(
                    IOSnafu {
                        message: format!("dynamodb error: found entries for {} but the returned data does not contain {} column", base_uri, version!())
                    }
                )?;

                let path_attribute = item
                .get(path!())
                .context(
                    IOSnafu {
                        message: format!("dynamodb error: found entries for {} but the returned data does not contain {} column", base_uri, path!())
                    }
                )?;

                match (version_attibute, path_attribute) {
                    (AttributeValue::N(version), AttributeValue::S(path)) => Ok(Some((
                        version.parse().map_err(|e| Error::IO {
                            message: format!("dynamodb error: could not parse the version number returned {}, error: {}", version, e),
                            location: location!(),
                        })?,
                        path.clone(),
                    ))),
                    _ => Err(Error::IO {
                        message: format!("dynamodb error: found entries for {base_uri} but the returned data is not number type"),
                        location: location!(),
                    })
                }
            }
            _ => Ok(None),
        }
    }

    /// Put the manifest path for a given base_uri and version, should fail if the version already exists
    async fn put_if_not_exists(&self, base_uri: &str, version: u64, path: &str) -> Result<()> {
        self.ddb_put()
            .item(base_uri!(), AttributeValue::S(base_uri.into()))
            .item(version!(), AttributeValue::N(version.to_string()))
            .item(path!(), AttributeValue::S(path.to_string()))
            .item(commiter!(), AttributeValue::S(self.commiter_name.clone()))
            .condition_expression(format!(
                "attribute_not_exists({}) AND attribute_not_exists({})",
                base_uri!(),
                version!(),
            ))
            .send()
            .await
            .map_err(|e| Error::IO {
                message: format!("dynamodb error: {}", e,),
                location: location!(),
            })?;

        Ok(())
    }

    /// Put the manifest path for a given base_uri and version, should fail if the version **does not** already exist
    async fn put_if_exists(&self, base_uri: &str, version: u64, path: &str) -> Result<()> {
        self.ddb_put()
            .item(base_uri!(), AttributeValue::S(base_uri.into()))
            .item(version!(), AttributeValue::N(version.to_string()))
            .item(path!(), AttributeValue::S(path.to_string()))
            .item(commiter!(), AttributeValue::S(self.commiter_name.clone()))
            .condition_expression(format!(
                "attribute_exists({}) AND attribute_exists({})",
                base_uri!(),
                version!(),
            ))
            .send()
            .await
            .map_err(|e| Error::IO {
                message: format!("dynamodb error: {}", e,),
                location: location!(),
            })?;

        Ok(())
    }
}

// TODO: these tests are copied from super::external_manifest::test
// since these tests applies to all external manifest stores,
// we should move them to a common place
// https://github.com/lancedb/lance/issues/1208
//
// The tests are linux only because
// GHA Mac runner doesn't have docker, which is required to run dynamodb-local
// Windows FS can't handle concurrent copy
#[cfg(all(test, target_os = "linux", feature = "dynamodb_tests"))]
mod test {
    use aws_credential_types::Credentials;
    use aws_sdk_dynamodb::{
        config::Region,
        types::{
            AttributeDefinition, KeySchemaElement, ProvisionedThroughput, ScalarAttributeType,
        },
    };
    use futures::{future::join_all, StreamExt, TryStreamExt};
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32};
    use object_store::local::LocalFileSystem;

    use crate::{
        dataset::{ReadParams, WriteMode, WriteParams},
        io::{
            commit::{
                external_manifest::ExternalManifestCommitHandler, latest_manifest_path,
                manifest_path, CommitHandler,
            },
            object_store::ObjectStoreParams,
        },
        Dataset,
    };

    use super::*;

    fn read_params(handler: Arc<dyn CommitHandler>) -> ReadParams {
        ReadParams {
            store_options: Some(ObjectStoreParams {
                commit_handler: Some(handler),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn write_params(handler: Arc<dyn CommitHandler>) -> WriteParams {
        WriteParams {
            store_params: Some(ObjectStoreParams {
                commit_handler: Some(handler),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    async fn make_dynamodb_store() -> Arc<dyn ExternalManifestStore> {
        let dynamodb_local_config = aws_sdk_dynamodb::config::Builder::new()
            .endpoint_url(
                // url for dynamodb-local
                "http://localhost:8000",
            )
            .region(Some(Region::new("us-east-1")))
            .credentials_provider(Credentials::new("DUMMYKEY", "DUMMYKEY", None, None, ""))
            .build();

        let table_name = uuid::Uuid::new_v4().to_string();

        let client = Client::from_conf(dynamodb_local_config);
        client
            .create_table()
            .table_name(&table_name)
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name(base_uri!())
                    .key_type(KeyType::Hash)
                    .build(),
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name(version!())
                    .key_type(KeyType::Range)
                    .build(),
            )
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name(base_uri!())
                    .attribute_type(ScalarAttributeType::S)
                    .build(),
            )
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name(version!())
                    .attribute_type(ScalarAttributeType::N)
                    .build(),
            )
            .provisioned_throughput(
                ProvisionedThroughput::builder()
                    .read_capacity_units(10)
                    .write_capacity_units(10)
                    .build(),
            )
            .send()
            .await
            .unwrap();
        DynamoDBExternalManifestStore::new_external_store(Arc::new(client), &table_name, "test")
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_store() {
        // test basic behavior of the store
        let store = make_dynamodb_store().await;
        // DNE should return None for latest
        assert_eq!(store.get_latest_version("test").await.unwrap(), None);
        // DNE should return Err for get specific version
        assert!(store
            .get("test", 1)
            .await
            .unwrap_err()
            .to_string()
            .starts_with("Not found: dynamodb not found: base_uri: test; version: 1"));
        // try to use the API for finalizing should return err when the version is DNE
        assert!(store.put_if_exists("test", 1, "test").await.is_err());

        // Put a new version should work
        assert!(store
            .put_if_not_exists("test", 1, "test.unfinalized")
            .await
            .is_ok());
        // put again should get err
        assert!(store
            .put_if_not_exists("test", 1, "test.unfinalized_1")
            .await
            .is_err());

        // Can get that new version back and is the latest
        assert_eq!(
            store.get_latest_version("test").await.unwrap(),
            Some((1, "test.unfinalized".to_string()))
        );
        assert_eq!(store.get("test", 1).await.unwrap(), "test.unfinalized");

        // Put a new version should work again
        assert!(store
            .put_if_not_exists("test", 2, "test.unfinalized_2")
            .await
            .is_ok());
        // latest should see update
        assert_eq!(
            store.get_latest_version("test").await.unwrap(),
            Some((2, "test.unfinalized_2".to_string()))
        );

        // try to finalize should work on existing version
        assert!(store.put_if_exists("test", 2, "test").await.is_ok());

        // latest should see update
        assert_eq!(
            store.get_latest_version("test").await.unwrap(),
            Some((2, "test".to_string()))
        );
        // get should see new data
        assert_eq!(store.get("test", 2).await.unwrap(), "test");
    }

    #[tokio::test]
    async fn test_dataset_can_onboard_external_store() {
        // First write a dataset WITHOUT external store
        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let reader = data_gen.batch(100);
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();
        Dataset::write(reader, ds_uri, None).await.unwrap();

        // Then try to load the dataset with external store handler set
        let store = make_dynamodb_store().await;
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: store,
        };
        let options = read_params(Arc::new(handler));
        Dataset::open_with_params(ds_uri, &options).await.expect(
            "If this fails, it means the external store handler does not correctly handle the case when a dataset exist, but it has never used external store before."
        );
    }

    #[tokio::test]
    async fn test_can_create_dataset_with_external_store() {
        let store = make_dynamodb_store().await;
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: store,
        };
        let handler = Arc::new(handler);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let reader = data_gen.batch(100);
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();
        Dataset::write(reader, ds_uri, Some(write_params(handler.clone())))
            .await
            .unwrap();

        // load the data and check the content
        let ds = Dataset::open_with_params(ds_uri, &read_params(handler))
            .await
            .unwrap();
        assert_eq!(ds.count_rows().await.unwrap(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_commits_are_okay() {
        let store = make_dynamodb_store().await;
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: store,
        };
        let handler = Arc::new(handler);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();

        Dataset::write(
            data_gen.batch(10),
            ds_uri,
            Some(write_params(handler.clone())),
        )
        .await
        .unwrap();

        // we have 5 retries by default, more than this will just fail
        let write_futs = (0..5)
            .map(|_| data_gen.batch(10))
            .map(|data| {
                let mut params = write_params(handler.clone());
                params.mode = WriteMode::Append;
                Dataset::write(data, ds_uri, Some(params))
            })
            .collect::<Vec<_>>();

        let res = join_all(write_futs).await;

        let errors = res
            .into_iter()
            .filter(|r| r.is_err())
            .map(|r| r.unwrap_err())
            .collect::<Vec<_>>();

        assert!(errors.is_empty(), "{:?}", errors);

        // load the data and check the content
        let ds = Dataset::open_with_params(ds_uri, &read_params(handler))
            .await
            .unwrap();
        assert_eq!(ds.count_rows().await.unwrap(), 60);
    }

    #[tokio::test]
    async fn test_out_of_sync_dataset_can_recover() {
        let store = make_dynamodb_store().await;
        let handler = ExternalManifestCommitHandler {
            external_manifest_store: store.clone(),
        };
        let handler = Arc::new(handler);

        let mut data_gen =
            BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("x".to_owned())));
        let dir = tempfile::tempdir().unwrap();
        let ds_uri = dir.path().to_str().unwrap();

        let mut ds = Dataset::write(
            data_gen.batch(10),
            ds_uri,
            Some(write_params(handler.clone())),
        )
        .await
        .unwrap();

        for _ in 0..5 {
            let data = data_gen.batch(10);
            let mut params = write_params(handler.clone());
            params.mode = WriteMode::Append;
            ds = Dataset::write(data, ds_uri, Some(params)).await.unwrap();
        }

        // manually simulate last version is out of sync
        let localfs: Box<dyn object_store::ObjectStore> = Box::new(LocalFileSystem::new());
        localfs.delete(&manifest_path(&ds.base, 6)).await.unwrap();
        localfs
            .copy(&manifest_path(&ds.base, 5), &latest_manifest_path(&ds.base))
            .await
            .unwrap();
        // set the store back to dataset path with -{uuid} suffix
        let mut version_six = localfs
            .list(Some(&ds.base))
            .await
            .unwrap()
            .try_filter(|p| {
                let p = p.clone();
                async move { p.location.filename().unwrap().starts_with("6.manifest-") }
            })
            .collect::<Vec<_>>()
            .await;
        assert_eq!(version_six.len(), 1);
        let version_six_staging_location = version_six.pop().unwrap().unwrap().location;
        store
            .put_if_exists(ds.base.as_ref(), 6, version_six_staging_location.as_ref())
            .await
            .unwrap();

        // Open without external store handler, should not see the out-of-sync commit
        let params = ReadParams::default();
        let ds = Dataset::open_with_params(ds_uri, &params).await.unwrap();
        assert_eq!(ds.version().version, 5);
        assert_eq!(ds.count_rows().await.unwrap(), 50);

        // Open with external store handler, should sync the out-of-sync commit on open
        let ds = Dataset::open_with_params(ds_uri, &read_params(handler))
            .await
            .unwrap();
        assert_eq!(ds.version().version, 6);
        assert_eq!(ds.count_rows().await.unwrap(), 60);

        // Open without external store handler again, should see the newly sync'd commit
        let params = ReadParams::default();
        let ds = Dataset::open_with_params(ds_uri, &params).await.unwrap();
        assert_eq!(ds.version().version, 6);
        assert_eq!(ds.count_rows().await.unwrap(), 60);
    }
}
