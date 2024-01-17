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
use aws_sdk_dynamodb::error::SdkError;
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

impl<E> From<SdkError<E>> for Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn from(e: SdkError<E>) -> Self {
        let error_type = format!("{}", e);
        Self::IO {
            message: format!(
                "dynamodb error: {}, source: {:?}",
                error_type,
                e.into_source()
            ),
            location: location!(),
        }
    }
}

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
            .await?;
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
            .await?;

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
            .await?;

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
            .await?;

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
            .await?;

        Ok(())
    }
}
