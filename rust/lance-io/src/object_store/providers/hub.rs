// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::{Arc, LazyLock},
};

use futures::{StreamExt, future, stream, stream::BoxStream};
use lance_core::error::{Error, Result};
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore as OSObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, Result as OSResult, path::Path,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::fs;
use url::Url;

use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, ObjectStoreRegistry, StorageOptions,
    StorageOptionsAccessor, uri_to_url,
};

// Currently we only allow 255 hub shards at most, to make the simple_mod calculation quicker.
const MAX_HUB_SHARDS: usize = 255;

// The scheme that the hub object store uses.
const HUB_SCHEME: &str = "hub";

// The environment variable that stores the configuration file path.
const HUB_STORE_CONFIG_ENV: &str = "HUB_STORE_CONFIG";
const HUB_STORE_CONFIG_OPTION: &str = "hub_store_config";

// The full name of the hub object store.
const HUB_OBJECT_STORE_NAME: &str = "HubObjectStore";

static SHARD_PROVIDER_REGISTRY: LazyLock<ObjectStoreRegistry> =
    LazyLock::new(ObjectStoreRegistry::default);

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ReadPolicy {
    SimpleMod,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ReadPolicyConfig {
    policy: ReadPolicy,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum WritePolicy {
    SimpleMod,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct WritePolicyConfig {
    policy: WritePolicy,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HubShardConfig {
    uri: String,
    #[serde(default)]
    storage_options: HashMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HubBucketConfig {
    shards: Vec<HubShardConfig>,
    read: ReadPolicyConfig,
    write: WritePolicyConfig,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HubObjectStoreConfig {
    buckets: BTreeMap<String, HubBucketConfig>,
}

#[derive(Clone, Debug)]
struct HubShard {
    inner: ObjectStore,
}

#[derive(Debug)]
struct HubObjectStore {
    bucket_name: String,
    shards: Vec<HubShard>,
    write_policy: WritePolicy,
    read_policy: ReadPolicy,
}

impl fmt::Display for HubObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{HUB_OBJECT_STORE_NAME}({})", self.bucket_name)
    }
}

fn simple_mod_shard_index(location: &Path, num_shards: usize) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(location.as_ref().as_bytes());
    let digest = hasher.finalize();
    (digest[digest.len() - 1] as usize) % num_shards
}

fn parse_hub_config(
    config_contents: &str,
    config_path: &str,
    bucket_name: &str,
) -> Result<BTreeMap<String, HubBucketConfig>> {
    serde_json::from_str::<HubObjectStoreConfig>(config_contents)
        .map(|config| config.buckets)
        .map_err(|error| {
            Error::invalid_input(format!(
                "failed to parse JSON in hub store config file '{}' for hub bucket '{}': {}",
                config_path, bucket_name, error
            ))
        })
}

impl HubObjectStore {
    fn new(
        bucket_name: String,
        shards: Vec<HubShard>,
        write_policy: WritePolicy,
        read_policy: ReadPolicy,
    ) -> OSResult<Self> {
        if bucket_name.is_empty() {
            return Err(object_store::Error::Generic {
                store: HUB_OBJECT_STORE_NAME,
                source: "hub bucket name must not be empty".into(),
            });
        }
        if shards.is_empty() {
            return Err(object_store::Error::Generic {
                store: HUB_OBJECT_STORE_NAME,
                source: "hub object store must configure at least 1 shard".into(),
            });
        }
        if shards.len() > MAX_HUB_SHARDS {
            return Err(object_store::Error::Generic {
                store: HUB_OBJECT_STORE_NAME,
                source: format!(
                    "hub object store cannot configure more than {} shards",
                    MAX_HUB_SHARDS
                )
                .into(),
            });
        }
        Ok(Self {
            bucket_name,
            shards,
            write_policy,
            read_policy,
        })
    }

    fn index_of_read_shard(&self, location: &Path) -> usize {
        match self.read_policy {
            ReadPolicy::SimpleMod => simple_mod_shard_index(location, self.shards.len()),
        }
    }

    fn index_of_write_shard(&self, location: &Path) -> usize {
        match self.write_policy {
            WritePolicy::SimpleMod => simple_mod_shard_index(location, self.shards.len()),
        }
    }
}

#[async_trait::async_trait]
impl OSObjectStore for HubObjectStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.shards[self.index_of_write_shard(location)]
            .inner
            .inner
            .put_opts(location, payload, opts)
            .await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.shards[self.index_of_write_shard(location)]
            .inner
            .inner
            .put_multipart_opts(location, opts)
            .await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.shards[self.index_of_read_shard(location)]
            .inner
            .inner
            .get_opts(location, options)
            .await
    }

    async fn delete(&self, location: &Path) -> OSResult<()> {
        self.shards[self.index_of_write_shard(location)]
            .inner
            .inner
            .delete(location)
            .await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        let prefix = prefix.cloned();
        stream::iter(
            self.shards
                .clone()
                .into_iter()
                .map(move |shard| shard.inner.inner.list(prefix.as_ref())),
        )
        .flatten()
        .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        let prefix = prefix.cloned();
        let shard_results = future::try_join_all(
            self.shards
                .iter()
                .map(|shard| shard.inner.inner.list_with_delimiter(prefix.as_ref())),
        )
        .await?;

        let mut common_prefixes = Vec::new();
        let mut objects = Vec::new();

        for shard_result in shard_results {
            for common_prefix in shard_result.common_prefixes {
                if !common_prefixes.contains(&common_prefix) {
                    common_prefixes.push(common_prefix);
                }
            }
            for object in shard_result.objects {
                if !objects
                    .iter()
                    .any(|existing: &ObjectMeta| existing.location == object.location)
                {
                    objects.push(object);
                }
            }
        }

        Ok(ListResult {
            common_prefixes,
            objects,
        })
    }

    async fn copy(&self, from: &Path, to: &Path) -> OSResult<()> {
        Err(object_store::Error::NotSupported {
            source: format!("hub does not support copy from '{}' to '{}'", from, to).into(),
        })
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()> {
        Err(object_store::Error::NotSupported {
            source: format!(
                "hub does not support copy_if_not_exists from '{}' to '{}'",
                from, to
            )
            .into(),
        })
    }
}

#[derive(Default, Debug)]
pub struct HubStoreProvider;

fn hub_bucket_name(url: &Url) -> Result<&str> {
    url.host_str().ok_or_else(|| {
        Error::invalid_input(format!(
            "hub url '{}' must include a hub bucket in the authority component",
            url
        ))
    })
}

fn resolve_hub_store_config_path(
    storage_options: Option<&HashMap<String, String>>,
    env_config_path: Option<String>,
) -> Result<String> {
    if let Some(config_path) =
        storage_options.and_then(|options| options.get(HUB_STORE_CONFIG_OPTION))
    {
        if config_path.is_empty() {
            return Err(Error::invalid_input(format!(
                "storage option '{}' must not be empty when using {}:// URIs",
                HUB_STORE_CONFIG_OPTION, HUB_SCHEME
            )));
        }
        return Ok(config_path.clone());
    }

    let config_path = env_config_path.ok_or_else(|| {
        Error::invalid_input(format!(
            "storage option '{}' or environment variable {} must be set before using {}:// URIs",
            HUB_STORE_CONFIG_OPTION, HUB_STORE_CONFIG_ENV, HUB_SCHEME
        ))
    })?;
    if config_path.is_empty() {
        return Err(Error::invalid_input(format!(
            "environment variable {} must not be empty when using {}:// URIs",
            HUB_STORE_CONFIG_ENV, HUB_SCHEME
        )));
    }
    Ok(config_path)
}

fn hub_store_config_path(storage_options: Option<&HashMap<String, String>>) -> Result<String> {
    resolve_hub_store_config_path(storage_options, std::env::var(HUB_STORE_CONFIG_ENV).ok())
}

#[async_trait::async_trait]
impl ObjectStoreProvider for HubStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let bucket_name = hub_bucket_name(&base_path)?.to_string();
        let config_path = hub_store_config_path(params.storage_options())?;
        let config_contents = fs::read_to_string(&config_path).await.map_err(|error| {
            Error::io(format!(
                "failed to read hub store config file '{}' for hub bucket '{}': {}",
                config_path, bucket_name, error
            ))
        })?;
        let mut config = parse_hub_config(&config_contents, &config_path, &bucket_name)?;

        let available_buckets = if config.is_empty() {
            "<none>".to_string()
        } else {
            config.keys().cloned().collect::<Vec<_>>().join(", ")
        };
        let bucket_config = config.remove(&bucket_name).ok_or_else(|| {
            Error::invalid_input(format!(
                "hub bucket '{}' was not found in hub store config file '{}'. Available hub buckets: {}",
                bucket_name, config_path, available_buckets
            ))
        })?;
        let HubBucketConfig {
            shards: shard_configs,
            read,
            write,
        } = bucket_config;

        let mut shards = Vec::with_capacity(shard_configs.len());
        for shard_config in shard_configs {
            let shard_uri = shard_config.uri;
            let shard_url = uri_to_url(&shard_uri).map_err(|error| {
                Error::invalid_input(format!(
                    "failed to parse shard uri '{}' for hub bucket '{}': {}",
                    shard_uri, bucket_name, error
                ))
            })?;
            if shard_url.scheme() == HUB_SCHEME {
                return Err(Error::invalid_input(format!(
                    "hub bucket '{}' contains shard uri '{}' with scheme '{}', which is not allowed",
                    bucket_name,
                    shard_uri,
                    shard_url.scheme()
                )));
            }

            let provider = SHARD_PROVIDER_REGISTRY
                .get_provider(shard_url.scheme())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "no object store provider found for shard uri '{}' in hub bucket '{}'. Scheme '{}' is not registered",
                        shard_uri,
                        bucket_name,
                        shard_url.scheme()
                    ))
                })?;

            let shard_params = ObjectStoreParams {
                storage_options_accessor: (!shard_config.storage_options.is_empty()).then(|| {
                    Arc::new(StorageOptionsAccessor::with_static_options(
                        shard_config.storage_options,
                    ))
                }),
                ..Default::default()
            };

            shards.push(HubShard {
                inner: provider.new_store(shard_url, &shard_params).await?,
            });
        }

        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let max_iop_size = shards
            .iter()
            .map(|shard| shard.inner.max_iop_size())
            .min()
            .unwrap_or(*DEFAULT_MAX_IOP_SIZE);
        let use_constant_size_upload_parts = params.use_constant_size_upload_parts
            || shards
                .iter()
                .any(|shard| shard.inner.use_constant_size_upload_parts);
        let io_parallelism = shards
            .iter()
            .map(|shard| shard.inner.io_parallelism())
            .min()
            .unwrap_or(DEFAULT_CLOUD_IO_PARALLELISM);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());
        let download_retry_count = storage_options.download_retry_count();

        Ok(ObjectStore {
            inner: Arc::new(HubObjectStore::new(
                bucket_name,
                shards,
                write.policy,
                read.policy,
            )?),
            scheme: HUB_SCHEME.to_string(),
            block_size,
            max_iop_size,
            use_constant_size_upload_parts,
            list_is_lexically_ordered: false,
            io_parallelism,
            download_retry_count,
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
        })
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        let bucket_name = hub_bucket_name(url)?;
        let config_path = hub_store_config_path(storage_options)?;
        Ok(format!("{HUB_SCHEME}${bucket_name}@{config_path}"))
    }
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;
    use object_store::ObjectStore as _;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_hub_object_store_validates_inputs() {
        let shard = ObjectStore::memory();
        let err = HubObjectStore::new(
            String::new(),
            vec![HubShard {
                inner: shard.clone(),
            }],
            WritePolicy::SimpleMod,
            ReadPolicy::SimpleMod,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("bucket name must not be empty"),
            "unexpected error: {}",
            err
        );

        let err = HubObjectStore::new(
            "bucket".to_string(),
            Vec::new(),
            WritePolicy::SimpleMod,
            ReadPolicy::SimpleMod,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("at least 1 shard"),
            "unexpected error: {}",
            err
        );

        let too_many_shards = vec![HubShard { inner: shard }; MAX_HUB_SHARDS + 1];
        let err = HubObjectStore::new(
            "bucket".to_string(),
            too_many_shards,
            WritePolicy::SimpleMod,
            ReadPolicy::SimpleMod,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains(&MAX_HUB_SHARDS.to_string()),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_parse_hub_config_requires_nested_buckets_key() {
        let nested = json!({
            "buckets": {
                "bucket-a": {
                    "shards": [{"uri": "memory://"}],
                    "read": {"policy": "simple_mod"},
                    "write": {"policy": "simple_mod"}
                }
            }
        });

        let buckets = parse_hub_config(&nested.to_string(), "/tmp/hub.json", "bucket-a").unwrap();

        assert!(buckets.contains_key("bucket-a"));
    }

    #[test]
    fn test_parse_hub_config_rejects_flat_layout() {
        let flat = json!({
            "bucket-a": {
                "shards": [{"uri": "memory://"}],
                "read": {"policy": "simple_mod"},
                "write": {"policy": "simple_mod"}
            }
        });

        let err = parse_hub_config(&flat.to_string(), "/tmp/hub.json", "bucket-a").unwrap_err();
        assert!(
            err.to_string().contains("failed to parse JSON"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_resolve_hub_store_config_path_prefers_storage_option() {
        let config_path = resolve_hub_store_config_path(
            Some(&HashMap::from([(
                HUB_STORE_CONFIG_OPTION.to_string(),
                "/tmp/from-storage.json".to_string(),
            )])),
            Some("/tmp/from-env.json".to_string()),
        )
        .unwrap();

        assert_eq!(config_path, "/tmp/from-storage.json");
    }

    #[test]
    fn test_resolve_hub_store_config_path_requires_config_source() {
        let err = resolve_hub_store_config_path(None, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("storage option 'hub_store_config' or environment variable HUB_STORE_CONFIG must be set"),
            "unexpected error: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_hub_routes_put_to_expected_shard() {
        let shard_a = ObjectStore::memory();
        let shard_b = ObjectStore::memory();
        let hub = HubObjectStore::new(
            "bucket".to_string(),
            vec![
                HubShard {
                    inner: shard_a.clone(),
                },
                HubShard {
                    inner: shard_b.clone(),
                },
            ],
            WritePolicy::SimpleMod,
            ReadPolicy::SimpleMod,
        )
        .unwrap();

        let location = Path::from("dataset/file.lance");
        let payload = PutPayload::from_static(b"hello");
        hub.put(&location, payload).await.unwrap();

        let expected_index = simple_mod_shard_index(&location, 2);
        let shards = [shard_a, shard_b];

        for (index, shard) in shards.into_iter().enumerate() {
            let result = shard.inner.get(&location).await;
            if index == expected_index {
                let bytes = result.unwrap().bytes().await.unwrap();
                assert_eq!(bytes.as_ref(), b"hello");
            } else {
                let is_not_found = matches!(&result, Err(object_store::Error::NotFound { .. }));
                assert!(
                    is_not_found,
                    "expected shard {} to miss object, got {:?}",
                    index, result
                );
            }
        }
    }

    #[tokio::test]
    async fn test_hub_list_merges_shards() {
        let shard_a = ObjectStore::memory();
        let shard_b = ObjectStore::memory();

        shard_a
            .inner
            .put(&Path::from("a/file1"), PutPayload::from_static(b"a"))
            .await
            .unwrap();
        shard_b
            .inner
            .put(&Path::from("b/file2"), PutPayload::from_static(b"b"))
            .await
            .unwrap();

        let hub = HubObjectStore::new(
            "bucket".to_string(),
            vec![HubShard { inner: shard_a }, HubShard { inner: shard_b }],
            WritePolicy::SimpleMod,
            ReadPolicy::SimpleMod,
        )
        .unwrap();

        let mut locations = hub
            .list(None)
            .map_ok(|meta| meta.location)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        locations.sort_by(|left, right| left.as_ref().cmp(right.as_ref()));

        assert_eq!(
            locations,
            vec![Path::from("a/file1"), Path::from("b/file2")]
        );
    }

    #[tokio::test]
    async fn test_hub_store_provider_builds_from_bucket_config() {
        let config_file = tempfile::NamedTempFile::new().unwrap();
        let config_path = config_file.path().to_str().unwrap().to_string();
        let config = json!({
            "buckets": {
                "bucket-a": {
                    "shards": [
                        {"uri": "memory://"},
                        {"uri": "memory://"}
                    ],
                    "read": {"policy": "simple_mod"},
                    "write": {"policy": "simple_mod"}
                }
            }
        });
        std::fs::write(config_file.path(), config.to_string()).unwrap();

        let provider = HubStoreProvider;
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                HashMap::from([(HUB_STORE_CONFIG_OPTION.to_string(), config_path.clone())]),
            ))),
            ..Default::default()
        };
        let store = provider
            .new_store(
                Url::parse("hub://bucket-a/datasets/example.lance").unwrap(),
                &params,
            )
            .await
            .unwrap();

        assert_eq!(store.scheme, HUB_SCHEME);
        assert_eq!(store.block_size, DEFAULT_CLOUD_BLOCK_SIZE);
        assert_eq!(
            store.store_prefix,
            format!("{HUB_SCHEME}$bucket-a@{config_path}")
        );

        let location = Path::from("datasets/example.lance");
        store
            .inner
            .put(&location, PutPayload::from_static(b"hello"))
            .await
            .unwrap();
        let bytes = store
            .inner
            .get(&location)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), b"hello");
    }
}
