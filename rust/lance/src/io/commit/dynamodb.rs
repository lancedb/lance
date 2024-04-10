// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Keep the tests in `lance` crate because it has dependency on [Dataset].
//
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
    use std::sync::Arc;

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
