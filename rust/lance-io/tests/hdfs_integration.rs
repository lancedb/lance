// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//! These integration tests can only be run against a real GCS bucket.  
//! They do not work against any local emulator right now.
#![cfg(feature = "hdfs-test")]

use std::sync::Arc;

// TODO: Once we re-use this logic for S3, we can instead use tests against
// Minio to validate the multipart upload logic.
use hdfs_native::minidfs::MiniDfs;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use serial_test::serial;
use tokio::io::AsyncWriteExt;

async fn get_store(path: &str) -> Arc<ObjectStore> {
    ObjectStore::from_uri(path).await.unwrap().0
}

#[tokio::test]
#[serial]
async fn test_small_read_write() {
    let _dfs = MiniDfs::default();
    let store = get_store("hdfs://127.0.0.1:9000/small-object").await;

    let path = Path::from("test_small_upload");
    if store.exists(&path).await.unwrap() {
        store.delete(&path).await.unwrap();
    }

    // Write an empty file.
    let mut writer = store.create(&path).await.unwrap();
    writer.shutdown().await.unwrap();
    let meta = store.inner.head(&path).await.unwrap();
    assert_eq!(meta.size, 0);
    store.delete(&path).await.unwrap();

    // Write a small file, with two small writes.
    {
        let mut writer = store.create(&path).await.unwrap();
        writer.write_all(b"hello").await.unwrap();
        writer.write_all(b"world").await.unwrap();
        writer.shutdown().await.unwrap();
    }
    let meta = store.inner.head(&path).await.unwrap();
    assert_eq!(meta.size, 10);

    assert!(store.exists(&path).await.unwrap());
    // test if we can read back the content
    {
        let content = store.inner.get(&path).await.unwrap().bytes().await.unwrap();
        assert_eq!(&content[..], b"helloworld");
    }
    store.delete(&path).await.unwrap();
    assert!(!store.exists(&path).await.unwrap());
}

#[tokio::test]
#[serial]
async fn test_large_read_write() {
    let _dfs = MiniDfs::default();
    let store = get_store("hdfs://127.0.0.1:9000/large-object").await;

    let path = Path::from("test_large_upload");
    if store.exists(&path).await.unwrap() {
        store.delete(&path).await.unwrap();
    }

    let mut writer = store.create(&path).await.unwrap();

    // Write a few 3MB buffers
    writer
        .write_all(&vec![b'a'; 3 * 1024 * 1024])
        .await
        .unwrap();
    writer
        .write_all(&vec![b'b'; 3 * 1024 * 1024])
        .await
        .unwrap();
    writer
        .write_all(&vec![b'c'; 3 * 1024 * 1024])
        .await
        .unwrap();
    writer
        .write_all(&vec![b'd'; 3 * 1024 * 1024])
        .await
        .unwrap();

    // Write a 40MB buffer
    writer
        .write_all(&vec![b'e'; 40 * 1024 * 1024])
        .await
        .unwrap();

    writer.flush().await.unwrap();
    writer.shutdown().await.unwrap();

    let meta = store.inner.head(&path).await.unwrap();
    assert_eq!(meta.size, 52 * 1024 * 1024);

    let data = store.inner.get(&path).await.unwrap().bytes().await.unwrap();
    assert_eq!(data.len(), 52 * 1024 * 1024);
    assert_eq!(data[0], b'a');
    assert_eq!(data[3 * 1024 * 1024], b'b');
    assert_eq!(data[6 * 1024 * 1024], b'c');
    assert_eq!(data[9 * 1024 * 1024], b'd');
    assert_eq!(data[12 * 1024 * 1024], b'e');
}
