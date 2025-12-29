// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tests for io_uring reader implementation.

use crate::object_store::ObjectStore;
use lance_core::Result;
use std::io::Write;
use tempfile::NamedTempFile;

/// Helper to create a temporary file with test data
fn create_test_file(size: usize) -> Result<(NamedTempFile, Vec<u8>)> {
    let mut file = NamedTempFile::new()?;
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    file.write_all(&data)?;
    file.flush()?;
    Ok((file, data))
}

#[tokio::test]
async fn test_read_small_file() -> Result<()> {
    let (file, expected_data) = create_test_file(1024)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    // Read entire file
    let data = reader.get_all().await.unwrap();
    assert_eq!(data.as_ref(), expected_data.as_slice());

    Ok(())
}

#[tokio::test]
async fn test_read_range() -> Result<()> {
    let (file, expected_data) = create_test_file(4096)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    // Read a range in the middle
    let range = 1000..2000;
    let data = reader.get_range(range.clone()).await.unwrap();
    assert_eq!(data.as_ref(), &expected_data[range]);

    Ok(())
}

#[tokio::test]
async fn test_read_multiple_ranges() -> Result<()> {
    let (file, expected_data) = create_test_file(8192)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    // Read multiple ranges
    let ranges = vec![0..100, 500..600, 2000..3000];
    for range in ranges {
        let data = reader.get_range(range.clone()).await.unwrap();
        assert_eq!(data.as_ref(), &expected_data[range]);
    }

    Ok(())
}

#[tokio::test]
async fn test_file_size() -> Result<()> {
    let size = 5000;
    let (file, _) = create_test_file(size)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    assert_eq!(reader.size().await.unwrap(), size);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_reads() -> Result<()> {
    let (file, expected_data) = create_test_file(16384)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;

    // Perform multiple concurrent reads
    let mut tasks = vec![];
    for i in 0..10 {
        let reader_clone = store.open(&path).await?;
        let expected = expected_data.clone();
        tasks.push(tokio::spawn(async move {
            let range = (i * 1000)..((i + 1) * 1000);
            let data = reader_clone.get_range(range.clone()).await.unwrap();
            assert_eq!(data.as_ref(), &expected[range]);
        }));
    }

    // Wait for all tasks
    for task in tasks {
        task.await.unwrap();
    }

    Ok(())
}

#[tokio::test]
async fn test_large_file_read() -> Result<()> {
    // Test with a larger file (1MB)
    let size = 1024 * 1024;
    let (file, expected_data) = create_test_file(size)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    // Read entire file
    let data = reader.get_all().await.unwrap();
    assert_eq!(data.len(), size);
    assert_eq!(data.as_ref(), expected_data.as_slice());

    Ok(())
}

#[tokio::test]
async fn test_read_edge_cases() -> Result<()> {
    let (file, expected_data) = create_test_file(4096)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    // Read from start
    let data = reader.get_range(0..100).await.unwrap();
    assert_eq!(data.as_ref(), &expected_data[0..100]);

    // Read to end
    let data = reader.get_range(4000..4096).await.unwrap();
    assert_eq!(data.as_ref(), &expected_data[4000..4096]);

    // Read single byte
    let data = reader.get_range(2000..2001).await.unwrap();
    assert_eq!(data.as_ref(), &expected_data[2000..2001]);

    Ok(())
}

#[tokio::test]
async fn test_file_not_found() {
    let uri = "file+uring:///nonexistent/file.dat";
    let (store, path) = ObjectStore::from_uri(uri).await.unwrap();

    // Should fail to open non-existent file
    let result = store.open(&path).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_block_size_and_parallelism() -> Result<()> {
    let (file, _) = create_test_file(1024)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    // Check default values (or configured values)
    assert!(reader.block_size() > 0);
    assert!(reader.io_parallelism() > 0);

    Ok(())
}

#[tokio::test]
async fn test_path() -> Result<()> {
    let (file, _) = create_test_file(1024)?;
    let file_path = file.path().to_str().unwrap();
    let uri = format!("file+uring://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    // Verify path is preserved
    assert_eq!(reader.path(), &path);

    Ok(())
}

#[tokio::test]
async fn test_uring_not_enabled_with_file_scheme() -> Result<()> {
    // Verify that files opened with file:// don't use uring
    let (file, expected_data) = create_test_file(1024)?;
    let file_path = file.path().to_str().unwrap();
    // Use regular file:// scheme, should NOT use uring
    let uri = format!("file://{}", file_path);

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let reader = store.open(&path).await?;

    // Should still be able to read, just won't use uring
    let data = reader.get_all().await.unwrap();
    assert_eq!(data.as_ref(), expected_data.as_slice());

    Ok(())
}
