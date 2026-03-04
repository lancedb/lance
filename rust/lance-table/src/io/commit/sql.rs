// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SQL-based external manifest store
//!
//! Uses sqlx with the `Any` driver to support PostgreSQL, MySQL, and SQLite.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use object_store::path::Path;
use sqlx::any::{AnyPoolOptions, AnyRow};
use sqlx::{AnyPool, Row};
use tokio::sync::RwLock;
use tracing::warn;

use crate::io::commit::external_manifest::ExternalManifestStore;
use lance_core::error::box_error;
use lance_core::{Error, Result};

use super::ManifestLocation;
use super::external_manifest::detect_naming_scheme_from_path;

/// Connection pool configuration for the SQL external manifest store.
#[derive(Debug, Clone)]
pub struct SqlPoolConfig {
    /// Maximum number of connections in the pool. Default: 5.
    pub max_connections: u32,
    /// Minimum number of idle connections to maintain. Default: 1.
    pub min_connections: u32,
    /// Maximum time to wait for a connection from the pool. Default: 10s.
    pub acquire_timeout: Duration,
    /// Maximum idle time for a connection before it is closed. Default: 300s.
    pub idle_timeout: Duration,
    /// Maximum lifetime of a connection. Default: 1800s (30 min).
    pub max_lifetime: Duration,
}

impl Default for SqlPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 5,
            min_connections: 1,
            acquire_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(1800),
        }
    }
}

/// An external manifest store backed by a SQL database (via sqlx).
///
/// When calling SqlExternalManifestStore::new_external_store()
/// the table schema is checked. If the table does not exist,
/// or the required columns are not present, an error is returned.
///
/// The table schema is expected as follows:
/// bucket_name    -- VARCHAR(255) NOT NULL  (part of PRIMARY KEY)
/// base_path      -- VARCHAR(255) NOT NULL  (part of PRIMARY KEY)
/// version        -- BIGINT UNSIGNED NOT NULL  (part of PRIMARY KEY)
/// manifest_uuid  -- VARCHAR(255) NOT NULL  (staging manifest uuid suffix)
/// file_size      -- BIGINT UNSIGNED NOT NULL
/// e_tag          -- VARCHAR(255) NOT NULL
/// create_time    -- BIGINT UNSIGNED NOT NULL
/// update_time    -- BIGINT UNSIGNED NOT NULL
/// PRIMARY KEY (bucket_name, base_path, version)
///
/// Example DDL for MySQL:
/// ```sql
/// CREATE TABLE IF NOT EXISTS external_manifest (
///     bucket_name   VARCHAR(255)    NOT NULL COMMENT 'object store bucket',
///     base_path     VARCHAR(255)    NOT NULL COMMENT 'dataset path',
///     version       BIGINT UNSIGNED NOT NULL COMMENT 'version number',
///     manifest_uuid VARCHAR(255)    NOT NULL COMMENT 'staging manifest uuid',
///     file_size     BIGINT UNSIGNED NOT NULL COMMENT 'file size',
///     e_tag         VARCHAR(255)    NOT NULL COMMENT 'ETag checksum',
///     create_time   BIGINT UNSIGNED NOT NULL COMMENT 'creation time',
///     update_time   BIGINT UNSIGNED NOT NULL COMMENT 'update time',
///     PRIMARY KEY (bucket_name, base_path, version)
/// ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
///   COMMENT='external manifest';
/// ```
///
/// Consistency: This store relies on the database's transaction semantics
/// for read-after-write consistency.
///
/// Transaction Safety: Uses INSERT with primary key conflict detection to ensure
/// only one writer can win per version.
#[derive(Debug)]
pub struct SqlExternalManifestStore {
    pool: AnyPool,
    table_name: String,
    bucket_name: String,
}

/// Required columns that must exist in the table.
const REQUIRED_COLUMNS: &[&str] = &[
    "bucket_name",
    "base_path",
    "version",
    "manifest_uuid",
    "file_size",
    "e_tag",
    "create_time",
    "update_time",
];

/// The identifier between the manifest base name and the uuid suffix.
const MANIFEST_UUID_SEPARATOR: &str = ".manifest-";

/// Validate that a SQL identifier (table name) contains only safe characters.
///
/// Accepts only ASCII letters, digits, and underscores (`[a-zA-Z0-9_]`).
/// This prevents SQL injection via crafted table names.
fn validate_sql_identifier(name: &str, param_name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::invalid_input(format!(
            "`{}` must not be empty",
            param_name
        )));
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(Error::invalid_input(format!(
            "`{}` contains invalid characters: '{}'. \
             Only ASCII letters, digits, and underscores are allowed.",
            param_name, name
        )));
    }
    Ok(())
}

impl SqlExternalManifestStore {
    /// Create a new SQL-backed external manifest store.
    ///
    /// `database_url` should be a valid sqlx connection string, e.g.:
    /// - `sqlite://lance_manifest.db`
    /// - `postgres://user:pass@host/db`
    /// - `mysql://user:pass@host/db`
    ///
    /// `bucket_name` is the object store bucket name, stored as part of the
    /// primary key to support multi-bucket scenarios.
    ///
    /// `pool_config` controls connection pool sizing and timeouts.
    ///
    /// The table must already exist with the expected schema.
    /// An error is returned if the table does not exist or is missing required columns.
    pub async fn new_external_store(
        database_url: &str,
        table_name: &str,
        bucket_name: &str,
        pool_config: SqlPoolConfig,
    ) -> Result<Arc<dyn ExternalManifestStore>> {
        // Validate table_name to prevent SQL injection
        validate_sql_identifier(table_name, "sqlTableName")?;

        // Global pool cache keyed by database_url to reuse connections
        static POOL_CACHE: std::sync::LazyLock<RwLock<HashMap<String, AnyPool>>> =
            std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

        // Cache already-checked tables to avoid repeated schema validation
        static SANITY_CHECK_CACHE: std::sync::LazyLock<RwLock<HashSet<String>>> =
            std::sync::LazyLock::new(|| RwLock::new(HashSet::new()));

        sqlx::any::install_default_drivers();

        // Reuse existing pool or create a new one.
        // Fast path: check with a read lock first.
        let pool = {
            let cache = POOL_CACHE.read().await;
            cache.get(database_url).cloned()
        };
        let pool = match pool {
            Some(p) => p,
            None => {
                // Slow path: acquire write lock and double-check to avoid TOCTOU race.
                let mut cache = POOL_CACHE.write().await;
                if let Some(p) = cache.get(database_url) {
                    p.clone()
                } else {
                    let p = AnyPoolOptions::new()
                        .max_connections(pool_config.max_connections)
                        .min_connections(pool_config.min_connections)
                        .acquire_timeout(pool_config.acquire_timeout)
                        .idle_timeout(pool_config.idle_timeout)
                        .max_lifetime(pool_config.max_lifetime)
                        .connect(database_url)
                        .await
                        .map_err(|e| Error::io_source(box_error(e)))?;
                    cache.insert(database_url.to_string(), p.clone());
                    p
                }
            }
        };

        let store = Arc::new(Self {
            pool,
            table_name: table_name.to_string(),
            bucket_name: bucket_name.to_string(),
        });

        let cache_key = format!("{}:{}", database_url, table_name);
        // already checked this table before, skip
        if SANITY_CHECK_CACHE.read().await.contains(&cache_key) {
            return Ok(store);
        }

        // Verify the table exists and has the required columns by issuing
        // a lightweight query. This works across all SQL backends without
        // needing backend-specific information_schema queries.
        let check_sql = format!(
            "SELECT {} FROM {} WHERE 1 = 0",
            REQUIRED_COLUMNS.join(", "),
            table_name
        );
        sqlx::query(&check_sql)
            .execute(&store.pool)
            .await
            .map_err(|e| {
                Error::io(format!(
                    "sql table '{}' does not exist or is missing required columns ({}): {}",
                    table_name,
                    REQUIRED_COLUMNS.join(", "),
                    e
                ))
            })?;

        SANITY_CHECK_CACHE.write().await.insert(cache_key);

        Ok(store)
    }

    /// Build the full manifest path from base_path, version, and manifest_uuid.
    ///
    /// Path rule: `{base_path}/_versions/{u64::MAX - version}.manifest[-uuid]`
    fn build_manifest_path(base_path: &str, version: u64, manifest_uuid: &str) -> String {
        let mut path = format!("{}/_versions/{}.manifest", base_path, u64::MAX - version);
        if !manifest_uuid.is_empty() {
            path.push('-');
            path.push_str(manifest_uuid);
        }
        path
    }

    /// Extract the uuid suffix from a manifest path.
    ///
    /// If the path contains `.manifest-{uuid}`, returns the uuid part.
    /// Otherwise, returns an empty string.
    fn extract_uuid_from_path(manifest_path: &str) -> String {
        if let Some(pos) = manifest_path.find(MANIFEST_UUID_SEPARATOR) {
            manifest_path[pos + MANIFEST_UUID_SEPARATOR.len()..].to_string()
        } else {
            String::new()
        }
    }

    fn now_timestamp() -> i64 {
        chrono::Utc::now().timestamp()
    }
}

fn sql_err(e: sqlx::Error) -> Error {
    warn!(target: "lance::sql", "SQL error: {e:?}");
    Error::io_source(box_error(e))
}

/// Extract a string column from a row.
fn get_string(row: &AnyRow, col: &str) -> Result<String> {
    row.try_get::<String, _>(col)
        .map_err(|e| Error::io_source(box_error(e)))
}

/// Extract an i64 column and convert to u64.
fn get_u64(row: &AnyRow, col: &str) -> Result<u64> {
    row.try_get::<i64, _>(col)
        .map(|v| v as u64)
        .map_err(|e| Error::io_source(box_error(e)))
}

/// Extract an optional string column from a row.
fn get_optional_string(row: &AnyRow, col: &str) -> Option<String> {
    row.try_get::<String, _>(col).ok().filter(|s| !s.is_empty())
}

#[async_trait]
impl ExternalManifestStore for SqlExternalManifestStore {
    /// Get the manifest path for a given base_uri and version
    async fn get(&self, base_uri: &str, version: u64) -> Result<String> {
        let sql = format!(
            "SELECT manifest_uuid FROM {} WHERE bucket_name = ? AND base_path = ? AND version = ?",
            self.table_name
        );
        let row: AnyRow = sqlx::query(&sql)
            .bind(&self.bucket_name)
            .bind(base_uri)
            .bind(version as i64)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| match e {
                sqlx::Error::RowNotFound => Error::not_found(format!(
                    "sql not found: bucket: {}; base_path: {}; version: {}",
                    self.bucket_name, base_uri, version
                )),
                other => sql_err(other),
            })?;

        let manifest_uuid = get_string(&row, "manifest_uuid")?;
        Ok(Self::build_manifest_path(base_uri, version, &manifest_uuid))
    }

    async fn get_manifest_location(
        &self,
        base_uri: &str,
        version: u64,
    ) -> Result<ManifestLocation> {
        let sql = format!(
            "SELECT manifest_uuid, file_size, e_tag FROM {} \
             WHERE bucket_name = ? AND base_path = ? AND version = ?",
            self.table_name
        );
        let row: AnyRow = sqlx::query(&sql)
            .bind(&self.bucket_name)
            .bind(base_uri)
            .bind(version as i64)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| match e {
                sqlx::Error::RowNotFound => Error::not_found(format!(
                    "sql not found: bucket: {}; base_path: {}; version: {}",
                    self.bucket_name, base_uri, version
                )),
                other => sql_err(other),
            })?;

        let manifest_uuid = get_string(&row, "manifest_uuid")?;
        let full_path = Self::build_manifest_path(base_uri, version, &manifest_uuid);
        let path = Path::from(full_path);
        let size = Some(get_u64(&row, "file_size")?);
        let e_tag = get_optional_string(&row, "e_tag");
        let naming_scheme = detect_naming_scheme_from_path(&path)?;

        Ok(ManifestLocation {
            version,
            path,
            size,
            naming_scheme,
            e_tag,
        })
    }

    /// Get the latest version of a dataset at the base_uri
    async fn get_latest_version(&self, base_uri: &str) -> Result<Option<(u64, String)>> {
        self.get_latest_manifest_location(base_uri)
            .await
            .map(|location| location.map(|loc| (loc.version, loc.path.to_string())))
    }

    async fn get_latest_manifest_location(
        &self,
        base_uri: &str,
    ) -> Result<Option<ManifestLocation>> {
        let sql = format!(
            "SELECT version, manifest_uuid, file_size, e_tag FROM {} \
             WHERE bucket_name = ? AND base_path = ? \
             ORDER BY version DESC LIMIT 1",
            self.table_name
        );
        let maybe_row: Option<AnyRow> = sqlx::query(&sql)
            .bind(&self.bucket_name)
            .bind(base_uri)
            .fetch_optional(&self.pool)
            .await
            .map_err(sql_err)?;

        match maybe_row {
            Some(row) => {
                let version = get_u64(&row, "version")?;
                let manifest_uuid = get_string(&row, "manifest_uuid")?;
                let full_path = Self::build_manifest_path(base_uri, version, &manifest_uuid);
                let path = Path::from(full_path);
                let size = Some(get_u64(&row, "file_size")?);
                let e_tag = get_optional_string(&row, "e_tag");
                let naming_scheme = detect_naming_scheme_from_path(&path)?;

                Ok(Some(ManifestLocation {
                    version,
                    path,
                    size,
                    naming_scheme,
                    e_tag,
                }))
            }
            None => Ok(None),
        }
    }

    /// Put the manifest path for a given base_uri and version.
    /// Fails if the version already exists.
    ///
    /// Uses a transaction with a single INSERT — the PRIMARY KEY constraint
    /// atomically rejects duplicates. No need to SELECT first.
    async fn put_if_not_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        e_tag: Option<String>,
    ) -> Result<()> {
        let manifest_uuid = Self::extract_uuid_from_path(path);
        let now = Self::now_timestamp();

        let mut tx = self.pool.begin().await.map_err(sql_err)?;

        let sql = format!(
            "INSERT INTO {} (bucket_name, base_path, version, manifest_uuid, file_size, e_tag, create_time, update_time) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            self.table_name
        );
        let res = sqlx::query(&sql)
            .bind(&self.bucket_name)
            .bind(base_uri)
            .bind(version as i64)
            .bind(&manifest_uuid)
            .bind(size as i64)
            .bind(e_tag.as_deref().unwrap_or(""))
            .bind(now)
            .bind(now)
            .execute(&mut *tx)
            .await;

        match res {
            Ok(_) => {
                tx.commit().await.map_err(sql_err)?;
                Ok(())
            }
            Err(sqlx::Error::Database(db_err)) if db_err.is_unique_violation() => {
                tx.rollback().await.map_err(sql_err)?;
                Err(Error::io(format!(
                    "put_if_not_exists: version {} already exists for bucket: {}; base_path: {}",
                    version, self.bucket_name, base_uri
                )))
            }
            Err(e) => {
                tx.rollback().await.map_err(sql_err)?;
                Err(sql_err(e))
            }
        }
    }

    /// Put the manifest path for a given base_uri and version.
    /// Fails if the version does NOT already exist.
    ///
    /// Uses a transaction with a single UPDATE WHERE — `rows_affected() == 0`
    /// means the row doesn't exist. No need to SELECT first.
    async fn put_if_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        e_tag: Option<String>,
    ) -> Result<()> {
        let manifest_uuid = Self::extract_uuid_from_path(path);
        let now = Self::now_timestamp();

        let mut tx = self.pool.begin().await.map_err(sql_err)?;

        let sql = format!(
            "UPDATE {} SET manifest_uuid = ?, file_size = ?, e_tag = ?, update_time = ? \
             WHERE bucket_name = ? AND base_path = ? AND version = ?",
            self.table_name
        );
        let result = sqlx::query(&sql)
            .bind(&manifest_uuid)
            .bind(size as i64)
            .bind(e_tag.as_deref().unwrap_or(""))
            .bind(now)
            .bind(&self.bucket_name)
            .bind(base_uri)
            .bind(version as i64)
            .execute(&mut *tx)
            .await
            .map_err(sql_err)?;

        if result.rows_affected() == 0 {
            tx.rollback().await.map_err(sql_err)?;
            return Err(Error::not_found(format!(
                "put_if_exists: version {} not found for bucket: {}; base_path: {}",
                version, self.bucket_name, base_uri
            )));
        }

        tx.commit().await.map_err(sql_err)?;
        Ok(())
    }

    /// Delete all manifest information for the given base_uri
    async fn delete(&self, base_uri: &str) -> Result<()> {
        let sql = format!(
            "DELETE FROM {} WHERE bucket_name = ? AND base_path = ?",
            self.table_name
        );
        sqlx::query(&sql)
            .bind(&self.bucket_name)
            .bind(base_uri)
            .execute(&self.pool)
            .await
            .map_err(sql_err)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TABLE: &str = "external_manifest";
    const TEST_BUCKET: &str = "test_bucket";
    const TEST_BASE_URI: &str = "datasets/my_table";

    /// Helper: create a fresh in-memory SQLite store with the required schema.
    async fn create_test_store() -> Arc<dyn ExternalManifestStore> {
        // Use a unique in-memory database for each test to avoid shared state.
        // SQLite `:memory:` with a shared cache name keeps the DB alive across
        // connections in the same pool.
        let db_name = uuid::Uuid::new_v4().to_string().replace('-', "");
        let url = format!("sqlite:file:{}?mode=memory&cache=shared", db_name);

        sqlx::any::install_default_drivers();

        let pool = AnyPoolOptions::new()
            .max_connections(2)
            .connect(&url)
            .await
            .expect("failed to connect to in-memory SQLite");

        // Create the table with the expected schema.
        sqlx::query(&format!(
            "CREATE TABLE IF NOT EXISTS {} (
                bucket_name   TEXT NOT NULL,
                base_path     TEXT NOT NULL,
                version       INTEGER NOT NULL,
                manifest_uuid TEXT NOT NULL,
                file_size     INTEGER NOT NULL,
                e_tag         TEXT NOT NULL,
                create_time   INTEGER NOT NULL,
                update_time   INTEGER NOT NULL,
                PRIMARY KEY (bucket_name, base_path, version)
            )",
            TEST_TABLE
        ))
        .execute(&pool)
        .await
        .expect("failed to create test table");

        Arc::new(SqlExternalManifestStore {
            pool,
            table_name: TEST_TABLE.to_string(),
            bucket_name: TEST_BUCKET.to_string(),
        })
    }

    #[tokio::test]
    async fn test_put_if_not_exists_succeeds() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/18446744073709551614.manifest-abc123";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 1024, Some("etag1".into()))
            .await
            .expect("first put_if_not_exists should succeed");
    }

    #[tokio::test]
    async fn test_put_if_not_exists_duplicate_fails() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/18446744073709551614.manifest-abc123";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 1024, Some("etag1".into()))
            .await
            .expect("first insert should succeed");

        let result = store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 2048, Some("etag2".into()))
            .await;
        assert!(result.is_err(), "duplicate version should fail");
    }

    #[tokio::test]
    async fn test_put_if_exists_succeeds() {
        let store = create_test_store().await;
        let path1 = "datasets/my_table/_versions/18446744073709551614.manifest-staging1";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path1, 1024, Some("etag1".into()))
            .await
            .unwrap();

        let path2 = "datasets/my_table/_versions/18446744073709551614.manifest-final1";
        store
            .put_if_exists(TEST_BASE_URI, 1, path2, 2048, Some("etag2".into()))
            .await
            .expect("put_if_exists on existing version should succeed");

        // Verify the update took effect
        let loc = store.get_manifest_location(TEST_BASE_URI, 1).await.unwrap();
        assert_eq!(loc.size, Some(2048));
        assert_eq!(loc.e_tag, Some("etag2".to_string()));
    }

    #[tokio::test]
    async fn test_put_if_exists_missing_fails() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/18446744073709551614.manifest-abc";
        let result = store
            .put_if_exists(TEST_BASE_URI, 1, path, 1024, Some("etag1".into()))
            .await;
        assert!(
            result.is_err(),
            "put_if_exists on non-existent version should fail"
        );
    }

    #[tokio::test]
    async fn test_get_roundtrip() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/18446744073709551614.manifest-uuid1";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 512, Some("etag_a".into()))
            .await
            .unwrap();

        let got_path = store.get(TEST_BASE_URI, 1).await.unwrap();
        assert!(got_path.contains("uuid1"), "path should contain the uuid");
    }

    #[tokio::test]
    async fn test_get_latest_version() {
        let store = create_test_store().await;
        // Insert v1 and v3 (skip v2)
        let p1 = "datasets/my_table/_versions/18446744073709551614.manifest-a";
        let p3 = "datasets/my_table/_versions/18446744073709551612.manifest-c";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, p1, 100, None)
            .await
            .unwrap();
        store
            .put_if_not_exists(TEST_BASE_URI, 3, p3, 300, None)
            .await
            .unwrap();

        let latest = store
            .get_latest_version(TEST_BASE_URI)
            .await
            .unwrap()
            .expect("should have a latest version");
        assert_eq!(latest.0, 3);
    }

    #[tokio::test]
    async fn test_get_manifest_location_roundtrip() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/18446744073709551614.manifest-someuuid";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 4096, Some("my_etag".into()))
            .await
            .unwrap();

        let loc = store.get_manifest_location(TEST_BASE_URI, 1).await.unwrap();
        assert_eq!(loc.version, 1);
        assert_eq!(loc.size, Some(4096));
        assert_eq!(loc.e_tag, Some("my_etag".to_string()));
        assert!(loc.path.as_ref().contains("someuuid"));
    }

    #[tokio::test]
    async fn test_delete_clears_all_versions() {
        let store = create_test_store().await;
        let p1 = "datasets/my_table/_versions/18446744073709551614.manifest-a";
        let p2 = "datasets/my_table/_versions/18446744073709551613.manifest-b";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, p1, 100, None)
            .await
            .unwrap();
        store
            .put_if_not_exists(TEST_BASE_URI, 2, p2, 200, None)
            .await
            .unwrap();

        store.delete(TEST_BASE_URI).await.unwrap();

        let latest = store.get_latest_version(TEST_BASE_URI).await.unwrap();
        assert!(latest.is_none(), "after delete, no versions should remain");
    }

    #[test]
    fn test_validate_sql_identifier_rejects_injection() {
        assert!(validate_sql_identifier("good_table_1", "sqlTableName").is_ok());
        assert!(validate_sql_identifier("ExternalManifest", "sqlTableName").is_ok());

        // SQL injection attempts
        assert!(validate_sql_identifier("t; DROP TABLE t--", "sqlTableName").is_err());
        assert!(validate_sql_identifier("table name", "sqlTableName").is_err());
        assert!(validate_sql_identifier("table'name", "sqlTableName").is_err());
        assert!(validate_sql_identifier("", "sqlTableName").is_err());
        assert!(validate_sql_identifier("table.name", "sqlTableName").is_err());
    }
}
