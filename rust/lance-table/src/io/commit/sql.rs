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

fn placeholder(style: ParamStyle, n: usize) -> String {
    match style {
        ParamStyle::Numbered => format!("${n}"),
        ParamStyle::Positional => "?".to_string(),
    }
}

fn placeholders(style: ParamStyle, count: usize) -> String {
    (1..=count)
        .map(|i| placeholder(style, i))
        .collect::<Vec<_>>()
        .join(", ")
}

// Parameter placeholder style used when rendering SQL statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamStyle {
    /// `$1`, `$2`, … (PostgreSQL).
    Numbered,
    /// `?` (MySQL/MariaDB/SQLite).
    Positional,
}

impl ParamStyle {
    fn from_url(url: &str) -> Self {
        let lower = url.to_ascii_lowercase();
        if lower.starts_with("postgres:") || lower.starts_with("postgresql:") {
            Self::Numbered
        } else {
            Self::Positional
        }
    }
}

#[derive(Debug, Clone)]
pub struct SqlPoolConfig {
    pub max_connections: u32,
    pub min_connections: u32,
    pub acquire_timeout: Duration,
    pub idle_timeout: Duration,
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

#[derive(Debug, Clone)]
pub struct SqlStoreParams {
    /// sqlx connection string, e.g. `sqlite://lance.db`,
    /// `postgres://user:pass@host/db`, or `mysql://user:pass@host/db`.
    pub conn_str: String,
    /// Table name that holds external manifest records.
    pub table_name: String,
    pub bucket_name: String,
    pub pool_config: SqlPoolConfig,
}

/// An external manifest store backed by a SQL database (via sqlx).
///
/// When calling [`SqlExternalManifestStore::new_external_store`] the table
/// schema is checked. The table must already exist with the expected
/// columns and a uniqueness constraint over `(bucket_name, base_path,
/// version)`; otherwise an error is returned.
///
/// The expected schema is:
///
/// | column          | type                    |
/// |-----------------|-------------------------|
/// | `bucket_name`   | `VARCHAR(255) NOT NULL` |
/// | `base_path`     | `VARCHAR(255) NOT NULL` |
/// | `version`       | `BIGINT NOT NULL`       |
/// | `manifest_path` | `VARCHAR(512) NOT NULL` |
/// | `file_size`     | `BIGINT NOT NULL`       |
/// | `e_tag`         | `VARCHAR(255) NOT NULL` |
/// | `create_time`   | `BIGINT NOT NULL`       |
/// | `update_time`   | `BIGINT NOT NULL`       |
#[derive(Debug)]
pub struct SqlExternalManifestStore {
    pool: AnyPool,
    table_name: String,
    bucket_name: String,
    param_style: ParamStyle,
}

/// Required columns that must exist in the table.
const REQUIRED_COLUMNS: &[&str] = &[
    "bucket_name",
    "base_path",
    "version",
    "manifest_path",
    "file_size",
    "e_tag",
    "create_time",
    "update_time",
];

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
    pub async fn new_external_store(
        params: SqlStoreParams,
    ) -> Result<Arc<dyn ExternalManifestStore>> {
        let SqlStoreParams {
            conn_str,
            table_name,
            bucket_name,
            pool_config,
        } = params;
        validate_sql_identifier(&table_name, "sqlTableName")?;

        // Global pool cache keyed by database_url to reuse connections
        static POOL_CACHE: std::sync::LazyLock<RwLock<HashMap<String, AnyPool>>> =
            std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));
        static SANITY_CHECK_CACHE: std::sync::LazyLock<RwLock<HashSet<String>>> =
            std::sync::LazyLock::new(|| RwLock::new(HashSet::new()));

        sqlx::any::install_default_drivers();

        let pool = {
            let cache = POOL_CACHE.read().await;
            cache.get(&conn_str).cloned()
        };
        let pool = match pool {
            Some(p) => p,
            None => {
                let mut cache = POOL_CACHE.write().await;
                if let Some(p) = cache.get(&conn_str) {
                    p.clone()
                } else {
                    let p = AnyPoolOptions::new()
                        .max_connections(pool_config.max_connections)
                        .min_connections(pool_config.min_connections)
                        .acquire_timeout(pool_config.acquire_timeout)
                        .idle_timeout(pool_config.idle_timeout)
                        .max_lifetime(pool_config.max_lifetime)
                        .connect(&conn_str)
                        .await
                        .map_err(|e| Error::io(format!("failed to connect to sql store: {e}")))?;
                    cache.insert(conn_str.clone(), p.clone());
                    p
                }
            }
        };

        let store = Arc::new(Self {
            pool,
            table_name: table_name.clone(),
            bucket_name,
            param_style: ParamStyle::from_url(&conn_str),
        });

        let sanity_key = format!("{conn_str}|{table_name}");
        if SANITY_CHECK_CACHE.read().await.contains(&sanity_key) {
            return Ok(store);
        }

        store.sanity_check().await?;
        SANITY_CHECK_CACHE.write().await.insert(sanity_key);

        Ok(store)
    }

    /// Verify the table has the required columns and a uniqueness constraint
    /// over `(bucket_name, base_path, version)`.
    async fn sanity_check(&self) -> Result<()> {
        let check_sql = format!(
            "SELECT {} FROM {} WHERE 1 = 0",
            REQUIRED_COLUMNS.join(", "),
            self.table_name
        );
        sqlx::query(&check_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                Error::io(format!(
                    "sql table '{}' does not exist or is missing required columns ({}): {}",
                    self.table_name,
                    REQUIRED_COLUMNS.join(", "),
                    e
                ))
            })?;

        self.probe_uniqueness().await
    }

    async fn probe_uniqueness(&self) -> Result<()> {
        const SENTINEL: &str = "__lance_sql_probe__";
        const V1: i64 = i64::MAX;
        const V2: i64 = i64::MAX - 1;

        let insert_sql = format!(
            "INSERT INTO {} (bucket_name, base_path, version, manifest_path, \
             file_size, e_tag, create_time, update_time) VALUES ({})",
            self.table_name,
            placeholders(self.param_style, 8),
        );
        let cleanup_sql = format!(
            "DELETE FROM {} WHERE bucket_name = {} AND base_path = {}",
            self.table_name,
            placeholder(self.param_style, 1),
            placeholder(self.param_style, 2),
        );

        let mut tx = self.pool.begin().await.map_err(sql_err)?;

        // Clean up any leftover probe rows from a previous crashed run.
        sqlx::query(&cleanup_sql)
            .bind(SENTINEL)
            .bind(SENTINEL)
            .execute(&mut *tx)
            .await
            .map_err(sql_err)?;

        let result = self
            .run_uniqueness_probe(&mut tx, &insert_sql, SENTINEL, V1, V2)
            .await;
        let _ = tx.rollback().await;
        result
    }

    async fn run_uniqueness_probe(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Any>,
        insert_sql: &str,
        sentinel: &str,
        v1: i64,
        v2: i64,
    ) -> Result<()> {
        Self::insert_probe_row(tx, insert_sql, sentinel, v1)
            .await
            .map_err(|e| self.probe_io_err("insert v1", e))?;

        if let Err(e) = Self::insert_probe_row(tx, insert_sql, sentinel, v2).await {
            return Err(match e {
                sqlx::Error::Database(db) if db.is_unique_violation() => {
                    Error::invalid_input(format!(
                        "table '{}' has a UNIQUE constraint that does not include `version`; \
                         inserting two rows with the same (bucket_name, base_path) but different \
                         versions was rejected. The constraint must cover (bucket_name, base_path, version) \
                         exactly, otherwise compare-and-swap semantics break.",
                        self.table_name
                    ))
                }
                other => self.probe_io_err("insert v2", other),
            });
        }

        match Self::insert_probe_row(tx, insert_sql, sentinel, v1).await {
            Err(sqlx::Error::Database(db)) if db.is_unique_violation() => Ok(()),
            Ok(_) => Err(Error::invalid_input(format!(
                "table '{}' is missing a UNIQUE/PRIMARY KEY constraint over \
                 (bucket_name, base_path, version); duplicate inserts were accepted, \
                 which would break compare-and-swap semantics",
                self.table_name
            ))),
            Err(e) => Err(self.probe_io_err("duplicate insert", e)),
        }
    }

    fn probe_io_err(&self, stage: &str, e: sqlx::Error) -> Error {
        Error::io(format!(
            "schema sanity probe failed ({}) for '{}': {}",
            stage, self.table_name, e
        ))
    }

    /// Insert a fixed-shape probe row used only by `probe_uniqueness`.
    /// All non-key columns are filled with empty / zero placeholders.
    async fn insert_probe_row(
        tx: &mut sqlx::Transaction<'_, sqlx::Any>,
        sql: &str,
        sentinel: &str,
        version: i64,
    ) -> std::result::Result<sqlx::any::AnyQueryResult, sqlx::Error> {
        sqlx::query(sql)
            .bind(sentinel)
            .bind(sentinel)
            .bind(version)
            .bind("")
            .bind(0_i64)
            .bind("")
            .bind(0_i64)
            .bind(0_i64)
            .execute(&mut **tx)
            .await
    }
}

fn sql_err(e: sqlx::Error) -> Error {
    warn!(target: "lance::sql", "SQL error: {e}");
    Error::io_source(box_error(e))
}

fn get_string(row: &AnyRow, col: &str) -> Result<String> {
    row.try_get::<String, _>(col)
        .map_err(|e| Error::io_source(box_error(e)))
}

fn get_u64(row: &AnyRow, col: &str) -> Result<u64> {
    row.try_get::<i64, _>(col)
        .map(|v| v as u64)
        .map_err(|e| Error::io_source(box_error(e)))
}

fn get_optional_string(row: &AnyRow, col: &str) -> Option<String> {
    row.try_get::<String, _>(col).ok().filter(|s| !s.is_empty())
}

#[async_trait]
impl ExternalManifestStore for SqlExternalManifestStore {
    async fn get(&self, base_uri: &str, version: u64) -> Result<String> {
        let sql = format!(
            "SELECT manifest_path FROM {} WHERE bucket_name = {} AND base_path = {} AND version = {}",
            self.table_name,
            placeholder(self.param_style, 1),
            placeholder(self.param_style, 2),
            placeholder(self.param_style, 3),
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
        get_string(&row, "manifest_path")
    }

    async fn get_manifest_location(
        &self,
        base_uri: &str,
        version: u64,
    ) -> Result<ManifestLocation> {
        let sql = format!(
            "SELECT manifest_path, file_size, e_tag FROM {} \
             WHERE bucket_name = {} AND base_path = {} AND version = {}",
            self.table_name,
            placeholder(self.param_style, 1),
            placeholder(self.param_style, 2),
            placeholder(self.param_style, 3),
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

        let full_path = get_string(&row, "manifest_path")?;
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
            "SELECT version, manifest_path, file_size, e_tag FROM {} \
             WHERE bucket_name = {} AND base_path = {} AND version >= 0 \
             ORDER BY version DESC LIMIT 1",
            self.table_name,
            placeholder(self.param_style, 1),
            placeholder(self.param_style, 2),
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
                let full_path = get_string(&row, "manifest_path")?;
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

    async fn put_if_not_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        e_tag: Option<String>,
    ) -> Result<()> {
        let now = chrono::Utc::now().timestamp();

        let sql = format!(
            "INSERT INTO {} (bucket_name, base_path, version, manifest_path, file_size, e_tag, create_time, update_time) \
             VALUES ({})",
            self.table_name,
            placeholders(self.param_style, 8),
        );

        let mut tx = self.pool.begin().await.map_err(sql_err)?;

        let res = sqlx::query(&sql)
            .bind(&self.bucket_name)
            .bind(base_uri)
            .bind(version as i64)
            .bind(path)
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
                let _ = tx.rollback().await;
                Err(Error::invalid_input(format!(
                    "put_if_not_exists: version {} already exists for bucket: {}; base_path: {}",
                    version, self.bucket_name, base_uri
                )))
            }
            Err(e) => {
                let _ = tx.rollback().await;
                Err(sql_err(e))
            }
        }
    }

    async fn put_if_exists(
        &self,
        base_uri: &str,
        version: u64,
        path: &str,
        size: u64,
        e_tag: Option<String>,
    ) -> Result<()> {
        let now = chrono::Utc::now().timestamp();

        let sql = format!(
            "UPDATE {} SET manifest_path = {}, file_size = {}, e_tag = {}, update_time = {} \
             WHERE bucket_name = {} AND base_path = {} AND version = {}",
            self.table_name,
            placeholder(self.param_style, 1),
            placeholder(self.param_style, 2),
            placeholder(self.param_style, 3),
            placeholder(self.param_style, 4),
            placeholder(self.param_style, 5),
            placeholder(self.param_style, 6),
            placeholder(self.param_style, 7),
        );

        let mut tx = self.pool.begin().await.map_err(sql_err)?;

        let result = sqlx::query(&sql)
            .bind(path)
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
            let _ = tx.rollback().await;
            return Err(Error::not_found(format!(
                "put_if_exists: version {} not found for bucket: {}; base_path: {}",
                version, self.bucket_name, base_uri
            )));
        }

        tx.commit().await.map_err(sql_err)?;
        Ok(())
    }

    async fn delete(&self, base_uri: &str) -> Result<()> {
        let sql = format!(
            "DELETE FROM {} WHERE bucket_name = {} AND base_path = {}",
            self.table_name,
            placeholder(self.param_style, 1),
            placeholder(self.param_style, 2),
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
    use crate::format::DETACHED_VERSION_MASK;

    const TEST_TABLE: &str = "external_manifest";
    const TEST_BUCKET: &str = "test_bucket";
    const TEST_BASE_URI: &str = "datasets/my_table";

    /// Create a fresh in-memory SQLite store with the required schema.
    async fn create_test_store() -> Arc<dyn ExternalManifestStore> {
        // A fresh shared-cache name keeps the DB isolated per test while
        // still allowing multiple pool connections to see the same data.
        let db_name = uuid::Uuid::new_v4().to_string().replace('-', "");
        let url = format!("sqlite:file:{}?mode=memory&cache=shared", db_name);

        sqlx::any::install_default_drivers();

        let pool = AnyPoolOptions::new()
            .max_connections(2)
            .connect(&url)
            .await
            .expect("failed to connect to in-memory SQLite");

        sqlx::query(&format!(
            "CREATE TABLE IF NOT EXISTS {} (
                bucket_name   TEXT NOT NULL,
                base_path     TEXT NOT NULL,
                version       INTEGER NOT NULL,
                manifest_path TEXT NOT NULL,
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

        let store = Arc::new(SqlExternalManifestStore {
            pool,
            table_name: TEST_TABLE.to_string(),
            bucket_name: TEST_BUCKET.to_string(),
            param_style: ParamStyle::from_url(&url),
        });
        store
            .sanity_check()
            .await
            .expect("test schema must pass sanity check");
        store
    }

    #[tokio::test]
    async fn test_put_if_not_exists_succeeds() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/00000000000000000001.manifest-abc123";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 1024, Some("etag1".into()))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_put_if_not_exists_duplicate_fails() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/00000000000000000001.manifest-abc123";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 1024, Some("etag1".into()))
            .await
            .unwrap();

        let result = store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 2048, Some("etag2".into()))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_put_if_exists_succeeds() {
        let store = create_test_store().await;
        let path1 = "datasets/my_table/_versions/00000000000000000001.manifest-staging1";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path1, 1024, Some("etag1".into()))
            .await
            .unwrap();

        let path2 = "datasets/my_table/_versions/00000000000000000001.manifest";
        store
            .put_if_exists(TEST_BASE_URI, 1, path2, 2048, Some("etag2".into()))
            .await
            .unwrap();

        let loc = store.get_manifest_location(TEST_BASE_URI, 1).await.unwrap();
        assert_eq!(loc.size, Some(2048));
        assert_eq!(loc.e_tag, Some("etag2".to_string()));
        assert_eq!(loc.path.as_ref(), path2);
    }

    #[tokio::test]
    async fn test_put_if_exists_missing_fails() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/00000000000000000001.manifest-abc";
        let result = store
            .put_if_exists(TEST_BASE_URI, 1, path, 1024, Some("etag1".into()))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_roundtrip_v2() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/00000000000000000001.manifest-uuid1";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 512, Some("etag_a".into()))
            .await
            .unwrap();

        let got_path = store.get(TEST_BASE_URI, 1).await.unwrap();
        assert_eq!(got_path, path);
    }

    #[tokio::test]
    async fn test_get_roundtrip_v1() {
        // V1 path is `_versions/{version}.manifest` (not zero-padded).
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/1.manifest";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 512, Some("etag_a".into()))
            .await
            .unwrap();

        let got_path = store.get(TEST_BASE_URI, 1).await.unwrap();
        assert_eq!(got_path, path);

        let loc = store.get_manifest_location(TEST_BASE_URI, 1).await.unwrap();
        assert_eq!(
            loc.naming_scheme,
            crate::io::commit::ManifestNamingScheme::V1
        );
    }

    #[tokio::test]
    async fn test_get_latest_version() {
        let store = create_test_store().await;
        let p1 = "datasets/my_table/_versions/00000000000000000001.manifest-a";
        let p3 = "datasets/my_table/_versions/00000000000000000003.manifest-c";
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
            .unwrap();
        assert_eq!(latest.0, 3);
        assert_eq!(latest.1, p3);
    }

    #[tokio::test]
    async fn test_get_manifest_location_roundtrip() {
        let store = create_test_store().await;
        let path = "datasets/my_table/_versions/00000000000000000001.manifest-someuuid";
        store
            .put_if_not_exists(TEST_BASE_URI, 1, path, 4096, Some("my_etag".into()))
            .await
            .unwrap();

        let loc = store.get_manifest_location(TEST_BASE_URI, 1).await.unwrap();
        assert_eq!(loc.version, 1);
        assert_eq!(loc.size, Some(4096));
        assert_eq!(loc.e_tag, Some("my_etag".to_string()));
        assert_eq!(loc.path.as_ref(), path);
    }

    #[tokio::test]
    async fn test_delete_clears_all_versions() {
        let store = create_test_store().await;
        let p1 = "datasets/my_table/_versions/00000000000000000001.manifest-a";
        let p2 = "datasets/my_table/_versions/00000000000000000002.manifest-b";
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
        assert!(latest.is_none());
    }

    #[tokio::test]
    async fn test_detached_version_roundtrip() {
        // Detached versions (high bit set) must be stored and read back
        // transparently — they exercise the u64 -> i64 -> u64 path through
        // BIGINT columns.
        let store = create_test_store().await;
        let detached = DETACHED_VERSION_MASK | 7;
        let path = format!("datasets/my_table/_versions/d{detached}.manifest");

        store
            .put_if_not_exists(TEST_BASE_URI, detached, &path, 100, Some("etag_d".into()))
            .await
            .unwrap();

        assert_eq!(store.get(TEST_BASE_URI, detached).await.unwrap(), path);

        let loc = store
            .get_manifest_location(TEST_BASE_URI, detached)
            .await
            .unwrap();
        assert_eq!(loc.version, detached);
        assert_eq!(loc.path.to_string(), path);
        assert_eq!(loc.size, Some(100));
        assert_eq!(loc.e_tag.as_deref(), Some("etag_d"));
    }

    #[tokio::test]
    async fn test_get_latest_ignores_detached() {
        // get_latest_* must never return a detached version, even when its
        // u64 value is numerically larger than every attached version.
        let store = create_test_store().await;
        let detached = DETACHED_VERSION_MASK | 7;

        store
            .put_if_not_exists(
                TEST_BASE_URI,
                1,
                "datasets/my_table/_versions/1.manifest",
                10,
                None,
            )
            .await
            .unwrap();
        store
            .put_if_not_exists(
                TEST_BASE_URI,
                2,
                "datasets/my_table/_versions/2.manifest",
                20,
                None,
            )
            .await
            .unwrap();
        store
            .put_if_not_exists(
                TEST_BASE_URI,
                detached,
                &format!("datasets/my_table/_versions/d{detached}.manifest"),
                30,
                None,
            )
            .await
            .unwrap();

        let latest = store.get_latest_version(TEST_BASE_URI).await.unwrap();
        assert_eq!(latest.map(|(v, _)| v), Some(2));
    }

    #[tokio::test]
    async fn test_sanity_rejects_no_constraint() {
        let db_name = uuid::Uuid::new_v4().to_string().replace('-', "");
        let url = format!("sqlite:file:{}?mode=memory&cache=shared", db_name);
        sqlx::any::install_default_drivers();
        let pool = AnyPoolOptions::new().connect(&url).await.unwrap();
        sqlx::query(
            "CREATE TABLE no_unique (
                bucket_name   TEXT NOT NULL,
                base_path     TEXT NOT NULL,
                version       INTEGER NOT NULL,
                manifest_path TEXT NOT NULL,
                file_size     INTEGER NOT NULL,
                e_tag         TEXT NOT NULL,
                create_time   INTEGER NOT NULL,
                update_time   INTEGER NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        let store = SqlExternalManifestStore {
            pool,
            table_name: "no_unique".to_string(),
            bucket_name: TEST_BUCKET.to_string(),
            param_style: ParamStyle::from_url(&url),
        };
        let err = store.sanity_check().await.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("UNIQUE") || msg.contains("PRIMARY KEY"),
            "error must mention the missing constraint, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_sanity_accepts_unique_index() {
        // A table without PRIMARY KEY but with a UNIQUE INDEX over the
        // CAS-relevant columns must pass — `probe_uniqueness` checks the
        // unique constraint behaviorally, not the specific DDL form.
        let db_name = uuid::Uuid::new_v4().to_string().replace('-', "");
        let url = format!("sqlite:file:{}?mode=memory&cache=shared", db_name);
        sqlx::any::install_default_drivers();
        let pool = AnyPoolOptions::new().connect(&url).await.unwrap();
        sqlx::query(
            "CREATE TABLE unique_idx_only (
                bucket_name   TEXT NOT NULL,
                base_path     TEXT NOT NULL,
                version       INTEGER NOT NULL,
                manifest_path TEXT NOT NULL,
                file_size     INTEGER NOT NULL,
                e_tag         TEXT NOT NULL,
                create_time   INTEGER NOT NULL,
                update_time   INTEGER NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE UNIQUE INDEX uq_unique_idx_only \
             ON unique_idx_only (bucket_name, base_path, version)",
        )
        .execute(&pool)
        .await
        .unwrap();

        let store = SqlExternalManifestStore {
            pool,
            table_name: "unique_idx_only".to_string(),
            bucket_name: TEST_BUCKET.to_string(),
            param_style: ParamStyle::from_url(&url),
        };
        store
            .sanity_check()
            .await
            .expect("unique index over the CAS columns must satisfy sanity check");
    }

    #[tokio::test]
    async fn test_sanity_rejects_partial_constraint() {
        // A UNIQUE constraint that does not cover `version` must be rejected
        // — duplicates on (bucket, base, vN) would not collide and CAS would
        // silently corrupt data.
        let db_name = uuid::Uuid::new_v4().to_string().replace('-', "");
        let url = format!("sqlite:file:{}?mode=memory&cache=shared", db_name);
        sqlx::any::install_default_drivers();
        let pool = AnyPoolOptions::new().connect(&url).await.unwrap();
        sqlx::query(
            "CREATE TABLE partial_unique (
                bucket_name   TEXT NOT NULL,
                base_path     TEXT NOT NULL,
                version       INTEGER NOT NULL,
                manifest_path TEXT NOT NULL,
                file_size     INTEGER NOT NULL,
                e_tag         TEXT NOT NULL,
                create_time   INTEGER NOT NULL,
                update_time   INTEGER NOT NULL,
                UNIQUE (bucket_name, base_path)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        let store = SqlExternalManifestStore {
            pool,
            table_name: "partial_unique".to_string(),
            bucket_name: TEST_BUCKET.to_string(),
            param_style: ParamStyle::from_url(&url),
        };
        // The first probe insert will already violate the (bucket, base)
        // unique constraint because the cleanup happens in the same tx;
        // either way the probe must surface this as an error rather than
        // silently passing.
        let err = store.sanity_check().await.unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("UNIQUE")
                || msg.contains("PRIMARY KEY")
                || msg.contains("schema sanity probe failed"),
            "error must signal the constraint problem, got: {msg}"
        );
    }

    #[test]
    fn test_validate_identifier_rejects_injection() {
        assert!(validate_sql_identifier("good_table_1", "sqlTableName").is_ok());
        assert!(validate_sql_identifier("ExternalManifest", "sqlTableName").is_ok());

        assert!(validate_sql_identifier("t; DROP TABLE t--", "sqlTableName").is_err());
        assert!(validate_sql_identifier("table name", "sqlTableName").is_err());
        assert!(validate_sql_identifier("table'name", "sqlTableName").is_err());
        assert!(validate_sql_identifier("", "sqlTableName").is_err());
        assert!(validate_sql_identifier("table.name", "sqlTableName").is_err());
    }

    #[test]
    fn test_placeholder_style() {
        assert_eq!(placeholder(ParamStyle::Positional, 1), "?");
        assert_eq!(placeholder(ParamStyle::Positional, 3), "?");
        assert_eq!(placeholder(ParamStyle::Numbered, 1), "$1");
        assert_eq!(placeholder(ParamStyle::Numbered, 7), "$7");
        assert_eq!(placeholders(ParamStyle::Numbered, 3), "$1, $2, $3");
        assert_eq!(placeholders(ParamStyle::Positional, 3), "?, ?, ?");
    }

    #[test]
    fn test_param_style_from_url() {
        assert_eq!(
            ParamStyle::from_url("postgres://h/db"),
            ParamStyle::Numbered
        );
        assert_eq!(
            ParamStyle::from_url("postgresql://h/db"),
            ParamStyle::Numbered
        );
        assert_eq!(ParamStyle::from_url("mysql://h/db"), ParamStyle::Positional);
        assert_eq!(
            ParamStyle::from_url("sqlite://lance.db"),
            ParamStyle::Positional
        );
    }
}
