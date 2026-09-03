// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use opendal::{Operator, services::GooseFs};
use url::Url;

use crate::object_store::opendal_store::OpendalStore;
use crate::object_store::{
    DEFAULT_CLOUD_BLOCK_SIZE, DEFAULT_CLOUD_IO_PARALLELISM, DEFAULT_MAX_IOP_SIZE, ObjectStore,
    ObjectStoreParams, ObjectStoreProvider, StorageOptions,
};
use lance_core::error::{Error, Result};

/// Default GooseFS Master gRPC port.
const DEFAULT_GOOSEFS_PORT: u16 = 9200;

/// Canonical `storage_options` keys. Matching environment variables use the
/// `GOOSEFS_*` spelling; dict keys must be this lowercase form. Wrong-case
/// variants are rejected up front so they cannot silently fall back to the
/// URL authority / OS username and surface as a misleading auth error.
const STORAGE_OPTION_KEYS: &[&str] = &[
    "goosefs_master_addr",
    "goosefs_root",
    "goosefs_write_type",
    "goosefs_block_size",
    "goosefs_chunk_size",
    "goosefs_auth_type",
    "goosefs_auth_username",
];

/// Filename searched under `GOOSEFS_CONF_DIR` / `GOOSEFS_HOME/conf` / `~/.goosefs` / `/etc/goosefs`.
const SITE_PROPERTIES_FILENAME: &str = "goosefs-site.properties";

/// Discover `goosefs-site.properties` using the same search order as goosefs-sdk.
///
/// 1. `$GOOSEFS_CONFIG_FILE`
/// 2. `$GOOSEFS_CONF_DIR/goosefs-site.properties`
/// 3. `$GOOSEFS_HOME/conf/goosefs-site.properties`
/// 4. `~/.goosefs/goosefs-site.properties`
/// 5. `/etc/goosefs/goosefs-site.properties`
fn discover_site_properties_file() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GOOSEFS_CONFIG_FILE") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Ok(dir) = std::env::var("GOOSEFS_CONF_DIR") {
        let p = PathBuf::from(dir).join(SITE_PROPERTIES_FILENAME);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("GOOSEFS_HOME") {
        let p = PathBuf::from(home)
            .join("conf")
            .join(SITE_PROPERTIES_FILENAME);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let p = PathBuf::from(home)
            .join(".goosefs")
            .join(SITE_PROPERTIES_FILENAME);
        if p.is_file() {
            return Some(p);
        }
    }
    let system = PathBuf::from("/etc/goosefs").join(SITE_PROPERTIES_FILENAME);
    if system.is_file() {
        return Some(system);
    }
    None
}

/// True when the properties file names a master via HA list or hostname.
///
/// A file that exists but has neither key must *not* suppress the URI
/// fallback, otherwise `from_properties_auto()` would dial `127.0.0.1:9200`.
fn site_file_has_master_addresses(path: &Path) -> bool {
    let Ok(content) = std::fs::read_to_string(path) else {
        return false;
    };
    parse_properties_has_master(&content)
}

/// Parse `goosefs-site.properties` the same way goosefs-sdk 0.1.8 does, and
/// report whether that parse would set a master from the file.
///
/// Syntax (SDK `PropertiesMap::parse`):
/// - `#` / `!` comments and blank lines are skipped
/// - separator is the first `=` if the line contains one, otherwise the first `:`
/// - empty keys are skipped; empty values are kept so a later line can override
/// - duplicate keys last-win
///
/// Master selection (SDK `into_goosefs_config`):
/// - a non-empty `goosefs.master.rpc.addresses` list wins
/// - otherwise `goosefs.master.hostname` if that key was present
/// - an empty addresses value does *not* fall through to hostname
fn parse_properties_has_master(content: &str) -> bool {
    let mut hostname: Option<&str> = None;
    let mut addresses: Option<&str> = None;
    for raw in content.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        // Prefer `=` when present, else `:`. Matching `find('=').or_else(find(':'))`
        // rather than "first of either" keeps colon-form HA (`host:port` in the
        // value) aligned with the pinned SDK parser.
        let Some(sep_pos) = line.find('=').or_else(|| line.find(':')) else {
            continue;
        };
        let key = line[..sep_pos].trim();
        let value = line[sep_pos + 1..].trim();
        if key.is_empty() {
            continue;
        }
        match key {
            "goosefs.master.rpc.addresses" => addresses = Some(value),
            "goosefs.master.hostname" => hostname = Some(value),
            _ => {}
        }
    }
    if let Some(addrs) = addresses {
        return addrs.split(',').any(|s| !s.trim().is_empty());
    }
    hostname.is_some()
}

fn site_properties_have_master_addresses() -> bool {
    discover_site_properties_file().is_some_and(|path| site_file_has_master_addresses(&path))
}

/// GooseFS object store provider.
///
/// Uses OpenDAL's GooseFs service to access GooseFS via gRPC.
/// URL format: `goosefs://host:port/path`
///
/// Where:
/// - `host:port` is the GooseFS Master address (default port: 9200)
/// - `/path` is the filesystem path within GooseFS
///
/// Path handling model (S3-style):
/// - The OpenDAL `root` is fixed to `/` (or a user-supplied cluster-wide base)
///   so that a single `Operator` can serve every dataset under the same
///   master. This keeps the `ObjectStoreRegistry` cache correct: two URLs
///   like `goosefs://host:9200/a.lance` and `goosefs://host:9200/b.lance`
///   share one store and each request carries its own object key.
/// - Path extraction relies on the default [`ObjectStoreProvider::extract_path`]
///   implementation, which returns the URL path (percent-decoded) as the key
///   passed to `ObjectStore::get`, `put`, etc. — mirroring how `s3://bucket/k`
///   yields key `k`.
///
/// Supported configuration keys (via `storage_options` or environment variables,
/// resolved with priority: `storage_options` > `GOOSEFS_MASTER_ADDR` >
/// `goosefs.master.rpc.addresses` in `goosefs-site.properties` > URL authority).
/// `storage_options` keys must be lowercase; uppercase/mixed-case spellings are
/// rejected rather than silently ignored.
///
/// | storage_options key       | env var                 | purpose                                                                                       |
/// |---------------------------|-------------------------|-----------------------------------------------------------------------------------------------|
/// | `goosefs_master_addr`     | `GOOSEFS_MASTER_ADDR`   | Master gRPC address, e.g. `host:9200`. Supports HA: `addr1:port,addr2:port`.                  |
/// | `goosefs_root`            | `GOOSEFS_ROOT`          | Cluster-wide OpenDAL root shared by all datasets under the same master. Defaults to `/`.      |
/// | `goosefs_write_type`      | `GOOSEFS_WRITE_TYPE`    | GooseFS write type (e.g. `MUST_CACHE`, `CACHE_THROUGH`, `THROUGH`, `ASYNC_THROUGH`).          |
/// | `goosefs_block_size`      | `GOOSEFS_BLOCK_SIZE`    | GooseFS block size. Bytes or suffixes like `4MB` (binary: 1KB=1024). Distinct from Lance I/O `block_size`. |
/// | `goosefs_chunk_size`      | `GOOSEFS_CHUNK_SIZE`    | GooseFS client chunk size. Bytes or suffixes like `4MB` (binary: 1KB=1024).                   |
/// | `goosefs_auth_type`       | `GOOSEFS_AUTH_TYPE`     | Authentication mode: `nosasl` or `simple`.                                                    |
/// | `goosefs_auth_username`   | `GOOSEFS_AUTH_USERNAME` | Username for `simple` auth mode.                                                              |
///
/// Note on `goosefs_root`: it is deliberately cluster-wide (not per-URL) so
/// that many datasets under the same master share a single cached `Operator`.
/// A custom root also participates in the `ObjectStoreRegistry` cache prefix,
/// so stores rooted at different subtrees do not collide.
#[derive(Default, Debug)]
pub struct GooseFsStoreProvider;

/// Parse the exponent of a scientific coefficient (`e10`, `E-3`, `e+02`).
///
/// Magnitudes that do not fit in `i32` saturate to `i32::MAX` / `i32::MIN`
/// so a zero significand such as `0e9999999999` still evaluates to 0, while
/// a non-zero significand overflows later in checked arithmetic.
fn parse_decimal_exponent(s: &str) -> Option<i32> {
    if s.is_empty() {
        return None;
    }
    let (negative, digits) = if let Some(rest) = s.strip_prefix('+') {
        (false, rest)
    } else if let Some(rest) = s.strip_prefix('-') {
        (true, rest)
    } else {
        (false, s)
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    match digits.parse::<i32>() {
        Ok(n) => Some(if negative { -n } else { n }),
        Err(_) => Some(if negative { i32::MIN } else { i32::MAX }),
    }
}

impl GooseFsStoreProvider {
    /// Reject GooseFS `storage_options` keys that match a canonical key
    /// case-insensitively but are not lowercase.
    ///
    /// `GOOSEFS_MASTER_ADDR` is valid as an environment variable, not as a
    /// dict key. Silently ignoring it previously fell back to the URI host
    /// and default OS username, which then failed as `authentication failed`.
    fn validate_storage_option_keys(storage_options: &StorageOptions) -> Result<()> {
        let mut wrong_case = Vec::new();
        for key in storage_options.0.keys() {
            let lower = key.to_ascii_lowercase();
            if let Some(&canonical) = STORAGE_OPTION_KEYS
                .iter()
                .find(|&&canonical| canonical == lower.as_str())
                && key != canonical
            {
                wrong_case.push((key.clone(), canonical));
            }
        }
        if wrong_case.is_empty() {
            return Ok(());
        }
        wrong_case.sort_by(|a, b| a.1.cmp(b.1));
        let details = wrong_case
            .iter()
            .map(|(got, want)| format!("`{got}` (use `{want}`)"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::invalid_input(format!(
            "GooseFS storage_options keys must be lowercase; got {details}"
        )))
    }

    /// Resolve the GooseFS Master address to pass to OpenDAL.
    ///
    /// Returns `Ok(Some(addr))` when Lance should set OpenDAL `master_addr`
    /// (this *overrides* anything `from_properties_auto()` loaded from
    /// `goosefs-site.properties`). Returns `Ok(None)` when the site file
    /// already names masters and neither `storage_options` nor
    /// `GOOSEFS_MASTER_ADDR` was set — OpenDAL then uses the file as-is,
    /// so a dummy URI authority is not dialed.
    ///
    /// Priority:
    /// 1. `storage_options["goosefs_master_addr"]` (supports HA: "addr1:port,addr2:port")
    /// 2. `GOOSEFS_MASTER_ADDR` environment variable
    /// 3. `goosefs.master.rpc.addresses` / `goosefs.master.hostname` in the
    ///    discovered `goosefs-site.properties` (`Ok(None)` — do not overlay)
    /// 4. URL authority (host:port from the URL)
    fn resolve_master_addr(url: &Url, storage_options: &StorageOptions) -> Result<Option<String>> {
        // 1. storage_options
        if let Some(addr) = storage_options
            .0
            .get("goosefs_master_addr")
            .filter(|v| !v.is_empty())
        {
            return Ok(Some(addr.clone()));
        }

        // 2. Environment variable
        if let Ok(addr) = std::env::var("GOOSEFS_MASTER_ADDR")
            && !addr.is_empty()
        {
            return Ok(Some(addr));
        }

        // 3. Site properties already name a master. Leave OpenDAL's
        // `from_properties_auto()` overlay empty so HA from the file wins
        // over a dummy `goosefs://192.0.2.5:9999/...` URI.
        if site_properties_have_master_addresses() {
            return Ok(None);
        }

        // 4. URL authority — required when nothing else supplied a master.
        let host = url.host_str().ok_or_else(|| {
            Error::invalid_input(
                "GooseFS URL must contain a master address (host), e.g. goosefs://host:port/path, \
                 or set goosefs_master_addr / GOOSEFS_MASTER_ADDR / \
                 goosefs.master.rpc.addresses in goosefs-site.properties",
            )
        })?;

        let port = url.port().unwrap_or(DEFAULT_GOOSEFS_PORT);
        Ok(Some(format!("{}:{}", host, port)))
    }

    /// Resolve a storage option from storage_options or environment variable.
    fn resolve_option(
        storage_options: &StorageOptions,
        option_key: &str,
        env_key: &str,
    ) -> Option<String> {
        storage_options
            .0
            .get(option_key)
            .cloned()
            .or_else(|| std::env::var(env_key).ok())
            .filter(|v| !v.is_empty())
    }

    /// Resolve the OpenDAL `root` for this Operator. See the file-level docs on
    /// [`GooseFsStoreProvider`] for the semantics of `goosefs_root`.
    fn resolve_root(storage_options: &StorageOptions) -> String {
        Self::resolve_option(storage_options, "goosefs_root", "GOOSEFS_ROOT")
            .unwrap_or_else(|| "/".to_string())
    }

    /// Parse a GooseFS space-size string into bytes.
    ///
    /// OpenDAL's GooseFS config deserializes `chunk_size` / `block_size` as
    /// `u64`, so a value like `4MB` fails with `ParseIntError`. GooseFS itself
    /// accepts Hadoop-style suffixes via `FormatUtils.parseSpaceSize`, where
    /// units are **binary** (`1KB = 1024`, so `4MB` is `4194304`), not SI
    /// (`10^6`). Generic crates such as `parse-size` / `bytesize` default to
    /// SI for `MB`, so this parser matches GooseFS instead of adding a dep.
    ///
    /// Accepted suffixes (case-insensitive): `b`, `k`/`kb`, `m`/`mb`,
    /// `g`/`gb`, `t`/`tb`, `p`/`pb`. A missing suffix means bytes. Fractional
    /// and scientific coefficients such as `1.5MB` or `1e3KB` are accepted
    /// and converted with exact decimal arithmetic (truncated toward zero,
    /// matching GooseFS `parseSpaceSize` without the `double` rounding fudge).
    fn parse_space_size(option_key: &str, value: &str) -> Result<u64> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(Error::invalid_input(format!(
                "{option_key} must be a size such as `4MB` or `4194304`, got `{value}`"
            )));
        }

        let suffix_start = trimmed
            .char_indices()
            .rev()
            .find(|(_, c)| c.is_ascii_digit())
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        let (number, suffix) = trimmed.split_at(suffix_start);
        if number.is_empty() {
            return Err(Error::invalid_input(format!(
                "invalid {option_key} `{value}`: missing numeric coefficient"
            )));
        }

        let multiplier = match suffix.to_ascii_lowercase().as_str() {
            "" | "b" => 1u64,
            "k" | "kb" => 1024,
            "m" | "mb" => 1024 * 1024,
            "g" | "gb" => 1024 * 1024 * 1024,
            "t" | "tb" => 1024u64.pow(4),
            "p" | "pb" => 1024u64.pow(5),
            other => {
                return Err(Error::invalid_input(format!(
                    "invalid {option_key} `{value}`: unrecognized size suffix `{other}` \
                     (supported: b, k/kb, m/mb, g/gb, t/tb, p/pb; units are binary, 1KB=1024)"
                )));
            }
        };

        let overflow = || {
            Error::invalid_input(format!(
                "invalid {option_key} `{value}`: size overflows u64"
            ))
        };

        // Exact integer path. Digit-only coefficients that fail `u64` parsing
        // have overflowed; do not retry them as decimals.
        if let Ok(n) = number.parse::<u64>() {
            return n.checked_mul(multiplier).ok_or_else(overflow);
        }
        if number.chars().all(|c| c.is_ascii_digit()) {
            return Err(overflow());
        }

        Self::parse_decimal_times_unit(option_key, value, number, multiplier)
    }

    /// Parse a decimal or scientific coefficient and return `floor(coeff * multiplier)`.
    ///
    /// GooseFS `FormatUtils.parseSpaceSize` truncates toward zero after
    /// `coeff * unit` in IEEE-754 `double`. That silently changes integers
    /// above `2^53` (e.g. `9007199254740993.0` becomes `9007199254740992`).
    /// This path keeps the same truncation semantics with checked decimal
    /// arithmetic so every accepted configuration preserves its value.
    fn parse_decimal_times_unit(
        option_key: &str,
        value: &str,
        number: &str,
        multiplier: u64,
    ) -> Result<u64> {
        let invalid_number = || {
            Error::invalid_input(format!(
                "invalid {option_key} `{value}`: `{number}` is not a valid number"
            ))
        };
        let overflow = || {
            Error::invalid_input(format!(
                "invalid {option_key} `{value}`: size overflows u64"
            ))
        };
        let negative = || {
            Error::invalid_input(format!(
                "invalid {option_key} `{value}`: size must be a non-negative number"
            ))
        };

        if number.starts_with('-') {
            return Err(negative());
        }
        let s = number.strip_prefix('+').unwrap_or(number);
        if s.is_empty() {
            return Err(invalid_number());
        }

        let (mantissa, exp) = if let Some(e_idx) = s.find(['e', 'E']) {
            let mantissa = &s[..e_idx];
            let exp_str = &s[e_idx + 1..];
            if mantissa.is_empty() {
                return Err(invalid_number());
            }
            let exp = parse_decimal_exponent(exp_str).ok_or_else(invalid_number)?;
            (mantissa, exp)
        } else {
            (s, 0i32)
        };

        let mut parts = mantissa.split('.');
        let int_str = parts.next().unwrap_or("");
        let frac_raw = parts.next();
        if parts.next().is_some() {
            return Err(invalid_number());
        }
        if !int_str.chars().all(|c| c.is_ascii_digit()) {
            return Err(invalid_number());
        }
        let frac_raw = match frac_raw {
            Some(frac) if frac.chars().all(|c| c.is_ascii_digit()) => frac,
            Some(_) => return Err(invalid_number()),
            None => "",
        };
        if int_str.is_empty() && frac_raw.is_empty() {
            return Err(invalid_number());
        }
        let frac_str = frac_raw.trim_end_matches('0');

        let int_part: u128 = if int_str.is_empty() {
            0
        } else {
            int_str.parse().map_err(|_| overflow())?
        };
        let frac_digits = u32::try_from(frac_str.len()).map_err(|_| overflow())?;
        let frac_part: u128 = if frac_str.is_empty() {
            0
        } else {
            frac_str.parse().map_err(|_| overflow())?
        };

        let significand = if frac_digits == 0 {
            int_part
        } else {
            let pow10 = 10u128.checked_pow(frac_digits).ok_or_else(overflow)?;
            int_part
                .checked_mul(pow10)
                .and_then(|v| v.checked_add(frac_part))
                .ok_or_else(overflow)?
        };
        if significand == 0 {
            return Ok(0);
        }

        // floor(significand * multiplier * 10^exp / 10^frac_digits)
        let mut num = significand
            .checked_mul(u128::from(multiplier))
            .ok_or_else(overflow)?;
        let scale = i64::from(frac_digits) - i64::from(exp);
        if scale > 0 {
            match u32::try_from(scale)
                .ok()
                .and_then(|s| 10u128.checked_pow(s))
            {
                Some(den) => num /= den,
                None => num = 0,
            }
        } else if scale < 0 {
            let raise = u32::try_from(scale.unsigned_abs()).map_err(|_| overflow())?;
            let factor = 10u128.checked_pow(raise).ok_or_else(overflow)?;
            num = num.checked_mul(factor).ok_or_else(overflow)?;
        }

        u64::try_from(num).map_err(|_| overflow())
    }

    fn resolve_space_size(
        storage_options: &StorageOptions,
        option_key: &str,
        env_key: &str,
    ) -> Result<Option<u64>> {
        let Some(raw) = Self::resolve_option(storage_options, option_key, env_key) else {
            return Ok(None);
        };
        Self::parse_space_size(option_key, &raw).map(Some)
    }
}

#[async_trait::async_trait]
impl ObjectStoreProvider for GooseFsStoreProvider {
    async fn new_store(&self, base_path: Url, params: &ObjectStoreParams) -> Result<ObjectStore> {
        let block_size = params.block_size.unwrap_or(DEFAULT_CLOUD_BLOCK_SIZE);
        let storage_options = StorageOptions(params.storage_options().cloned().unwrap_or_default());

        Self::validate_storage_option_keys(&storage_options)?;

        // Resolve a stable cluster-wide root. The URL path is *not* used here
        // because it varies per dataset; per-request keys are supplied by
        // `extract_path` instead.
        let root = Self::resolve_root(&storage_options);

        // Build OpenDAL config map. `master_addr` is omitted when site
        // properties already name masters so OpenDAL `from_properties_auto()`
        // keeps HA instead of dialing a dummy URI authority.
        let mut config_map: HashMap<String, String> = HashMap::new();
        if let Some(master_addr) = Self::resolve_master_addr(&base_path, &storage_options)? {
            config_map.insert("master_addr".to_string(), master_addr);
        }
        config_map.insert("root".to_string(), root);

        // Optional: write_type
        if let Some(wt) =
            Self::resolve_option(&storage_options, "goosefs_write_type", "GOOSEFS_WRITE_TYPE")
        {
            config_map.insert("write_type".to_string(), wt);
        }

        // Optional: block_size (for GooseFS, not Lance block_size).
        // Normalize suffixes like `64MB` to a raw byte count; OpenDAL expects u64.
        if let Some(bs) =
            Self::resolve_space_size(&storage_options, "goosefs_block_size", "GOOSEFS_BLOCK_SIZE")?
        {
            config_map.insert("block_size".to_string(), bs.to_string());
        }

        // Optional: chunk_size
        if let Some(cs) =
            Self::resolve_space_size(&storage_options, "goosefs_chunk_size", "GOOSEFS_CHUNK_SIZE")?
        {
            config_map.insert("chunk_size".to_string(), cs.to_string());
        }

        // Optional: auth_type (nosasl / simple)
        if let Some(at) =
            Self::resolve_option(&storage_options, "goosefs_auth_type", "GOOSEFS_AUTH_TYPE")
        {
            config_map.insert("auth_type".to_string(), at);
        }

        // Optional: auth_username (used in SIMPLE auth mode)
        if let Some(au) = Self::resolve_option(
            &storage_options,
            "goosefs_auth_username",
            "GOOSEFS_AUTH_USERNAME",
        ) {
            config_map.insert("auth_username".to_string(), au);
        }

        // Create OpenDAL Operator with GooseFS service
        let operator = Operator::from_iter::<GooseFs>(config_map).map_err(|e| {
            Error::invalid_input(format!("Failed to create GooseFS operator: {:?}", e))
        })?;

        // Wrap as object_store::ObjectStore via OpendalStore bridge
        let opendal_store = Arc::new(OpendalStore::new(operator));

        Ok(ObjectStore {
            scheme: "goosefs".to_string(),
            inner: opendal_store,
            local_dir_operations: None,
            block_size,
            max_iop_size: *DEFAULT_MAX_IOP_SIZE,
            use_constant_size_upload_parts: params.use_constant_size_upload_parts,
            list_is_lexically_ordered: params.list_is_lexically_ordered.unwrap_or(false),
            io_parallelism: DEFAULT_CLOUD_IO_PARALLELISM,
            download_retry_count: storage_options.download_retry_count(),
            io_tracker: Default::default(),
            store_prefix: self
                .calculate_object_store_prefix(&base_path, params.storage_options())?,
            // Listed in full: no paginated lister covers OpenDAL yet.
            paginated_lister: None,
        })
    }

    // `extract_path` uses the default `ObjectStoreProvider` trait implementation:
    // it percent-decodes the URL path and returns it as the object key, exactly
    // like S3 does for `s3://bucket/key`. Overriding it here would only
    // duplicate that behavior. See the file-level doc comment above for the
    // full path-handling model.

    /// Calculate the object store prefix used as the registry cache key.
    ///
    /// Format: `goosefs$host:port`. Because the OpenDAL root is now cluster-
    /// wide (not per-URL), all datasets under the same master intentionally
    /// share the same cached [`ObjectStore`]; the URL path is disambiguated
    /// by [`Self::extract_path`] on each request. This is analogous to how
    /// two `s3://bucket/a` and `s3://bucket/b` URLs share one store.
    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<String> {
        // If a custom `goosefs_root` is provided, include it in the prefix so
        // that stores built with different roots don't accidentally collide.
        let opts = StorageOptions(storage_options.cloned().unwrap_or_default());
        let root = Self::resolve_root(&opts);
        if root == "/" {
            Ok(format!("{}${}", url.scheme(), url.authority()))
        } else {
            Ok(format!("{}${}#{}", url.scheme(), url.authority(), root))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serial_test::serial;

    /// Point `GOOSEFS_CONFIG_FILE` at a temp properties file so discovery
    /// does not pick up `~/.goosefs` / `/etc/goosefs` from the host.
    fn with_site_properties(contents: &str, f: impl FnOnce()) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SITE_PROPERTIES_FILENAME);
        std::fs::write(&path, contents).unwrap();
        unsafe {
            std::env::set_var("GOOSEFS_CONFIG_FILE", &path);
            std::env::remove_var("GOOSEFS_MASTER_ADDR");
            std::env::remove_var("GOOSEFS_CONF_DIR");
            std::env::remove_var("GOOSEFS_HOME");
        }
        f();
        unsafe {
            std::env::remove_var("GOOSEFS_CONFIG_FILE");
        }
    }

    fn resolve(url: &str, opts: StorageOptions) -> Result<Option<String>> {
        GooseFsStoreProvider::resolve_master_addr(&Url::parse(url).unwrap(), &opts)
    }

    #[test]
    fn test_goosefs_extract_path_basic() {
        let provider = GooseFsStoreProvider;
        let url = Url::parse("goosefs://10.0.0.1:9200/data/embeddings.lance").unwrap();
        let path = provider.extract_path(&url).unwrap();
        assert_eq!(path.to_string(), "data/embeddings.lance");
    }

    #[test]
    fn test_goosefs_extract_path_root() {
        let provider = GooseFsStoreProvider;
        let url = Url::parse("goosefs://10.0.0.1:9200/").unwrap();
        let path = provider.extract_path(&url).unwrap();
        assert_eq!(path.to_string(), "");
    }

    #[test]
    fn test_goosefs_extract_path_deep() {
        let provider = GooseFsStoreProvider;
        let url = Url::parse("goosefs://master:9200/a/b/c/d.lance").unwrap();
        let path = provider.extract_path(&url).unwrap();
        assert_eq!(path.to_string(), "a/b/c/d.lance");
    }

    #[test]
    fn test_goosefs_extract_path_percent_decoded() {
        // The URL contains a percent-encoded space; extract_path must decode
        // it once so the ObjectStore layer does not double-encode later.
        let provider = GooseFsStoreProvider;
        let url = Url::parse("goosefs://master:9200/dir/with%20space/f.lance").unwrap();
        let path = provider.extract_path(&url).unwrap();
        assert_eq!(path.to_string(), "dir/with space/f.lance");
    }

    #[test]
    fn test_calculate_object_store_prefix_default_root() {
        let provider = GooseFsStoreProvider;
        let url = Url::parse("goosefs://10.0.0.1:9200/data").unwrap();
        let prefix = provider.calculate_object_store_prefix(&url, None).unwrap();
        assert_eq!(prefix, "goosefs$10.0.0.1:9200");
    }

    #[test]
    fn test_calculate_object_store_prefix_with_hostname() {
        let provider = GooseFsStoreProvider;
        let url = Url::parse("goosefs://myhost:9200/data").unwrap();
        let prefix = provider.calculate_object_store_prefix(&url, None).unwrap();
        assert_eq!(prefix, "goosefs$myhost:9200");
    }

    /// Regression test: two URLs pointing at different datasets under the
    /// same master must produce the *same* cache prefix so they share one
    /// Operator, and correctness must come from `extract_path` returning
    /// distinct keys — never from a per-URL root baked into the prefix.
    #[test]
    fn test_prefix_shared_across_datasets_same_master() {
        let provider = GooseFsStoreProvider;
        let url_a = Url::parse("goosefs://10.0.0.1:9200/repro/a.lance").unwrap();
        let url_b = Url::parse("goosefs://10.0.0.1:9200/repro/b.lance").unwrap();

        let pa = provider
            .calculate_object_store_prefix(&url_a, None)
            .unwrap();
        let pb = provider
            .calculate_object_store_prefix(&url_b, None)
            .unwrap();
        assert_eq!(pa, pb, "same master must share one cache prefix");

        // Extracted keys must differ so the shared Operator can route
        // requests to the correct dataset.
        assert_ne!(
            provider.extract_path(&url_a).unwrap(),
            provider.extract_path(&url_b).unwrap(),
            "distinct URLs must yield distinct object keys",
        );
    }

    /// Different masters must never share a cache entry.
    #[test]
    fn test_prefix_isolated_across_masters() {
        let provider = GooseFsStoreProvider;
        let u1 = Url::parse("goosefs://host-a:9200/x.lance").unwrap();
        let u2 = Url::parse("goosefs://host-b:9200/x.lance").unwrap();
        assert_ne!(
            provider.calculate_object_store_prefix(&u1, None).unwrap(),
            provider.calculate_object_store_prefix(&u2, None).unwrap(),
        );
    }

    /// A user-supplied `goosefs_root` participates in the cache prefix so
    /// stores rooted at different subtrees don't collide.
    #[test]
    fn test_prefix_includes_custom_root() {
        let provider = GooseFsStoreProvider;
        let url = Url::parse("goosefs://host:9200/x.lance").unwrap();

        let default_prefix = provider.calculate_object_store_prefix(&url, None).unwrap();
        let custom_opts: HashMap<String, String> =
            HashMap::from([("goosefs_root".to_string(), "/tenant-a".to_string())]);
        let custom_prefix = provider
            .calculate_object_store_prefix(&url, Some(&custom_opts))
            .unwrap();

        assert_eq!(default_prefix, "goosefs$host:9200");
        assert_eq!(custom_prefix, "goosefs$host:9200#/tenant-a");
        assert_ne!(default_prefix, custom_prefix);
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_from_url() {
        with_site_properties("", || {
            let addr = resolve(
                "goosefs://10.0.0.1:9200/data",
                StorageOptions(HashMap::new()),
            )
            .unwrap();
            assert_eq!(addr.as_deref(), Some("10.0.0.1:9200"));
        });
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_default_port() {
        with_site_properties("", || {
            let addr = resolve("goosefs://10.0.0.1/data", StorageOptions(HashMap::new())).unwrap();
            assert_eq!(addr.as_deref(), Some("10.0.0.1:9200"));
        });
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_from_storage_options() {
        with_site_properties(
            "goosefs.master.rpc.addresses=10.0.0.9:9200,10.0.0.8:9200\n",
            || {
                let storage_options = StorageOptions(HashMap::from([(
                    "goosefs_master_addr".to_string(),
                    "10.0.0.2:9200,10.0.0.3:9200".to_string(),
                )]));
                let addr = resolve("goosefs://10.0.0.1:9200/data", storage_options).unwrap();
                assert_eq!(addr.as_deref(), Some("10.0.0.2:9200,10.0.0.3:9200"));
            },
        );
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_env_beats_uri_and_site() {
        with_site_properties(
            "goosefs.master.rpc.addresses=10.0.0.9:9200,10.0.0.8:9200\n",
            || {
                unsafe {
                    std::env::set_var("GOOSEFS_MASTER_ADDR", "10.0.0.7:9200,10.0.0.6:9200");
                }
                let addr = resolve(
                    "goosefs://192.0.2.5:9999/data",
                    StorageOptions(HashMap::new()),
                )
                .unwrap();
                unsafe {
                    std::env::remove_var("GOOSEFS_MASTER_ADDR");
                }
                assert_eq!(addr.as_deref(), Some("10.0.0.7:9200,10.0.0.6:9200"));
            },
        );
    }

    /// Ticket: dummy URI + empty storage_options + site HA must not overlay
    /// OpenDAL `master_addr` (OpenDAL then keeps `from_properties_auto()`).
    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_site_ha_suppresses_dummy_uri() {
        with_site_properties(
            "goosefs.master.rpc.addresses=172.31.5.10:9200,172.31.5.2:9200,172.31.5.11:9200\n",
            || {
                let addr = resolve(
                    "goosefs://192.0.2.5:9999/lance-cosn/lance_qta/xxx.lance",
                    StorageOptions(HashMap::new()),
                )
                .unwrap();
                assert_eq!(addr, None);
            },
        );
    }

    /// Colon-form HA is valid in goosefs-sdk 0.1.8 (`=` or `:`). Missing it
    /// here would overlay the dummy URI and wipe the SDK-loaded HA list.
    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_site_ha_colon_form_suppresses_dummy_uri() {
        with_site_properties(
            "goosefs.master.rpc.addresses:172.31.5.10:9200,172.31.5.2:9200,172.31.5.11:9200\n",
            || {
                let addr = resolve(
                    "goosefs://192.0.2.5:9999/lance-cosn/lance_qta/xxx.lance",
                    StorageOptions(HashMap::new()),
                )
                .unwrap();
                assert_eq!(addr, None);
            },
        );
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_site_ha_allows_uri_without_host() {
        with_site_properties(
            "goosefs.master.rpc.addresses=10.0.0.1:9200,10.0.0.2:9200\n",
            || {
                let addr =
                    resolve("goosefs:///data/ds.lance", StorageOptions(HashMap::new())).unwrap();
                assert_eq!(addr, None);
            },
        );
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_site_hostname_suppresses_uri() {
        with_site_properties("goosefs.master.hostname=goosefs-master\n", || {
            let addr = resolve(
                "goosefs://192.0.2.5:9999/data",
                StorageOptions(HashMap::new()),
            )
            .unwrap();
            assert_eq!(addr, None);
        });
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_site_hostname_colon_form_suppresses_uri() {
        with_site_properties("goosefs.master.hostname:goosefs-master\n", || {
            let addr = resolve(
                "goosefs://192.0.2.5:9999/data",
                StorageOptions(HashMap::new()),
            )
            .unwrap();
            assert_eq!(addr, None);
        });
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_empty_site_file_keeps_real_uri() {
        with_site_properties(
            "# no master keys\ngoosefs.user.block.size.bytes.default=4MB\n",
            || {
                let addr = resolve(
                    "goosefs://10.0.0.1:9200/data",
                    StorageOptions(HashMap::new()),
                )
                .unwrap();
                assert_eq!(addr.as_deref(), Some("10.0.0.1:9200"));
            },
        );
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_resolve_master_addr_no_host_without_site_errors() {
        with_site_properties("", || {
            let err =
                resolve("goosefs:///data/ds.lance", StorageOptions(HashMap::new())).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("must contain a master address"), "got: {msg}");
        });
    }

    #[test]
    #[serial(GOOSEFS_SITE_CONF)]
    fn test_discover_site_properties_via_conf_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(SITE_PROPERTIES_FILENAME);
        std::fs::write(
            &path,
            "goosefs.master.rpc.addresses=10.0.0.1:9200,10.0.0.2:9200\n",
        )
        .unwrap();
        unsafe {
            std::env::remove_var("GOOSEFS_CONFIG_FILE");
            std::env::remove_var("GOOSEFS_MASTER_ADDR");
            std::env::remove_var("GOOSEFS_HOME");
            std::env::set_var("GOOSEFS_CONF_DIR", dir.path());
        }

        let addr = resolve(
            "goosefs://192.0.2.5:9999/data",
            StorageOptions(HashMap::new()),
        )
        .unwrap();
        unsafe {
            std::env::remove_var("GOOSEFS_CONF_DIR");
        }
        assert_eq!(addr, None);
    }

    #[test]
    fn test_parse_properties_has_master_ignores_comments_and_empty() {
        assert!(!parse_properties_has_master(""));
        assert!(!parse_properties_has_master(
            "# goosefs.master.rpc.addresses=a:9200\n"
        ));
        assert!(!parse_properties_has_master(
            "goosefs.master.rpc.addresses=\n"
        ));
        assert!(!parse_properties_has_master(
            "goosefs.master.rpc.addresses=   ,  \n"
        ));
        assert!(parse_properties_has_master(
            "goosefs.master.rpc.addresses=10.0.0.1:9200,10.0.0.2:9200\n"
        ));
        assert!(parse_properties_has_master(
            "  goosefs.master.hostname = master.local  \n"
        ));
        // SDK accepts `:` as well as `=`; colon-form HA must not be missed.
        assert!(parse_properties_has_master(
            "goosefs.master.rpc.addresses:10.0.0.1:9200,10.0.0.2:9200\n"
        ));
        assert!(parse_properties_has_master(
            "goosefs.master.hostname:goosefs-master\n"
        ));
        // Last-wins, including an empty later line clearing a colon-form HA list.
        assert!(parse_properties_has_master(
            "goosefs.master.rpc.addresses=10.0.0.9:9200\n\
             goosefs.master.rpc.addresses:10.0.0.1:9200,10.0.0.2:9200\n"
        ));
        assert!(!parse_properties_has_master(
            "goosefs.master.rpc.addresses:10.0.0.1:9200,10.0.0.2:9200\n\
             goosefs.master.rpc.addresses=\n"
        ));
        // Empty addresses does not fall through to hostname (SDK if/else-if).
        assert!(!parse_properties_has_master(
            "goosefs.master.rpc.addresses=\ngoosefs.master.hostname=master.local\n"
        ));
    }

    #[test]
    fn test_resolve_root_defaults_to_slash() {
        let opts = StorageOptions(HashMap::new());
        assert_eq!(GooseFsStoreProvider::resolve_root(&opts), "/");
    }

    #[test]
    fn test_resolve_root_from_storage_options() {
        let opts = StorageOptions(HashMap::from([(
            "goosefs_root".to_string(),
            "/tenant-a".to_string(),
        )]));
        assert_eq!(GooseFsStoreProvider::resolve_root(&opts), "/tenant-a");
    }

    #[test]
    fn test_validate_storage_option_keys_accepts_lowercase() {
        let opts = StorageOptions(HashMap::from([
            (
                "goosefs_master_addr".to_string(),
                "10.0.0.1:9200,10.0.0.2:9200".to_string(),
            ),
            ("goosefs_auth_type".to_string(), "simple".to_string()),
            ("goosefs_auth_username".to_string(), "lance".to_string()),
            ("allow_http".to_string(), "true".to_string()),
        ]));
        GooseFsStoreProvider::validate_storage_option_keys(&opts).unwrap();
    }

    #[test]
    fn test_validate_storage_option_keys_rejects_uppercase() {
        let opts = StorageOptions(HashMap::from([(
            "GOOSEFS_MASTER_ADDR".to_string(),
            "10.0.0.1:9200,10.0.0.2:9200".to_string(),
        )]));
        let err = GooseFsStoreProvider::validate_storage_option_keys(&opts).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("must be lowercase"),
            "expected lowercase-key error, got: {msg}"
        );
        assert!(
            msg.contains("`GOOSEFS_MASTER_ADDR` (use `goosefs_master_addr`)"),
            "expected canonical key hint, got: {msg}"
        );
    }

    #[test]
    fn test_validate_storage_option_keys_rejects_mixed_case() {
        let opts = StorageOptions(HashMap::from([
            (
                "Goosefs_Master_Addr".to_string(),
                "10.0.0.1:9200".to_string(),
            ),
            ("GOOSEFS_AUTH_TYPE".to_string(), "simple".to_string()),
        ]));
        let err = GooseFsStoreProvider::validate_storage_option_keys(&opts).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("must be lowercase"), "got: {msg}");
        assert!(msg.contains("goosefs_master_addr"), "got: {msg}");
        assert!(msg.contains("goosefs_auth_type"), "got: {msg}");
    }

    #[rstest]
    #[case::bare_bytes("4096", 4096)]
    #[case::suffix_b("4096b", 4096)]
    #[case::suffix_k("4k", 4096)]
    #[case::suffix_kb("4KB", 4096)]
    #[case::suffix_m("4m", 4 * 1024 * 1024)]
    #[case::suffix_mb("4MB", 4 * 1024 * 1024)]
    #[case::suffix_gb("1GB", 1024 * 1024 * 1024)]
    #[case::fractional_mb("1.5MB", 1_572_864)]
    #[case::fractional_kb("2.5KB", 2560)]
    #[case::scientific_kb("1e3KB", 1000 * 1024)]
    #[case::leading_dot(".5KB", 512)]
    #[case::truncate_toward_zero("1.9", 1)]
    #[case::trimmed("  4MB  ", 4 * 1024 * 1024)]
    fn test_parse_space_size_accepts_goosefs_suffixes(#[case] input: &str, #[case] expected: u64) {
        assert_eq!(
            GooseFsStoreProvider::parse_space_size("goosefs_chunk_size", input).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::empty("")]
    #[case::whitespace("   ")]
    #[case::missing_number("MB")]
    #[case::unknown_suffix("4MiB")]
    #[case::si_mismatch_not_mib("4XB")]
    #[case::negative("-4MB")]
    #[case::two_dots("1.2.3")]
    fn test_parse_space_size_rejects_invalid(#[case] input: &str) {
        let err = GooseFsStoreProvider::parse_space_size("goosefs_chunk_size", input).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("goosefs_chunk_size"),
            "error should name the option, got: {msg}"
        );
        assert!(
            msg.contains(input.trim()) || input.trim().is_empty(),
            "error should include the input, got: {msg}"
        );
    }

    #[rstest]
    #[case::u64_max_plus_one("18446744073709551616")]
    #[case::u64_max_plus_one_bytes("18446744073709551616b")]
    #[case::integer_mul_overflow("16384PB")]
    #[case::fractional_mul_overflow("16384.0PB")]
    #[case::scientific_overflow("1e20")]
    #[case::scientific_u64_max_plus_one("1.8446744073709551616e19")]
    fn test_parse_space_size_rejects_overflow(#[case] input: &str) {
        let err = GooseFsStoreProvider::parse_space_size("goosefs_chunk_size", input).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("overflows u64"),
            "overflowing size should be rejected, got: {msg}"
        );
        assert!(
            msg.contains(input),
            "error should include the input, got: {msg}"
        );
    }

    #[test]
    fn test_parse_space_size_accepts_u64_max_integer() {
        assert_eq!(
            GooseFsStoreProvider::parse_space_size("goosefs_chunk_size", "18446744073709551615")
                .unwrap(),
            u64::MAX
        );
        assert_eq!(
            GooseFsStoreProvider::parse_space_size(
                "goosefs_chunk_size",
                "1.8446744073709551615e19"
            )
            .unwrap(),
            u64::MAX
        );
    }

    #[test]
    fn test_parse_space_size_preserves_values_beyond_f64_precision() {
        // 2^53+1 is not representable as f64. The previous `double` path
        // silently converted `9007199254740993.0` to `9007199254740992`.
        assert_eq!(
            GooseFsStoreProvider::parse_space_size("goosefs_chunk_size", "9007199254740993.0")
                .unwrap(),
            9007199254740993
        );
        assert_eq!(
            GooseFsStoreProvider::parse_space_size("goosefs_chunk_size", "9007199254740993.0b")
                .unwrap(),
            9007199254740993
        );
    }

    #[test]
    fn test_parse_space_size_four_mb_is_binary_not_si() {
        let bytes = GooseFsStoreProvider::parse_space_size("goosefs_chunk_size", "4MB").unwrap();
        assert_eq!(bytes, 4_194_304, "GooseFS MB is 1024^2, not 10^6");
        assert_ne!(bytes, 4_000_000);
    }

    #[test]
    fn test_resolve_space_size_from_storage_options() {
        let opts = StorageOptions(HashMap::from([(
            "goosefs_chunk_size".to_string(),
            "4MB".to_string(),
        )]));
        let bytes = GooseFsStoreProvider::resolve_space_size(
            &opts,
            "goosefs_chunk_size",
            "GOOSEFS_CHUNK_SIZE",
        )
        .unwrap();
        assert_eq!(bytes, Some(4 * 1024 * 1024));
    }
}
