// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::path::Path;
use serde_json::Value;
use std::fmt;

const CURRENT_API_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
}

pub trait PluginInstance {
    fn init(&mut self, config: &Value) -> Result<(), String>;
    fn execute(&self, input: &str) -> String;
    fn metadata(&self) -> PluginMetadata;
}

#[repr(C)]
pub struct PluginInterface {
    pub create_plugin: unsafe extern "C" fn() -> *mut dyn PluginInstance,
    pub destroy_plugin: unsafe extern "C" fn(*mut dyn PluginInstance),
    pub api_version: u32,
}

pub struct PluginManager {
    plugins: HashMap<String, (Box<dyn PluginInstance>, Library)>,
}

#[derive(Debug)]
pub enum PluginError {
    LibraryLoad(libloading::Error),
    SymbolError(String),
    IncompatibleAPI,
    NotFound(String),
    NullPointer(String),
    LibraryUnload(String),
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PluginError::LibraryLoad(e) => write!(f, "Library load error: {}", e),
            PluginError::SymbolError(e) => write!(f, "Symbol error: {}", e),
            PluginError::IncompatibleAPI => write!(f, "Incompatible API version"),
            PluginError::NotFound(e) => write!(f, "Plugin not found: {}", e),
            PluginError::NullPointer(e) => write!(f, "Null pointer: {}", e),
            PluginError::LibraryUnload(e) => write!(f, "Library unload error: {}", e),
        }
    }
}

impl PluginManager {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    pub fn load_plugin(&mut self, path: &Path) -> Result<(), PluginError> {
        unsafe {
            log::debug!("Loading plugin from: {}", path.display());

            let lib = Library::new(path).map_err(PluginError::LibraryLoad)?;

            let interface_ptr: Symbol<unsafe extern "C" fn() -> &'static PluginInterface> =
                lib.get(b"get_plugin_interface").map_err(|e| PluginError::SymbolError(format!(
                    "Failed to get plugin_interface for '{}': {:?}", path.display(), e
                )))?;
            let interface = interface_ptr();

            assert_eq!(
                std::mem::size_of::<PluginInterface>(),
                std::mem::size_of_val(&*interface),
                "ABI size mismatch"
            );

            if interface.api_version != CURRENT_API_VERSION {
                return Err(PluginError::IncompatibleAPI);
            }

            let plugin_ptr = (interface.create_plugin)();
            let mut plugin = Box::from_raw(plugin_ptr);

            plugin.init(&Value::Null).map_err(|e| PluginError::SymbolError(format!(
                "Failed to get plugin_interface for '{}': {:?}", path.display(), e
            )))?;

            let metadata = plugin.metadata();
            self.plugins.insert(metadata.name.clone(), (plugin, lib));

            Ok(())
        }
    }

    pub fn unload_plugin(&mut self, name: &str) -> Result<(), PluginError> {
        let (plugin, lib) = self.plugins.remove(name)
            .ok_or_else(|| PluginError::NotFound(format!("Plugin '{}' not loaded", name)))?;

        unsafe {
            let interface_ptr: Symbol<unsafe extern "C" fn() -> &'static PluginInterface> = lib
                .get(b"get_plugin_interface")
                .map_err(|e| PluginError::SymbolError(format!(
                    "Failed to get plugin_interface for '{}': {:?}", name, e
                )))?;
            let interface = interface_ptr();

            assert_eq!(
                std::mem::size_of::<PluginInterface>(),
                std::mem::size_of_val(&*interface),
                "ABI mismatch in unload (size {} vs {})",
                std::mem::size_of::<PluginInterface>(),
                std::mem::size_of_val(&*interface)
            );

            let raw_ptr = Box::into_raw(plugin);
            if !raw_ptr.is_null() {
                (interface.destroy_plugin)(raw_ptr);
            } else {
                return Err(PluginError::NullPointer(
                    format!("Null pointer when unloading '{}'", name)
                ));
            }

            lib.close().map_err(|e| PluginError::LibraryUnload(
                format!("Failed to close library for '{}': {:?}", name, e)
            ))?;
        }

        Ok(())
    }

    pub fn execute_plugin(&self, name: &str, input: &str) -> Result<String, String> {
        self.plugins
            .get(name)
            .map(|(p, _)| p.execute(input))
            .ok_or_else(|| format!("Plugin {} not found", name))
    }

    pub fn get_metadata(&self, name: &str) -> Option<PluginMetadata> {
        self.plugins.get(name).map(|(p, _)| p.metadata())
    }
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        let plugins = std::mem::take(&mut self.plugins);
        for (name, (plugin, lib)) in plugins.into_iter() {
            unsafe {
                if let Ok(interface) = lib.get::<PluginInterface>(b"plugin_interface") {
                    log::debug!("Dropping plugin: {}", name);
                    (interface.destroy_plugin)(Box::into_raw(plugin));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Once, OnceLock};

    static INIT: Once = Once::new();
    static PLUGIN_PATH: OnceLock<PathBuf> = OnceLock::new();

    fn init_logger() {
        INIT.call_once(|| {
            env_logger::builder()
                .filter_level(log::LevelFilter::Debug)
                .init();
        });
    }

    fn get_plugin_path() -> &'static Path {
        PLUGIN_PATH.get_or_init(|| {
            let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().join("target");

            let mut path = target_dir.join("debug").join("examples");

            #[cfg(target_os = "linux")]
            path.push("libtest_plugin.so");
            #[cfg(target_os = "macos")]
            path.push("libtest_plugin.dylib");
            #[cfg(target_os = "windows")]
            path.push("test_plugin.dll");

            assert!(path.exists(), "Plugin not found at: {}", path.display());
            path
        })
    }

    #[test]
    fn test_load_valid_plugin() {
        init_logger();
        let mut manager = PluginManager::new();
        let path = get_plugin_path();

        let result = manager.load_plugin(path);
        assert!(result.is_ok(), "Load failed: {:?}", result.err());

        let metadata = manager.get_metadata("test_plugin").unwrap();
        assert_eq!(metadata.version, "1.0");
    }

    #[test]
    fn test_load_nonexistent_library() {
        let mut manager = PluginManager::new();
        let path = Path::new("non_existent_plugin.so");

        let result = manager.load_plugin(path);
        assert!(
            matches!(result, Err(PluginError::LibraryLoad(_))),
            "Expected library load error"
        );
    }

    #[test]
    fn test_execute_plugin() {
        let mut manager = PluginManager::new();
        let path = get_plugin_path();
        manager.load_plugin(path).unwrap();

        let result = manager.execute_plugin("test_plugin", "test_input");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Processed: test_input");
    }

    #[test]
    fn test_execute_nonexistent_plugin() {
        let manager = PluginManager::new();

        let result = manager.execute_plugin("nonexistent_plugin", "input");
        assert!(
            result.is_err(),
            "Should return error for nonexistent plugin"
        );
    }

    #[test]
    fn test_unload_plugin() {
        let mut manager = PluginManager::new();
        let path = get_plugin_path();
        manager.load_plugin(path).unwrap();

        let result = manager.unload_plugin("test_plugin");
        println!("{:?}", result);
        assert!(result.is_ok(), "Unload failed");
        assert!(
            manager.get_metadata("test_plugin").is_none(),
            "Plugin metadata still present after unload"
        );
    }

    #[test]
    fn test_drop_cleanup() {
        let mut manager = PluginManager::new();
        let path = get_plugin_path();
        manager.load_plugin(path).unwrap();

        drop(manager);
    }

    #[test]
    fn test_metadata_retrieval() {
        let mut manager = PluginManager::new();
        let path = get_plugin_path();
        manager.load_plugin(path).unwrap();

        let metadata = manager.get_metadata("test_plugin").unwrap();
        assert_eq!(metadata.description, "Test Plugin");
    }

    #[test]
    fn test_reload_same_plugin() {
        let mut manager = PluginManager::new();
        let path = get_plugin_path();

        manager.load_plugin(path).unwrap();
        let first_load_count = manager.plugins.len();

        manager.load_plugin(path).unwrap();
        assert_eq!(
            manager.plugins.len(),
            first_load_count,
            "Reloading same plugin should replace existing entry"
        );
    }
}
