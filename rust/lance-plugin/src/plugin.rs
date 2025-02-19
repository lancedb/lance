// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
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
    fn execute(&self, input: Value) -> Result<Value, String>;
    fn metadata(&self) -> PluginMetadata;
}

#[repr(C)]
pub struct PluginInterface {
    pub create_plugin: unsafe extern "C" fn() -> *mut dyn PluginInstance,
    pub destroy_plugin: unsafe extern "C" fn(*mut dyn PluginInstance),
    pub execute_plugin: unsafe extern "C" fn(*mut dyn PluginInstance, *const c_char) -> *const c_char,
    pub api_version: u32,
}

pub struct PluginManager {
    plugins: HashMap<String, (*mut dyn PluginInstance, Library, &'static PluginInterface)>,
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
                "Failed to initialize plugin: {}", e
            )))?;

            let metadata = plugin.metadata();
            let raw_ptr = Box::into_raw(plugin);
            self.plugins.insert(metadata.name.clone(), (raw_ptr, lib, interface));

            Ok(())
        }
    }

    pub fn unload_plugin(&mut self, name: &str) -> Result<(), PluginError> {
        let (ptr, lib, interface) = self.plugins.remove(name)
            .ok_or_else(|| PluginError::NotFound(format!("Plugin '{}' not loaded", name)))?;

        unsafe {
            if !ptr.is_null() {
                (interface.destroy_plugin)(ptr);
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

    pub fn execute_plugin(&self, name: &str, input: &Value) -> Result<Value, String> {
        let (ptr, _, interface) = self.plugins.get(name)
            .ok_or_else(|| format!("Plugin {} not found", name))?;

        unsafe {
            let input_str = serde_json::to_string(input).map_err(|e| e.to_string())?;
            let c_input = CString::new(input_str).map_err(|e| e.to_string())?;

            let c_output = (interface.execute_plugin)(*ptr, c_input.as_ptr());

            let output_str = CStr::from_ptr(c_output)
                .to_str()
                .map_err(|e| e.to_string())?;

            serde_json::from_str(output_str).map_err(|e| e.to_string())
        }
    }

    pub fn get_metadata(&self, name: &str) -> Option<PluginMetadata> {
        self.plugins.get(name).map(|(ptr, _, _)| {
            let plugin = unsafe { &**ptr };
            plugin.metadata()
        })
    }
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        for (name, (ptr, lib, interface)) in self.plugins.drain() {
            unsafe {
                if !ptr.is_null() {
                    (interface.destroy_plugin)(ptr);
                }
                lib.close().unwrap_or_else(|e| {
                    log::error!("Failed to close library for {}: {:?}", name, e);
                });
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
            path.push("libdemo_plugin.so");
            #[cfg(target_os = "macos")]
            path.push("libdemo_plugin.dylib");
            #[cfg(target_os = "windows")]
            path.push("demo_plugin.dll");

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

        let metadata = manager.get_metadata("demo_plugin").unwrap();
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

        let input = serde_json::json!({"input": "test"});
        let result = manager.execute_plugin("demo_plugin", &input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), serde_json::json!({"result": "Processed: test"}));
    }

    #[test]
    fn test_execute_nonexistent_plugin() {
        let manager = PluginManager::new();

        let input = serde_json::json!({"input": "test"});
        let result = manager.execute_plugin("nonexistent_plugin", &input);
        assert!(
            result.is_err(),
            "Should return error for nonexistent plugin"
        );
    }

    #[test]
    fn test_unload_plugin() {
        let mut manager = PluginManager::new();
        let path = get_plugin_path();
        manager.load_plugin(path);

        let result = manager.unload_plugin("demo_plugin");
        println!("{:?}", result);
        assert!(result.is_ok(), "Unload failed");
        assert!(
            manager.get_metadata("demo_plugin").is_none(),
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

        let metadata = manager.get_metadata("demo_plugin").unwrap();
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


    fn get_plugins_dir_path() -> &'static Path {
        PLUGIN_PATH.get_or_init(|| {
            let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let project_root = manifest_dir
                .parent().unwrap();

            let mut path = project_root.join("lance").join("plugins");

            #[cfg(target_os = "linux")]
            path.push("libdemo_plugin.so");
            #[cfg(target_os = "macos")]
            path.push("libdemo_plugin.dylib");
            #[cfg(target_os = "windows")]
            path.push("demo_plugin.dll");

            assert!(
                path.exists(),
                "Plugin not found in plugins dir: {}\nHINT: Run `cargo build --example demo_plugin --features demo-plugin` first",
                path.display()
            );
            path
        })
    }

    #[test]
    fn test_load_from_plugins_dir() {
        init_logger();
        let mut manager = PluginManager::new();
        let path = get_plugins_dir_path();

        let load_result = manager.load_plugin(path);
        assert!(load_result.is_ok(), "Failed to load from plugins dir: {:?}", load_result);

        let metadata = manager.get_metadata("demo_plugin").unwrap();
        assert_eq!(metadata.version, "1.0");
        assert_eq!(metadata.description, "Test Plugin");

        let input = serde_json::json!({"input": "from_plugins_dir"});
        let output = manager.execute_plugin("demo_plugin", &input).unwrap();
        assert_eq!(
            output,
            serde_json::json!({"result": "Processed: from_plugins_dir"}),
            "Execution result mismatch"
        );

        let unload_result = manager.unload_plugin("demo_plugin");
        assert!(unload_result.is_ok(), "Unload failed: {:?}", unload_result);
    }
}
