// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ffi::{c_char, CStr, CString};
use serde_json::Value;
use lance_plugin::plugin::{PluginInstance, PluginMetadata};

#[repr(C)]
pub struct TestPlugin;

impl PluginInstance for TestPlugin {
    fn init(&mut self, _: &Value) -> Result<(), String> {
        Ok(())
    }

    fn execute(&self, input: Value) -> Result<Value, String> {
        let input_str = input["input"].as_str().unwrap_or("");
        Ok(serde_json::json!({
        "result": format!("Processed: {}", input_str)
    }))
    }

    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "demo_plugin".into(),
            version: "1.0".into(),
            description: "Test Plugin".into(),
        }
    }
}

type CreateFunc = extern "C" fn() -> *mut dyn PluginInstance;
type DestroyFunc = extern "C" fn(*mut dyn PluginInstance);

#[no_mangle]
pub extern "C" fn create() -> *mut dyn PluginInstance {
    Box::into_raw(Box::new(TestPlugin) as Box<dyn PluginInstance>)
}

#[no_mangle]
pub extern "C" fn destroy(plugin: *mut dyn PluginInstance) {
    unsafe { let _ = Box::from_raw(plugin); };
}

#[no_mangle]
pub extern "C" fn execute(plugin: *mut dyn PluginInstance, input: *const c_char) -> *const c_char {
    let input_cstr = unsafe { CStr::from_ptr(input) };
    let input_str = input_cstr.to_str().unwrap();
    let input_value = serde_json::from_str(input_str).unwrap();

    let plugin_ref = unsafe { &*plugin };
    let result = plugin_ref.execute(input_value).unwrap();
    let result_str = serde_json::to_string(&result).unwrap();

    CString::new(result_str).unwrap().into_raw()
}

#[repr(C)]
pub struct PluginInterface {
    create_plugin: CreateFunc,
    destroy_plugin: DestroyFunc,
    execute_plugin: extern "C" fn(*mut dyn PluginInstance, *const c_char) -> *const c_char,
    api_version: u32,
}

#[no_mangle]
pub extern "C" fn get_plugin_interface() -> &'static PluginInterface {
    &PluginInterface {
        create_plugin: create,
        destroy_plugin: destroy,
        execute_plugin: execute,
        api_version: 1,
    }
}