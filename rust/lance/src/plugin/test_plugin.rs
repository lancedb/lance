// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use serde_json::Value;
use lance::plugin::{PluginInstance, PluginInterface, PluginMetadata};

pub struct TestPlugin;

impl PluginInstance for TestPlugin {
    fn init(&mut self, _: &Value) -> Result<(), String> {
        Ok(())
    }

    fn execute(&self, input: &str) -> String {
        format!("Processed: {}", input)
    }

    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "test_plugin".into(),
            version: "1.0".into(),
            description: "Test Plugin".into(),
        }
    }
}

#[no_mangle]
#[warn(improper_ctypes_definitions)]
pub extern "C" fn create() -> *mut dyn PluginInstance {
    Box::into_raw(Box::new(TestPlugin))
}

#[no_mangle]
#[warn(improper_ctypes_definitions)]
pub extern "C" fn destroy(plugin: *mut dyn PluginInstance) {
    unsafe { let _ = Box::from_raw(plugin); };
}


#[no_mangle]
#[warn(improper_ctypes_definitions)]
pub extern "C" fn get_plugin_interface() -> &'static PluginInterface {
    &PluginInterface {
        create_plugin: create,
        destroy_plugin: destroy,
        api_version: 1,
    }
}
