pub struct EnvVarGuard {
    key: String,
    original_value: Option<String>,
}

impl EnvVarGuard {
    pub fn new(key: &str, new_value: &str) -> Self {
        let original_value = std::env::var(key).ok();
        std::env::set_var(key, new_value);
        Self {
            key: key.to_string(),
            original_value,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(ref value) = self.original_value {
            std::env::set_var(&self.key, value);
        } else {
            std::env::remove_var(&self.key);
        }
    }
}
