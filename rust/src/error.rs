// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::error::Error as StdError;

use arrow_schema::ArrowError;

#[derive(Debug)]
pub enum LanceError {
    Arrow(String),
    Schema(String),
}

pub type Result<T> = std::result::Result<T, LanceError>;

impl std::fmt::Display for LanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (catelog, message) = match self {
            Self::Arrow(s) => ("Arrow", s),
            Self::Schema(s) => ("Schema", s),
        };
        write!(f, "LanceError({}): {}", catelog, message)
    }
}

impl From<ArrowError> for LanceError {
    fn from(value: ArrowError) -> Self {
        LanceError::Arrow(value.to_string())
    }
}

impl StdError for LanceError {}
