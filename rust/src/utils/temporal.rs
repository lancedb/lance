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

//! These utilities make it possible to mock out the "current time" when running
//! unit tests.  Anywhere in production code where we need to get the current time
//! we should use the below methods and types instead of the builtin methods and types

use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
#[cfg(test)]
use mock_instant::{SystemTime as NativeSystemTime, UNIX_EPOCH};

#[cfg(not(test))]
use std::time::{SystemTime as NativeSystemTime, UNIX_EPOCH};

pub type SystemTime = NativeSystemTime;

/// Mirror function that mimics DateTime<Utc>::now() with the exception that it
/// uses the potentially mocked system time.
pub fn utc_now() -> DateTime<Utc> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before Unix epoch");
    let naive =
        NaiveDateTime::from_timestamp_opt(now.as_secs() as i64, now.subsec_nanos()).unwrap();
    Utc.from_utc_datetime(&naive)
}
