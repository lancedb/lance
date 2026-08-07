// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An in-process stand-in for the list API of a cloud object store.
//!
//! The unit tests in [`read_dir`](super) drive a fake that implements
//! [`PaginatedDirLister`](super::PaginatedDirLister) directly, which leaves everything between
//! this crate and the network untested: the query parameters `object_store` builds, the XML it
//! parses, and whether a key survives a round trip through a URL. This serves the wire protocol
//! instead, so the real S3, GCS and Azure clients can be pointed at it without credentials,
//! behaving however [`StoreModel`] is told to behave.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::{Query, State};
use axum::http::header;
use axum::response::IntoResponse;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

use super::store_model::{LevelPage, Resume, StoreModel};

/// Every entry the emulator reports carries this timestamp, in each wire format's spelling.
const LAST_MODIFIED_ISO: &str = "2025-01-01T00:00:00.000Z";
const LAST_MODIFIED_RFC1123: &str = "Wed, 01 Jan 2025 00:00:00 GMT";

/// Which wire format the emulator speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wire {
    /// The S3 XML API, which the S3 and GCS clients both use.
    S3,
    /// Azure's list-blobs XML, whose parameters and element names are its own.
    Azure,
}

struct EmulatorState {
    model: StoreModel,
    wire: Wire,
    requests: AtomicUsize,
    /// The page limit each request asked for, in order, so a test can tell a listing that
    /// pushed its page size down from one that asked for the whole directory.
    limits: Mutex<Vec<Option<usize>>>,
}

/// A running emulator. Listening stops when this is dropped.
pub struct ListEmulator {
    url: String,
    state: Arc<EmulatorState>,
    server: JoinHandle<()>,
}

impl Drop for ListEmulator {
    fn drop(&mut self) {
        self.server.abort();
    }
}

impl ListEmulator {
    pub async fn start(model: StoreModel, wire: Wire) -> Self {
        let state = Arc::new(EmulatorState {
            model,
            wire,
            requests: AtomicUsize::new(0),
            limits: Mutex::new(Vec::new()),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new().fallback(list).with_state(state.clone());
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Self { url, state, server }
    }

    /// The endpoint to point a client at. The bucket or container is the first path segment
    /// and is ignored: the emulator holds one flat key space.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// How many list requests the emulator has answered.
    pub fn requests(&self) -> usize {
        self.state.requests.load(Ordering::SeqCst)
    }

    /// The page limit each request asked for, in the order the requests arrived.
    pub fn limits(&self) -> Vec<Option<usize>> {
        self.state.limits.lock().unwrap().clone()
    }
}

async fn list(
    State(state): State<Arc<EmulatorState>>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    state.requests.fetch_add(1, Ordering::SeqCst);

    let (offset_param, limit_param, token_param) = match state.wire {
        Wire::S3 => ("start-after", "max-keys", "continuation-token"),
        Wire::Azure => ("startFrom", "maxresults", "marker"),
    };
    let prefix = params.get("prefix").map(String::as_str).unwrap_or_default();
    // A listing resumes from the store's own token and nothing else. What an offset excludes
    // is the store's choice — S3 excludes the position, Azure includes it, Azurite drops it —
    // so sending one would make the listing depend on which store answered.
    assert!(
        !params.contains_key(offset_param),
        "a listing must not resume by offset, but sent {offset_param}"
    );
    let resume = match params.get(token_param) {
        Some(token) => Resume::Token(token),
        None => Resume::Start,
    };
    let max_keys = params.get(limit_param).and_then(|keys| keys.parse().ok());
    state.limits.lock().unwrap().push(max_keys);

    let page = state.model.list_level(prefix, resume, max_keys);
    let body = match state.wire {
        Wire::S3 => s3_xml(&page),
        Wire::Azure => azure_xml(&page),
    };
    ([(header::CONTENT_TYPE, "application/xml")], body)
}

fn s3_xml(page: &LevelPage) -> String {
    let mut body = String::from(r#"<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>"#);
    for prefix in &page.prefixes {
        let prefix = escape(prefix);
        body.push_str(&format!(
            "<CommonPrefixes><Prefix>{prefix}</Prefix></CommonPrefixes>"
        ));
    }
    for key in &page.objects {
        let key = escape(key);
        body.push_str(&format!(
            "<Contents><Key>{key}</Key><Size>1</Size>\
             <LastModified>{LAST_MODIFIED_ISO}</LastModified><ETag>\"1\"</ETag></Contents>"
        ));
    }
    body.push_str(&format!("<IsTruncated>{}</IsTruncated>", page.truncated));
    if let Some(token) = &page.next_token {
        let token = escape(token);
        body.push_str(&format!(
            "<NextContinuationToken>{token}</NextContinuationToken>"
        ));
    }
    body.push_str("</ListBucketResult>");
    body
}

fn azure_xml(page: &LevelPage) -> String {
    let mut body =
        String::from(r#"<?xml version="1.0" encoding="utf-8"?><EnumerationResults><Blobs>"#);
    for prefix in &page.prefixes {
        let prefix = escape(prefix);
        body.push_str(&format!("<BlobPrefix><Name>{prefix}</Name></BlobPrefix>"));
    }
    for key in &page.objects {
        let key = escape(key);
        body.push_str(&format!(
            "<Blob><Name>{key}</Name><Properties>\
             <Last-Modified>{LAST_MODIFIED_RFC1123}</Last-Modified>\
             <Content-Length>1</Content-Length>\
             <Content-Type>application/octet-stream</Content-Type>\
             <Etag>1</Etag></Properties></Blob>"
        ));
    }
    body.push_str("</Blobs>");
    if let Some(token) = &page.next_token {
        let token = escape(token);
        body.push_str(&format!("<NextMarker>{token}</NextMarker>"));
    }
    body.push_str("</EnumerationResults>");
    body
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
