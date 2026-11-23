//! Sandboxed live-artifact preview origin.
//!
//! A `srcdoc` frame inherits the webview's CSP, and the app's `script-src
//! 'self'` would block every generated inline script. The preview document is
//! therefore served from this dedicated scheme with its own policy, and the
//! frontend keeps the iframe sandboxed with `allow-scripts` only (never
//! `allow-same-origin`). The frame gets an opaque origin with no cookies,
//! storage, parent DOM or Tauri IPC, and this policy blocks external requests
//! unless the user explicitly turns the network on.
//!
//! The network permission is stored with the token, never read from the URL.
//! A generated script can navigate its own frame, so a query such as `?net=1`
//! must not be able to grant itself network access.
//!
//! Documents are registered ahead of time by the pane and addressed by an
//! unguessable token; the scheme never serves arbitrary files.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use tauri::http::{header, HeaderValue, Request, Response, StatusCode};

use crate::error::{AppError, AppResult};

pub const SCHEME: &str = "janas-artifact";
const MAX_DOCUMENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_TOTAL_BYTES: usize = 32 * 1024 * 1024;
const MAX_DOCUMENTS: usize = 32;

const STRICT_CSP: &str = "default-src 'none'; script-src 'unsafe-inline'; \
style-src 'unsafe-inline'; img-src data: blob:; font-src data:; \
media-src data: blob:; connect-src 'none'; form-action 'none'; \
base-uri 'none'; object-src 'none'; worker-src 'none'";

const NETWORK_CSP: &str = "default-src 'none'; \
script-src 'unsafe-inline' https: http:; \
style-src 'unsafe-inline' https: http:; \
img-src data: blob: https: http:; font-src data: https: http:; \
media-src data: blob: https: http:; connect-src https: http:; \
form-action 'none'; base-uri 'none'; object-src 'none'; worker-src 'none'";

struct FrameDocument {
    html: String,
    allow_network: bool,
}

#[derive(Default)]
struct FrameStore {
    documents: HashMap<String, FrameDocument>,
    order: VecDeque<String>,
    total_bytes: usize,
}

/// In-memory documents for the artifact scheme. Both the number of documents
/// and their total size are bounded, so a long stream of large revisions
/// cannot grow memory without limit.
#[derive(Default)]
pub struct ArtifactFrames {
    store: Mutex<FrameStore>,
}

impl ArtifactFrames {
    pub fn register(&self, html: String, allow_network: bool) -> AppResult<String> {
        if html.len() > MAX_DOCUMENT_BYTES {
            return Err(AppError::message(format!(
                "This preview is larger than the {} MiB sandbox limit.",
                MAX_DOCUMENT_BYTES / (1024 * 1024)
            )));
        }
        let token = uuid::Uuid::new_v4().to_string();
        let mut store = self
            .store
            .lock()
            .map_err(|_| AppError::message("The artifact preview store is unavailable."))?;
        let size = html.len();
        store.documents.insert(
            token.clone(),
            FrameDocument {
                html,
                allow_network,
            },
        );
        store.order.push_back(token.clone());
        store.total_bytes = store.total_bytes.saturating_add(size);
        while store.order.len() > MAX_DOCUMENTS || store.total_bytes > MAX_TOTAL_BYTES {
            let Some(oldest) = store.order.pop_front() else {
                break;
            };
            if let Some(document) = store.documents.remove(&oldest) {
                store.total_bytes = store.total_bytes.saturating_sub(document.html.len());
            }
        }
        Ok(token)
    }

    fn document(&self, token: &str) -> Option<FrameDocument> {
        let store = self.store.lock().ok()?;
        store.documents.get(token).map(|document| FrameDocument {
            html: document.html.clone(),
            allow_network: document.allow_network,
        })
    }
}

fn text_response(status: StatusCode, body: Vec<u8>) -> Response<Vec<u8>> {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    response
}

fn frame_headers(response: &mut Response<Vec<u8>>, csp: &str) {
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    if let Ok(value) = HeaderValue::from_str(csp) {
        headers.insert(header::CONTENT_SECURITY_POLICY, value);
    }
}

/// Serve a registered document for `GET /frame/<token>`. Unknown tokens are
/// 404 so the scheme cannot be used as a local file proxy. The query string is
/// deliberately ignored: the stored token decides whether network is allowed.
pub fn respond(frames: &ArtifactFrames, request: Request<Vec<u8>>) -> Response<Vec<u8>> {
    if request.method() != tauri::http::Method::GET {
        return text_response(StatusCode::METHOD_NOT_ALLOWED, Vec::new());
    }
    let token = request
        .uri()
        .path()
        .strip_prefix("/frame/")
        .unwrap_or_default();
    if token.is_empty() || token.contains('/') || token.contains("..") {
        return text_response(StatusCode::NOT_FOUND, Vec::new());
    }
    let Some(document) = frames.document(token) else {
        return text_response(StatusCode::NOT_FOUND, Vec::new());
    };
    let mut response = text_response(StatusCode::OK, document.html.into_bytes());
    frame_headers(
        &mut response,
        if document.allow_network {
            NETWORK_CSP
        } else {
            STRICT_CSP
        },
    );
    response
}

/// The protocol can theoretically be requested before `setup` registers the
/// store; answer with a clear, inert response instead of panicking.
pub fn unavailable() -> Response<Vec<u8>> {
    let mut response = text_response(
        StatusCode::SERVICE_UNAVAILABLE,
        b"Artifact preview is not ready.".to_vec(),
    );
    frame_headers(&mut response, STRICT_CSP);
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(path: &str) -> Request<Vec<u8>> {
        Request::builder()
            .method("GET")
            .uri(path)
            .body(Vec::new())
            .expect("request")
    }

    fn csp(response: &Response<Vec<u8>>) -> &str {
        response
            .headers()
            .get(header::CONTENT_SECURITY_POLICY)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
    }

    #[test]
    fn a_strict_token_ignores_a_network_query() {
        let frames = ArtifactFrames::default();
        let token = frames.register("<p>safe</p>".into(), false).unwrap();
        let response = respond(&frames, get(&format!("/frame/{token}?net=1")));
        assert_eq!(response.status(), StatusCode::OK);
        assert!(csp(&response).contains("connect-src 'none'"));
        assert!(!csp(&response).contains("connect-src https:"));
    }

    #[test]
    fn a_network_token_serves_the_network_policy() {
        let frames = ArtifactFrames::default();
        let token = frames.register("<p>online</p>".into(), true).unwrap();
        let response = respond(&frames, get(&format!("/frame/{token}")));
        assert_eq!(response.status(), StatusCode::OK);
        assert!(csp(&response).contains("connect-src https: http:"));
    }

    #[test]
    fn unknown_tokens_and_paths_are_404() {
        let frames = ArtifactFrames::default();
        assert_eq!(
            respond(&frames, get("/frame/not-a-token")).status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            respond(&frames, get("/other")).status(),
            StatusCode::NOT_FOUND
        );
    }

    #[test]
    fn the_oldest_document_is_evicted_at_the_count_limit() {
        let frames = ArtifactFrames::default();
        let first = frames.register("<p>first</p>".into(), false).unwrap();
        for _ in 1..MAX_DOCUMENTS {
            frames.register("<p>x</p>".into(), false).unwrap();
        }
        frames.register("<p>new</p>".into(), false).unwrap();
        assert_eq!(
            respond(&frames, get(&format!("/frame/{first}"))).status(),
            StatusCode::NOT_FOUND
        );
    }

    #[test]
    fn oversized_documents_are_rejected() {
        let frames = ArtifactFrames::default();
        let oversized = "x".repeat(MAX_DOCUMENT_BYTES + 1);
        assert!(frames.register(oversized, false).is_err());
    }
}
