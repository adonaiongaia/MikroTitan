//! Linux-only: enable microphone capture (`getUserMedia`) in WebKitGTK.
//!
//! WebKitGTK differs from WKWebView and WebView2 in two ways, and both must be
//! handled or the recorder fails with `NotAllowedError` on Linux only:
//!
//! * `enable-media-stream` is off by default, so `navigator.mediaDevices`
//!   never exposes a working `getUserMedia`; and
//! * WebKit's default `permission-request` handler denies every request.
//!
//! This module reaches the underlying `webkit2gtk::WebView` through Tauri's
//! `with_webview`, enables media streams, and installs a deny-by-default
//! handler that allows **audio-only** capture from a trusted app origin. A
//! document that somehow navigates the webview elsewhere can never inherit
//! silent microphone access.

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PROD_ORIGIN: &str = "tauri://localhost";
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const PROD_HTTP_ORIGIN: &str = "http://tauri.localhost";
#[cfg(debug_assertions)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const DEV_ORIGIN: &str = "http://localhost:1420";

/// Exact-origin matching, with a path allowed after the origin. This rejects
/// look-alikes such as `tauri://localhost.evil.example` and `:14200`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn is_trusted_media_origin(uri: &str) -> bool {
    fn matches(uri: &str, origin: &str) -> bool {
        uri == origin
            || uri
                .strip_prefix(origin)
                .is_some_and(|rest| rest.starts_with('/'))
    }

    if matches(uri, PROD_ORIGIN) || matches(uri, PROD_HTTP_ORIGIN) {
        return true;
    }
    #[cfg(debug_assertions)]
    if matches(uri, DEV_ORIGIN) {
        return true;
    }
    false
}

/// Allow microphone capture for the main webview. A no-op on other platforms,
/// whose webviews route media permission through the operating system.
#[cfg(target_os = "linux")]
pub fn enable_media_capture<R: tauri::Runtime>(webview: &tauri::Webview<R>) {
    use webkit2gtk::{
        glib::prelude::Cast, PermissionRequestExt, SettingsExt,
        UserMediaPermissionRequest, UserMediaPermissionRequestExt, WebViewExt,
    };

    let result = webview.with_webview(|platform_webview| {
        let view = platform_webview.inner();
        if let Some(settings) = WebViewExt::settings(&view) {
            settings.set_enable_media_stream(true);
        }

        view.connect_permission_request(|view, request| {
            let Some(media) = request.downcast_ref::<UserMediaPermissionRequest>() else {
                // Leave geolocation, notifications and other requests to
                // WebKit's own deny-by-default handling.
                return false;
            };
            let uri = view.uri().map(|value| value.to_string()).unwrap_or_default();
            if media.is_for_audio_device() && !media.is_for_video_device()
                && is_trusted_media_origin(&uri)
            {
                media.allow();
            } else {
                media.deny();
            }
            true
        });
    });

    if let Err(error) = result {
        eprintln!("MikroTitan: could not enable WebKitGTK microphone capture: {error}");
    }
}

#[cfg(not(target_os = "linux"))]
pub fn enable_media_capture<R: tauri::Runtime>(_webview: &tauri::Webview<R>) {}

#[cfg(test)]
mod tests {
    use super::is_trusted_media_origin;

    #[test]
    fn allows_production_origins() {
        assert!(is_trusted_media_origin("tauri://localhost"));
        assert!(is_trusted_media_origin("tauri://localhost/index.html"));
        assert!(is_trusted_media_origin("http://tauri.localhost"));
        assert!(is_trusted_media_origin("http://tauri.localhost/chat/"));
    }

    #[test]
    fn rejects_look_alike_and_foreign_origins() {
        assert!(!is_trusted_media_origin(""));
        assert!(!is_trusted_media_origin("https://evil.example.com"));
        assert!(!is_trusted_media_origin("tauri://localhost.evil.example"));
        assert!(!is_trusted_media_origin("tauri://localhostfoo"));
        assert!(!is_trusted_media_origin("http://tauri.localhost.evil.example"));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn allows_the_vite_origin_in_debug_only() {
        assert!(is_trusted_media_origin("http://localhost:1420"));
        assert!(is_trusted_media_origin("http://localhost:1420/"));
        assert!(!is_trusted_media_origin("http://localhost:14200"));
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn rejects_the_vite_origin_in_release() {
        assert!(!is_trusted_media_origin("http://localhost:1420"));
    }
}
