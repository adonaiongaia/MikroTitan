//! Headless end-to-end coverage for Agent Mode.
//!
//! These tests drive the real loop in `agent.rs` against a fake
//! OpenAI-compatible server over loopback TCP. No webview, no model download
//! and no Tauri runtime are involved: approvals come from a driver thread and
//! progress is captured through `AgentEvents`.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use serde_json::{json, Value};

use super::*;

// ---------------------------------------------------------------------------
// Fake OpenAI-compatible engine
// ---------------------------------------------------------------------------

struct FakeEngine {
    base_url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl FakeEngine {
    fn start(responder: impl Fn(&Value) -> String + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake engine");
        let address = listener.local_addr().expect("fake engine address");
        listener
            .set_nonblocking(true)
            .expect("nonblocking fake engine");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let responder = Arc::new(responder);
        let captured = requests.clone();
        let flag = stop.clone();
        let handle = thread::spawn(move || {
            while !flag.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let response = match read_json_request(&mut stream) {
                            Some(request) => {
                                let body = responder(&request);
                                if let Ok(mut guard) = captured.lock() {
                                    guard.push(request);
                                }
                                body
                            }
                            None => json!({ "error": "bad request" }).to_string(),
                        };
                        write_response(&mut stream, &response);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            base_url: format!("http://{address}"),
            requests,
            stop,
            handle: Some(handle),
        }
    }

    fn base_url(&self) -> String {
        self.base_url.clone()
    }

    fn requests(&self) -> Vec<Value> {
        self.requests
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }
}

impl Drop for FakeEngine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn read_json_request(stream: &mut std::net::TcpStream) -> Option<Value> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .ok()?;
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let read = stream.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        let header_end = find_bytes(&buffer, b"\r\n\r\n")?;
        let head = String::from_utf8_lossy(&buffer[..header_end]);
        let length = head
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                if key.eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let body_start = header_end + 4;
        if buffer.len() >= body_start + length {
            return serde_json::from_slice(&buffer[body_start..body_start + length]).ok();
        }
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn write_response(stream: &mut std::net::TcpStream, body: &str) {
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

fn sse(chunks: &[Value]) -> String {
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

fn sse_tool(name: &str, arguments: &str, id: &str) -> String {
    sse(&[
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]}}]}),
        json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
    ])
}

fn sse_text(text: &str) -> String {
    sse(&[
        json!({"choices":[{"delta":{"content":text}}]}),
        json!({"choices":[{"delta":{},"finish_reason":"stop"}]}),
    ])
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn workspace(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "janas-agent-e2e-{label}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create workspace");
    root
}

fn request_with(workspace: &Path, policy: ApprovalPolicy) -> AgentRequest {
    AgentRequest {
        run_id: format!("e2e-{}", std::process::id()),
        conversation_id: "e2e".into(),
        goal: "Inspect the workspace and report what you find.".into(),
        history: Vec::new(),
        system_prompt: None,
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        min_p: None,
        repeat_penalty: None,
        thinking: Some(false),
        max_tokens: Some(256),
        max_iterations: 6,
        approval_policy: policy,
        workspace_root: workspace.to_string_lossy().to_string(),
        planning_step: false,
        auto_compact: false,
        compaction_threshold: 0.75,
        web_search: false,
        verify_command: String::new(),
        mcp_servers: Vec::new(),
        rag_collections: Vec::new(),
        rag_top_k: None,
        rag_max_tokens: None,
    }
}

fn execute_headless(
    request: AgentRequest,
    base_url: String,
    context_window: usize,
    control: Arc<RunControl>,
    events: AgentEvents,
    journal: &RunJournal,
) -> Result<AgentFinish, AgentFailure> {
    execute_headless_with_recorder(
        request,
        base_url,
        context_window,
        control,
        events,
        journal,
        None,
    )
}

fn execute_headless_with_recorder(
    request: AgentRequest,
    base_url: String,
    context_window: usize,
    control: Arc<RunControl>,
    events: AgentEvents,
    journal: &RunJournal,
    recorder: Option<Arc<RunRecorder>>,
) -> Result<AgentFinish, AgentFailure> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    let http = reqwest::Client::new();
    let config = AgentRuntimeConfig {
        base_url,
        api_key: String::new(),
        model: "fake-model".into(),
        context_window,
    };
    runtime.block_on(async {
        let context = RunContext {
            events: &events,
            control: control.as_ref(),
            journal,
            http: &http,
            config: &config,
            request: &request,
            run_id: &request.run_id,
            conversation_id: &request.conversation_id,
            recorder,
            knowledge: None,
            knowledge_sources: Default::default(),
        };
        execute(&context).await
    })
}

/// Stops a helper thread when the test ends.
struct Driver {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for Driver {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Plays the user: approves every approval card exactly once.
fn approve_when_requested(events: AgentEvents, control: Arc<RunControl>) -> Driver {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = thread::spawn(move || {
        let mut approved = HashSet::new();
        while !flag.load(Ordering::SeqCst) {
            for event in events.captured() {
                if event.name != "agent:approval_needed" {
                    continue;
                }
                let Some(step_id) = event.payload.get("step_id").and_then(Value::as_str) else {
                    continue;
                };
                if approved.insert(step_id.to_string()) {
                    let _ = control.resolve(step_id, Resolution::Approved);
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
    });
    Driver {
        stop,
        handle: Some(handle),
    }
}

/// Plays the user: hits Abort as soon as the approval card appears.
fn abort_when_requested(events: AgentEvents, control: Arc<RunControl>) -> Driver {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = thread::spawn(move || {
        while !flag.load(Ordering::SeqCst) {
            if events
                .captured()
                .iter()
                .any(|event| event.name == "agent:approval_needed")
            {
                control.cancel();
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
    });
    Driver {
        stop,
        handle: Some(handle),
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

#[test]
fn read_only_calls_skip_approval_and_duplicates_are_blocked() {
    let root = workspace("approval");
    std::fs::write(root.join("note.txt"), "hello agent\n").expect("write note");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |request| {
        assert_eq!(
            request.get("stream").and_then(Value::as_bool),
            Some(true),
            "agent turns must stream"
        );
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            0 => sse_tool("read_file", r#"{"path":"note.txt"}"#, "call_1"),
            1 => sse_tool("read_file", r#"{"path":"note.txt"}"#, "call_2"),
            _ => sse_text("The note says hello agent."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let _approvals = approve_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();

    let mut request = request_with(&root, ApprovalPolicy::All);
    request.max_iterations = 5;
    let finish = execute_headless(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
    )
    .expect("agent run should finish");

    assert_eq!(finish.status, "completed");
    assert!(finish.final_content.contains("hello agent"));

    let captured = events.captured();
    let approvals: Vec<_> = captured
        .iter()
        .filter(|event| event.name == "agent:approval_needed")
        .collect();
    assert!(
        approvals.is_empty(),
        "read-only tools are blanket-approved even under the strict policy"
    );

    assert_eq!(
        captured
            .iter()
            .filter(|event| event.name == "agent:tool_start")
            .count(),
        2,
        "the duplicate call is announced but never approved or executed"
    );

    let results: Vec<_> = captured
        .iter()
        .filter(|event| event.name == "agent:tool_result")
        .collect();
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|event| {
        event.payload["ok"] == true
            && event.payload["output"]
                .as_str()
                .unwrap_or_default()
                .contains("hello agent")
    }));
    assert!(results.iter().any(|event| {
        event.payload["blocked"] == true
            && event.payload["output"]
                .as_str()
                .unwrap_or_default()
                .contains("repeated exactly")
    }));
    assert_eq!(
        engine.requests().len(),
        3,
        "read, duplicate retry and final answer"
    );
}

#[test]
fn claimed_change_without_a_tool_is_retried_once() {
    let root = workspace("claim");
    std::fs::write(root.join("note.txt"), "before\n").expect("write note");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |_request| {
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            0 => sse_text("Done. I've written 'after' to note.txt."),
            1 => sse_tool(
                "write_file",
                r#"{"path":"note.txt","content":"after\n"}"#,
                "call_1",
            ),
            _ => sse_text("The file now contains after."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let _approvals = approve_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();
    let mut request = request_with(&root, ApprovalPolicy::All);
    request.max_iterations = 5;

    let finish = execute_headless(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
    )
    .expect("the forced retry should complete the run");

    assert_eq!(finish.status, "completed");
    assert_eq!(
        std::fs::read_to_string(root.join("note.txt")).expect("read note"),
        "after\n"
    );
    assert!(events.captured().iter().any(|event| {
        event.name == "agent:notice"
            && event.payload["message"]
                .as_str()
                .unwrap_or_default()
                .contains("without calling a tool")
    }));
    assert_eq!(
        engine.requests().len(),
        3,
        "claim, forced tool attempt and final answer"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn invalid_tool_arguments_are_reported_and_repaired() {
    let root = workspace("schema");
    std::fs::write(root.join("note.txt"), "before\n").expect("write note");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |_request| {
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            // Missing the required `content` field.
            0 => sse_tool("write_file", r#"{"path":"note.txt"}"#, "call_1"),
            1 => sse_tool(
                "write_file",
                r#"{"path":"note.txt","content":"fixed\n"}"#,
                "call_2",
            ),
            _ => sse_text("The file now contains fixed."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let _approvals = approve_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();
    let mut request = request_with(&root, ApprovalPolicy::All);
    request.max_iterations = 5;

    let finish = execute_headless(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
    )
    .expect("the repaired call should complete the run");

    assert_eq!(finish.status, "completed");
    assert_eq!(
        std::fs::read_to_string(root.join("note.txt")).expect("read note"),
        "fixed\n"
    );
    let captured = events.captured();
    assert!(captured.iter().any(|event| {
        event.name == "agent:tool_result"
            && event.payload["ok"] == false
            && event.payload["output"]
                .as_str()
                .unwrap_or_default()
                .contains("missing the required argument")
    }));
    let approvals = captured
        .iter()
        .filter(|event| event.name == "agent:approval_needed")
        .count();
    assert_eq!(approvals, 1, "invalid arguments must not prompt for approval");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn compaction_runs_a_tool_free_summary_turn() {
    let root = workspace("compaction");
    let big = (1..=80)
        .map(|line| format!("line {line} of the big file"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(root.join("big.txt"), big).expect("write big file");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |request| {
        if request.get("stream").and_then(Value::as_bool) == Some(false) {
            assert!(
                request.get("tools").is_none(),
                "the summariser must not receive tool schemas"
            );
            assert!(
                request
                    .get("max_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    <= 512
            );
            return json!({"choices":[{"message":{"content":"- inspected big.txt in slices\n- older steps compacted"}}]})
                .to_string();
        }
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            turn @ 0..=3 => {
                let first = turn * 20 + 1;
                let last = first + 19;
                let arguments =
                    format!(r#"{{"path":"big.txt","start_line":{first},"end_line":{last}}}"#);
                sse_tool("read_file", &arguments, &format!("call_{turn}"))
            }
            _ => sse_text("Compaction kept the run going."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let journal = RunJournal::default();
    let mut request = request_with(&root, ApprovalPolicy::Auto);
    request.auto_compact = true;
    request.compaction_threshold = 0.5;
    request.max_iterations = 8;

    let finish = execute_headless(
        request,
        engine.base_url(),
        2_048,
        control,
        events.clone(),
        &journal,
    )
    .expect("agent run should finish");

    assert_eq!(finish.status, "completed");
    assert_eq!(finish.final_content, "Compaction kept the run going.");

    let captured = events.captured();
    let compactions: Vec<_> = captured
        .iter()
        .filter(|event| event.name == "agent:context_compacted")
        .collect();
    assert!(
        !compactions.is_empty(),
        "expected a compaction event; captured {captured:#?}"
    );
    assert!(
        compactions[0].payload["dropped_messages"]
            .as_u64()
            .unwrap_or(0)
            >= 2
    );
    assert!(
        compactions[0].payload["summary"]
            .as_str()
            .unwrap_or_default()
            .contains("compacted")
    );

    let requests = engine.requests();
    assert!(requests.iter().any(|request| {
        request.get("stream") == Some(&Value::Bool(false)) && request.get("tools").is_none()
    }));
}

#[test]
fn file_edits_preview_and_undo() {
    let root = workspace("undo");
    std::fs::write(root.join("note.txt"), "before\n").expect("write note");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |request| {
        assert_eq!(
            request.get("stream").and_then(Value::as_bool),
            Some(true),
            "agent turns must stream"
        );
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            0 => sse_tool(
                "write_file",
                r#"{"path":"note.txt","content":"after\n"}"#,
                "call_1",
            ),
            1 => sse_tool(
                "write_file",
                r#"{"path":"new.txt","content":"created\n"}"#,
                "call_2",
            ),
            _ => sse_text("Wrote both files."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let _approvals = approve_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();

    let mut request = request_with(&root, ApprovalPolicy::All);
    request.max_iterations = 5;
    let finish = execute_headless(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
    )
    .expect("agent run should finish");
    assert_eq!(finish.status, "completed");
    assert_eq!(
        std::fs::read_to_string(root.join("note.txt")).unwrap(),
        "after\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("new.txt")).unwrap(),
        "created\n"
    );

    let captured = events.captured();
    let previews: Vec<&Value> = captured
        .iter()
        .filter(|event| event.name == "agent:approval_needed")
        .filter_map(|event| event.payload.get("preview"))
        .filter(|preview| preview.is_object())
        .collect();
    assert_eq!(previews.len(), 2, "both writes need a diff preview");
    let note = previews
        .iter()
        .find(|preview| preview["path"] == "note.txt")
        .expect("note preview");
    assert!(note["diff"].as_str().unwrap_or_default().contains("-before"));
    assert!(note["diff"].as_str().unwrap_or_default().contains("+after"));
    assert_eq!(note["added"].as_u64(), Some(1));
    assert_eq!(note["removed"].as_u64(), Some(1));
    let created = previews
        .iter()
        .find(|preview| preview["path"] == "new.txt")
        .expect("new file preview");
    assert!(created["diff"]
        .as_str()
        .unwrap_or_default()
        .contains("+created"));
    assert_eq!(created["removed"].as_u64(), Some(0));

    let result = journal.undo_all();
    assert_eq!(result.restored, 1);
    assert_eq!(result.deleted, 1);
    assert!(result.files.is_empty(), "{:?}", result.files);
    assert_eq!(
        std::fs::read_to_string(root.join("note.txt")).unwrap(),
        "before\n"
    );
    assert!(!root.join("new.txt").exists());
}

#[test]
fn shell_command_does_not_attribute_concurrent_user_edits() {
    let root = workspace("shell-race");
    std::fs::write(root.join("builtin.txt"), "before\n").expect("write builtin");
    let checkpoint_root = root.with_extension("checkpoints");
    let _ = std::fs::remove_dir_all(&checkpoint_root);
    let store = Arc::new(CheckpointStore::new(checkpoint_root.clone()));
    let checkpoint = store
        .begin("e2e-shell-race", &root)
        .expect("begin checkpoint");
    let recorder = checkpoint.recorder();

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |_request| {
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            // The shell step creates agent.txt while the test writes user.txt.
            // Neither path was named by a built-in file tool, so neither may
            // become undoable.
            0 => sse_tool(
                "run_command",
                r#"{"command":"touch started.flag; sleep 1; echo agent > agent.txt"}"#,
                "call_1",
            ),
            1 => sse_tool(
                "write_file",
                r#"{"path":"builtin.txt","content":"after\n"}"#,
                "call_2",
            ),
            _ => sse_text("Finished."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let _approvals = approve_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();
    let mut request = request_with(&root, ApprovalPolicy::All);
    request.max_iterations = 6;

    let user_edit = {
        let root = root.clone();
        thread::spawn(move || {
            for _ in 0..200 {
                if root.join("started.flag").exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            assert!(
                root.join("started.flag").exists(),
                "the shell command never started"
            );
            std::fs::write(root.join("user.txt"), "user\n").expect("concurrent user edit");
        })
    };

    let finish = execute_headless_with_recorder(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
        Some(recorder),
    )
    .expect("agent run should finish");
    user_edit.join().expect("user edit thread");

    assert_eq!(finish.status, "completed");
    assert_eq!(
        std::fs::read_to_string(root.join("builtin.txt")).expect("read builtin"),
        "after\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("agent.txt")).expect("read agent file"),
        "agent\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("user.txt")).expect("read user file"),
        "user\n"
    );

    assert_eq!(
        checkpoint.finish().expect("finish checkpoint"),
        1,
        "only the built-in edit is attributable"
    );
    let state = store.state("e2e-shell-race").expect("checkpoint state");
    let paths: Vec<&str> = state.files.iter().map(|file| file.path.as_str()).collect();
    assert_eq!(paths, vec!["builtin.txt"], "{:?}", state.files);
    assert!(!paths.contains(&"agent.txt"));
    assert!(!paths.contains(&"user.txt"));
    assert!(!paths.contains(&"started.flag"));

    let outcome = store.undo("e2e-shell-race").expect("undo");
    assert_eq!(outcome.restored, 1);
    assert_eq!(
        std::fs::read_to_string(root.join("builtin.txt")).expect("read builtin"),
        "before\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("agent.txt")).expect("agent survives undo"),
        "agent\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("user.txt")).expect("user survives undo"),
        "user\n"
    );
    assert!(root.join("started.flag").exists(), "shell file survives undo");

    let canonical = std::fs::canonicalize(&root).expect("canonical workspace");
    let approvals: Vec<_> = events
        .captured()
        .into_iter()
        .filter(|event| event.name == "agent:approval_needed")
        .collect();
    assert_eq!(approvals.len(), 2, "shell and built-in edit both prompt");
    for approval in approvals {
        assert_eq!(
            approval.payload["workspace_root"],
            canonical.to_string_lossy().as_ref(),
            "approval cards carry the pinned absolute workspace"
        );
    }

    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(&checkpoint_root);
}

#[test]
fn file_edit_allowance_does_not_cover_shell_commands() {
    let root = workspace("allow-edits");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |_request| {
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            0 => sse_tool(
                "write_file",
                r#"{"path":"note.txt","content":"written\n"}"#,
                "call_1",
            ),
            1 => sse_tool("run_command", r#"{"command":"true"}"#, "call_2"),
            _ => sse_text("Finished."),
        }
    });

    let control = Arc::new(RunControl::default());
    control.allow_file_edits();
    let events = AgentEvents::capture();
    let _approvals = approve_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();
    let mut request = request_with(&root, ApprovalPolicy::All);
    request.max_iterations = 6;

    let finish = execute_headless(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
    )
    .expect("agent run should finish");

    assert_eq!(finish.status, "completed");
    assert_eq!(
        std::fs::read_to_string(root.join("note.txt")).expect("read note"),
        "written\n"
    );
    let approvals: Vec<_> = events
        .captured()
        .into_iter()
        .filter(|event| event.name == "agent:approval_needed")
        .collect();
    assert_eq!(approvals.len(), 1, "only the shell command should pause");
    assert_eq!(approvals[0].payload["tool"], "run_command");
    assert_eq!(approvals[0].payload["file_edit"], false);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn verification_gate_blocks_completion_until_the_command_passes() {
    let root = workspace("verify-gate");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |_request| {
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            0 => sse_tool(
                "write_file",
                r#"{"path":"broken.txt","content":"needs a fix\n"}"#,
                "call_1",
            ),
            1 => sse_text("Done. I wrote broken.txt."),
            2 => sse_tool(
                "write_file",
                r#"{"path":"fixed.txt","content":"ok\n"}"#,
                "call_2",
            ),
            _ => sse_text("Fixed the problem; verification should pass now."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let journal = RunJournal::default();
    let mut request = request_with(&root, ApprovalPolicy::Auto);
    request.max_iterations = 8;
    request.verify_command = if cfg!(windows) {
        "if exist fixed.txt (exit 0) else (exit 1)".into()
    } else {
        "test -f fixed.txt".into()
    };

    let finish = execute_headless(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
    )
    .expect("agent run should finish");

    assert_eq!(finish.status, "completed");
    assert!(finish.final_content.contains("verification should pass"));

    let captured = events.captured();
    let verifications: Vec<&Value> = captured
        .iter()
        .filter(|event| {
            event.name == "agent:tool_start" && event.payload["tool"] == "verify"
        })
        .map(|event| &event.payload)
        .collect();
    assert_eq!(verifications.len(), 2, "verification runs at each finish");
    let results: Vec<&Value> = captured
        .iter()
        .filter(|event| {
            event.name == "agent:tool_result" && event.payload["tool"] == "verify"
        })
        .map(|event| &event.payload)
        .collect();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["ok"], false);
    assert_eq!(results[1]["ok"], true);
    assert!(
        engine
            .requests()
            .iter()
            .any(|request| request["messages"]
                .to_string()
                .contains("verification command failed")),
        "the failure must be fed back to the model"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
#[cfg(unix)]
fn irreversible_shell_commands_prompt_even_in_full_auto() {
    let root = workspace("irreversible");
    std::fs::write(root.join("doomed.txt"), "keep me\n").expect("write doomed");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |_request| {
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            0 => sse_tool(
                "run_command",
                r#"{"command":"rm -rf doomed.txt"}"#,
                "call_1",
            ),
            _ => sse_text("Removed the file."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let _approvals = approve_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();
    let mut request = request_with(&root, ApprovalPolicy::Auto);
    request.max_iterations = 4;

    let finish = execute_headless(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
    )
    .expect("agent run should finish");

    assert_eq!(finish.status, "completed");
    let captured = events.captured();
    let approvals: Vec<&Value> = captured
        .iter()
        .filter(|event| event.name == "agent:approval_needed")
        .map(|event| &event.payload)
        .collect();
    assert_eq!(approvals.len(), 1, "Full auto must still stop here");
    assert_eq!(approvals[0]["risk"], "irreversible");
    assert!(approvals[0]["reason"]
        .as_str()
        .unwrap_or_default()
        .contains("Always confirm"));
    assert!(!root.join("doomed.txt").exists());
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn sensitive_files_prompt_and_their_output_is_redacted() {
    let root = workspace("sensitive");
    std::fs::write(root.join(".env"), "API_KEY=sk-testsecret1234567890\nSAFE=yes\n")
        .expect("write env");

    let turns = Arc::new(AtomicUsize::new(0));
    let engine_turns = turns.clone();
    let engine = FakeEngine::start(move |_request| {
        match engine_turns.fetch_add(1, Ordering::SeqCst) {
            0 => sse_tool("read_file", r#"{"path":".env"}"#, "call_1"),
            _ => sse_text("Read the environment file."),
        }
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let _approvals = approve_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();
    let mut request = request_with(&root, ApprovalPolicy::Auto);
    request.max_iterations = 4;

    let finish = execute_headless(
        request,
        engine.base_url(),
        16_384,
        control,
        events.clone(),
        &journal,
    )
    .expect("agent run should finish");

    assert_eq!(finish.status, "completed");
    let captured = events.captured();
    let approvals: Vec<&Value> = captured
        .iter()
        .filter(|event| event.name == "agent:approval_needed")
        .map(|event| &event.payload)
        .collect();
    assert_eq!(approvals.len(), 1, "a sensitive read must always ask");
    assert!(approvals[0]["reason"]
        .as_str()
        .unwrap_or_default()
        .contains("Sensitive path guard"));
    let result = captured
        .iter()
        .find(|event| {
            event.name == "agent:tool_result" && event.payload["tool"] == "read_file"
        })
        .expect("read result");
    let output = result.payload["output"].as_str().unwrap_or_default();
    assert!(output.contains("[REDACTED]"), "{output}");
    assert!(!output.contains("sk-testsecret1234567890"), "{output}");
    assert!(
        !engine
            .requests()
            .iter()
            .any(|request| request.to_string().contains("sk-testsecret1234567890")),
        "the secret must not reach the model"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn abort_stops_a_pending_approval() {
    let root = workspace("abort");
    std::fs::write(root.join("note.txt"), "hello agent\n").expect("write note");

    let engine = FakeEngine::start(|_| {
        sse_tool(
            "write_file",
            r#"{"path":"note.txt","content":"changed\n"}"#,
            "call_1",
        )
    });

    let control = Arc::new(RunControl::default());
    let events = AgentEvents::capture();
    let _abort = abort_when_requested(events.clone(), control.clone());
    let journal = RunJournal::default();

    let mut request = request_with(&root, ApprovalPolicy::All);
    request.max_iterations = 4;

    let (sender, receiver) = mpsc::channel();
    let run_events = events.clone();
    let run_control = control.clone();
    let base_url = engine.base_url();
    thread::spawn(move || {
        let outcome = execute_headless(
            request,
            base_url,
            16_384,
            run_control,
            run_events,
            &journal,
        );
        let _ = sender.send(matches!(outcome, Err(AgentFailure::Aborted)));
    });

    assert!(
        receiver
            .recv_timeout(Duration::from_secs(6))
            .expect("abort should stop a run waiting for approval"),
        "the run should end as Aborted"
    );
    assert!(events
        .captured()
        .iter()
        .any(|event| event.name == "agent:approval_needed"));
}
