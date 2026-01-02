use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::Value;

fn spawn_backend() -> std::process::Child {
    let exe = assert_cmd_path();
    Command::new(exe)
        .arg("--assets-dir")
        .arg(r"c:\Users\engdahlz\StemSep\StemSepApp\assets")
        .arg("--models-dir")
        .arg(r"c:\Users\engdahlz\StemSep\StemSepApp\dev_samples") // doesn't need to exist
        // Tests should exercise Rust-native handlers by default (no Python runtime required).
        .env("STEMSEP_PROXY_PYTHON", "0")
        .env("STEMSEP_PREFER_RUST_SEPARATION", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn backend")
}

fn spawn_backend_with_env(envs: &[(&str, &str)]) -> std::process::Child {
    let exe = assert_cmd_path();
    let mut cmd = Command::new(exe);
    cmd.arg("--assets-dir")
        .arg(r"c:\Users\engdahlz\StemSep\StemSepApp\assets")
        .arg("--models-dir")
        .arg(r"c:\Users\engdahlz\StemSep\StemSepApp\dev_samples")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    for (k, v) in envs {
        cmd.env(k, v);
    }

    cmd.spawn().expect("spawn backend")
}

fn assert_cmd_path() -> String {
    // Prefer target/debug from cargo test workspace
    let mut candidates = vec![
        "target\\debug\\stemsep-backend.exe".to_string(),
        "target\\debug\\stemsep-backend".to_string(),
        "target\\release\\stemsep-backend.exe".to_string(),
        "target\\release\\stemsep-backend".to_string(),
    ];

    for c in candidates.drain(..) {
        if std::path::Path::new(&c).exists() {
            return c;
        }
    }
    panic!("could not locate backend binary under target/(debug|release)");
}

fn read_json_line(reader: &mut BufReader<std::process::ChildStdout>) -> Value {
    let mut line = String::new();
    reader.read_line(&mut line).expect("read line");
    serde_json::from_str::<Value>(line.trim()).expect("parse json")
}

#[test]
fn separate_audio_forwards_volume_compensation_env_gated() {
    if std::env::var("STEMSEP_RUN_VC_TESTS").ok().as_deref() != Some("1") {
        return;
    }

    let python = match std::env::var("STEMSEP_PYTHON") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return,
    };

    let fake = PathBuf::from(r"tests\fixtures\fake_inference.py");
    if !fake.exists() {
        panic!("missing fake inference script: {fake:?}");
    }

    let mut child = spawn_backend_with_env(&[
        ("STEMSEP_PROXY_PYTHON", "0"),
        ("STEMSEP_PREFER_RUST_SEPARATION", "0"),
        ("STEMSEP_PYTHON", python.as_str()),
        ("STEMSEP_INFERENCE_SCRIPT", fake.to_string_lossy().as_ref()),
    ]);

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    let req = serde_json::json!({
        "command": "separate_audio",
        "id": 99,
        "file_path": "C:/this/does/not/matter.wav",
        "model_id": "dummy",
        "output_dir": "C:/tmp/out",
        "device": "cpu",
        "volume_compensation": {
            "enabled": true,
            "stage": "both",
            "db_per_extra_model": 3.0
        }
    });
    stdin
        .write_all(format!("{}\n", req.to_string()).as_bytes())
        .expect("write separate_audio");
    stdin.flush().ok();

    // First response envelope to the request
    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(99));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));

    let mut saw_debug_config = false;
    for _ in 0..50 {
        let msg = read_json_line(&mut reader);
        match msg.get("type").and_then(|v| v.as_str()) {
            Some("debug_config") => {
                let cfg = msg
                    .get("config")
                    .and_then(|v| v.as_object())
                    .expect("debug_config.config object");
                let vc = cfg
                    .get("volume_compensation")
                    .and_then(|v| v.as_object())
                    .expect("volume_compensation object");
                assert_eq!(vc.get("enabled").and_then(|v| v.as_bool()), Some(true));
                assert_eq!(vc.get("stage").and_then(|v| v.as_str()), Some("both"));
                assert_eq!(
                    vc.get("db_per_extra_model").and_then(|v| v.as_f64()),
                    Some(3.0)
                );
                saw_debug_config = true;
                break;
            }
            Some("separation_complete") => break,
            _ => {}
        }
    }

    assert!(
        saw_debug_config,
        "expected debug_config event with forwarded config"
    );
}

#[test]
fn emits_step_started_completed_events_env_gated() {
    if std::env::var("STEMSEP_RUN_STEP_EVENTS_TESTS")
        .ok()
        .as_deref()
        != Some("1")
    {
        return;
    }

    let python = match std::env::var("STEMSEP_PYTHON") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return,
    };

    let fake = PathBuf::from(r"tests\fixtures\fake_inference.py");
    if !fake.exists() {
        panic!("missing fake inference script: {fake:?}");
    }

    let mut child = spawn_backend_with_env(&[
        ("STEMSEP_PROXY_PYTHON", "0"),
        ("STEMSEP_PREFER_RUST_SEPARATION", "0"),
        ("STEMSEP_PYTHON", python.as_str()),
        ("STEMSEP_INFERENCE_SCRIPT", fake.to_string_lossy().as_ref()),
    ]);

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    // start a job (fake script ignores paths)
    let req = serde_json::json!({
        "command": "separate_audio",
        "id": 42,
        "file_path": "C:/this/does/not/matter.wav",
        "model_id": "dummy",
        "output_dir": "C:/tmp/out",
        "device": "cpu"
    });
    stdin
        .write_all(format!("{}\n", req.to_string()).as_bytes())
        .expect("write separate_audio");
    stdin.flush().ok();

    // First response envelope to the request
    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(42));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));
    let job_id = resp
        .get("data")
        .and_then(|d| d.get("job_id"))
        .and_then(|v| v.as_str())
        .expect("job_id in response")
        .to_string();

    let mut started = Vec::new();
    let mut completed = Vec::new();
    let mut saw_complete = false;

    for _ in 0..200 {
        let msg = read_json_line(&mut reader);
        match msg.get("type").and_then(|v| v.as_str()) {
            Some("separation_step_started") => {
                assert_eq!(
                    msg.get("job_id").and_then(|v| v.as_str()),
                    Some(job_id.as_str())
                );
                started.push(msg);
            }
            Some("separation_step_completed") => {
                assert_eq!(
                    msg.get("job_id").and_then(|v| v.as_str()),
                    Some(job_id.as_str())
                );
                completed.push(msg);
            }
            Some("separation_complete") => {
                saw_complete = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_complete, "expected separation_complete event");
    assert!(
        started.len() >= 2,
        "expected >=2 step_started events, got {}",
        started.len()
    );
    assert!(
        completed.len() >= 2,
        "expected >=2 step_completed events, got {}",
        completed.len()
    );

    // Validate meta.step for first started event
    let meta = started[0]
        .get("meta")
        .and_then(|v| v.as_object())
        .expect("meta object");
    assert_eq!(meta.get("phase").and_then(|v| v.as_str()), Some("pipeline"));
    let step = meta
        .get("step")
        .and_then(|v| v.as_object())
        .expect("meta.step object");
    assert_eq!(step.get("index").and_then(|v| v.as_i64()), Some(0));
    assert_eq!(step.get("name").and_then(|v| v.as_str()), Some("demix"));
    assert_eq!(
        step.get("model_id").and_then(|v| v.as_str()),
        Some("model-a")
    );
}

#[test]
fn emits_auto_retry_meta_env_gated() {
    if std::env::var("STEMSEP_RUN_RETRY_TESTS").ok().as_deref() != Some("1") {
        return;
    }

    let python = match std::env::var("STEMSEP_PYTHON") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return,
    };

    let fake = PathBuf::from(r"tests\fixtures\fake_inference.py");
    if !fake.exists() {
        panic!("missing fake inference script: {fake:?}");
    }

    let mut child = spawn_backend_with_env(&[
        ("STEMSEP_PROXY_PYTHON", "0"),
        ("STEMSEP_PREFER_RUST_SEPARATION", "0"),
        ("STEMSEP_PYTHON", python.as_str()),
        ("STEMSEP_INFERENCE_SCRIPT", fake.to_string_lossy().as_ref()),
        ("STEMSEP_FAKE_EMIT_RETRY", "1"),
    ]);

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    let req = serde_json::json!({
        "command": "separate_audio",
        "id": 77,
        "file_path": "C:/this/does/not/matter.wav",
        "model_id": "dummy",
        "output_dir": "C:/tmp/out",
        "device": "cpu"
    });
    stdin
        .write_all(format!("{}\n", req.to_string()).as_bytes())
        .expect("write separate_audio");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(77));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));
    let job_id = resp
        .get("data")
        .and_then(|d| d.get("job_id"))
        .and_then(|v| v.as_str())
        .expect("job_id in response")
        .to_string();

    let mut saw_retry = false;
    for _ in 0..200 {
        let msg = read_json_line(&mut reader);
        if msg.get("job_id").and_then(|v| v.as_str()) != Some(job_id.as_str()) {
            continue;
        }

        if msg.get("type").and_then(|v| v.as_str()) == Some("separation_progress") {
            let meta = msg.get("meta").and_then(|v| v.as_object());
            let auto_retry = meta
                .and_then(|m| m.get("auto_retry"))
                .and_then(|v| v.as_bool());
            if auto_retry == Some(true) {
                saw_retry = true;
                break;
            }
        }

        if msg.get("type").and_then(|v| v.as_str()) == Some("separation_complete") {
            break;
        }
    }

    assert!(
        saw_retry,
        "expected a separation_progress event annotated with meta.auto_retry"
    );
}

#[test]
fn emits_bridge_ready_event() {
    let mut child = spawn_backend();
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    let msg = read_json_line(&mut reader);
    assert_eq!(
        msg.get("type").and_then(|v| v.as_str()),
        Some("bridge_ready")
    );
    assert!(msg.get("models_count").is_some());
    assert!(msg.get("recipes_count").is_some());
}

#[test]
fn ping_roundtrip() {
    let mut child = spawn_backend();

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    stdin
        .write_all(b"{\"command\":\"ping\",\"id\":1}\n")
        .expect("write ping");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(1));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));
    let data = resp.get("data").expect("data");
    assert_eq!(data.get("status").and_then(|v| v.as_str()), Some("ok"));
}

#[test]
fn get_models_returns_array() {
    let mut child = spawn_backend();

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    stdin
        .write_all(b"{\"command\":\"get_models\",\"id\":2}\n")
        .expect("write get_models");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(2));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));
    assert!(resp.get("data").and_then(|v| v.as_array()).is_some());
}

#[test]
fn get_recipes_returns_array() {
    let mut child = spawn_backend();

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    stdin
        .write_all(b"{\"command\":\"get_recipes\",\"id\":3}\n")
        .expect("write get_recipes");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(3));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));
    assert!(resp.get("data").and_then(|v| v.as_array()).is_some());
}

#[test]
fn get_gpu_devices_returns_object() {
    let mut child = spawn_backend();

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    stdin
        .write_all(b"{\"command\":\"get-gpu-devices\",\"id\":4}\n")
        .expect("write get-gpu-devices");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(4));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));
    let data = resp.get("data").expect("data");
    assert!(data.as_object().is_some());

    // Renderer expects system_info for CPU/RAM display.
    let sys = data.get("system_info").expect("system_info");
    assert!(sys.as_object().is_some());
    assert!(sys.get("cpu_count").is_some());
    assert!(sys.get("memory_total_gb").is_some());
}

#[test]
fn separation_preflight_returns_report_on_errors() {
    let mut child = spawn_backend();

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    // Nonexistent input + model should still return success=true with can_proceed=false
    let req = serde_json::json!({
        "command": "separation_preflight",
        "id": 6,
        "file_path": "C:/this/does/not/exist.wav",
        "model_id": "definitely-not-installed",
        "device": "cpu"
    });
    stdin
        .write_all(format!("{}\n", req.to_string()).as_bytes())
        .expect("write separation_preflight");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(6));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));

    let data = resp.get("data").expect("data");
    assert_eq!(
        data.get("can_proceed").and_then(|v| v.as_bool()),
        Some(false)
    );
    assert!(data.get("errors").and_then(|v| v.as_array()).is_some());
    assert!(data.get("warnings").and_then(|v| v.as_array()).is_some());
    assert!(data.get("resolved").and_then(|v| v.as_object()).is_some());
}

#[test]
fn separation_preflight_includes_post_processing_models() {
    let mut child = spawn_backend();

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    let req = serde_json::json!({
        "command": "separation_preflight",
        "id": 7,
        "file_path": "C:/this/does/not/exist.wav",
        "model_id": "missing-base-model",
        "device": "cpu",
        "post_processing_steps": [
            {
                "type": "phase_fix",
                "modelId": "missing-ref-model",
                "description": "Phase Fix"
            }
        ]
    });
    stdin
        .write_all(format!("{}\n", req.to_string()).as_bytes())
        .expect("write separation_preflight");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(7));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));

    let data = resp.get("data").expect("data");
    assert_eq!(
        data.get("can_proceed").and_then(|v| v.as_bool()),
        Some(false)
    );

    let missing = data
        .get("missing_models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let missing_strs: Vec<&str> = missing.iter().filter_map(|v| v.as_str()).collect();
    assert!(missing_strs.contains(&"missing-base-model"));
    assert!(missing_strs.contains(&"missing-ref-model"));
}

#[test]
fn separation_preflight_resolves_recipe_step_models() {
    let mut child = spawn_backend();

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    // Recipe should resolve to a plan with step model ids (not treat the recipe id itself as a model artifact).
    let recipe_id = "recipe_demudder_metal";
    let req = serde_json::json!({
        "command": "separation_preflight",
        "id": 8,
        "file_path": "C:/this/does/not/exist.wav",
        "model_id": recipe_id,
        "device": "cpu"
    });
    stdin
        .write_all(format!("{}\n", req.to_string()).as_bytes())
        .expect("write separation_preflight");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(8));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));

    let report = resp.get("data").expect("data");
    let missing = report
        .get("missing_models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let missing_strs: Vec<String> = missing
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();

    assert!(
        missing_strs.iter().any(|s| s == "bs-roformer-viperx-1297"),
        "expected step model to be missing: bs-roformer-viperx-1297; got: {missing_strs:?}"
    );
    assert!(
        !missing_strs.iter().any(|s| s == "recipe_demudder_metal"),
        "recipe id should not be treated as a model artifact; got: {missing_strs:?}"
    );

    // Should include a resolved recipe plan for UI/debugging.
    let recipe = report
        .get("resolved")
        .and_then(|r| r.get("recipe"))
        .expect("resolved.recipe present");
    assert_eq!(recipe.get("id").and_then(|v| v.as_str()), Some(recipe_id));
    let steps = recipe
        .get("steps")
        .and_then(|v| v.as_array())
        .expect("recipe.steps present");
    assert!(steps
        .iter()
        .any(|s| s.get("model_id").and_then(|v| v.as_str()) == Some("bs-roformer-viperx-1297")));
}

#[test]
fn introspect_model_smoke_env_gated() {
    // This test is disabled by default because it requires a local ONNX file.
    // Provide an absolute path via STEMSEP_TEST_ONNX_PATH.
    let model_path = match std::env::var("STEMSEP_TEST_ONNX_PATH") {
        Ok(p) if !p.trim().is_empty() => p,
        _ => return,
    };

    let mut child = spawn_backend();

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    let req = serde_json::json!({
        "command": "introspect_model",
        "id": 5,
        "model_path": model_path,
    });
    stdin
        .write_all(format!("{}\n", req.to_string()).as_bytes())
        .expect("write introspect_model");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(5));

    if cfg!(feature = "tract") {
        assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));
        let data = resp.get("data").expect("data");
        assert_eq!(data.get("format").and_then(|v| v.as_str()), Some("onnx"));
        assert!(data.get("inputs").and_then(|v| v.as_array()).is_some());
        assert!(data.get("outputs").and_then(|v| v.as_array()).is_some());
    } else {
        assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(false));
        let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            err.contains("NOT_SUPPORTED"),
            "expected NOT_SUPPORTED error, got: {err}"
        );
    }
}

#[test]
fn python_proxy_model_preflight_smoke_env_gated() {
    // This test exercises the optional Rust->Python proxy layer.
    // It is disabled by default because it requires a local Python environment.
    if std::env::var("STEMSEP_RUN_PROXY_TESTS").ok().as_deref() != Some("1") {
        return;
    }

    let python = std::env::var("STEMSEP_PYTHON").expect("set STEMSEP_PYTHON to a python.exe");
    let bridge = std::env::var("STEMSEP_PYTHON_BRIDGE")
        .expect("set STEMSEP_PYTHON_BRIDGE to python-bridge.py");

    let exe = assert_cmd_path();
    let mut child = Command::new(exe)
        .arg("--assets-dir")
        .arg(r"c:\Users\engdahlz\StemSep\StemSepApp\assets")
        .arg("--models-dir")
        .arg(r"c:\Users\engdahlz\StemSep\_models_test")
        .env("STEMSEP_PROXY_PYTHON", "1")
        .env("STEMSEP_PYTHON", python)
        .env("STEMSEP_PYTHON_BRIDGE", bridge)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn backend");

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    stdin
        .write_all(b"{\"command\":\"model_preflight\",\"id\":10,\"model_id\":\"bs-roformer-viperx-1297\"}\n")
        .expect("write model_preflight");
    stdin.flush().ok();

    let resp = read_json_line(&mut reader);
    assert_eq!(resp.get("id").and_then(|v| v.as_i64()), Some(10));
    assert_eq!(resp.get("success").and_then(|v| v.as_bool()), Some(true));
    assert!(resp.get("data").is_some());
}

#[test]
fn rust_dummy_separation_smoke_env_gated() {
    if std::env::var("STEMSEP_RUN_RUST_SEPARATION_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        return;
    }

    let exe = assert_cmd_path();
    let temp = std::env::temp_dir().join("stemsep_rust_sep_smoke");
    let _ = std::fs::remove_dir_all(&temp);
    std::fs::create_dir_all(&temp).expect("create temp dir");

    // Create a short stereo WAV (440 Hz sine)
    let input = temp.join("input.wav");
    let sr: u32 = 44100;
    let frames = sr / 2; // 0.5 sec
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&input, spec).expect("wav create");
    for i in 0..frames {
        let t = i as f32 / sr as f32;
        let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        let v = (s * i16::MAX as f32) as i16;
        w.write_sample(v).unwrap();
        w.write_sample(v).unwrap();
    }
    w.finalize().unwrap();

    let out_dir = temp.join("out");

    let mut child = Command::new(exe)
        .arg("--assets-dir")
        .arg(r"c:\Users\engdahlz\StemSep\StemSepApp\assets")
        .arg("--models-dir")
        .arg(r"c:\Users\engdahlz\StemSep\_models_test")
        .env("STEMSEP_PROXY_PYTHON", "1")
        .env("STEMSEP_PREFER_RUST_SEPARATION", "1")
        .env("STEMSEP_ENABLE_DUMMY_SEPARATION", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn backend");

    let stdin = child.stdin.as_mut().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);

    // consume bridge_ready
    let _ = read_json_line(&mut reader);

    let req = serde_json::json!({
        "command": "separate_audio",
        "file_path": input.to_string_lossy(),
        "model_id": "dummy",
        "output_dir": out_dir.to_string_lossy(),
        "stems": ["vocals", "instrumental"],
        "device": "cpu"
    });
    stdin
        .write_all(format!("{}\n", req.to_string()).as_bytes())
        .expect("write separate_audio");
    stdin.flush().ok();

    // Read until separation_complete
    let mut complete: Option<serde_json::Value> = None;
    for _ in 0..50 {
        let msg = read_json_line(&mut reader);
        if msg.get("type").and_then(|v| v.as_str()) == Some("separation_complete") {
            complete = Some(msg);
            break;
        }
    }

    let complete = complete.expect("expected separation_complete event");
    let files = complete
        .get("output_files")
        .and_then(|v| v.as_object())
        .expect("output_files object");

    let vocals = PathBuf::from(
        files
            .get("vocals")
            .and_then(|v| v.as_str())
            .expect("vocals path"),
    );
    let inst = PathBuf::from(
        files
            .get("instrumental")
            .and_then(|v| v.as_str())
            .expect("instrumental path"),
    );
    assert!(vocals.exists(), "vocals output exists");
    assert!(inst.exists(), "instrumental output exists");
}
