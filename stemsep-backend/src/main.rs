use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use uuid::Uuid;

use sysinfo::System;

use reqwest;
use reqwest::header;

#[cfg(feature = "tract")]
use tract_onnx::prelude as tract;

#[cfg(feature = "tract")]
use tract_onnx::prelude::Framework;

#[derive(Debug, Deserialize)]
struct RequestEnvelope {
    #[serde(default)]
    id: Option<Value>,
    command: String,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

fn check_torch_available() -> bool {
    let python = locate_python_exe();
    let out = Command::new(&python)
        .arg("-c")
        .arg("import torch  # noqa: F401")
        .output();
    match out {
        Ok(out) => out.status.success(),
        Err(_) => false,
    }
}

#[derive(Debug, Serialize)]
struct ResponseEnvelope {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn now_ts_seconds() -> f64 {
    (Utc::now().timestamp_millis() as f64) / 1000.0
}

#[derive(Debug, Clone)]
struct BackendConfig {
    assets_dir: PathBuf,
    models_dir: PathBuf,
}

fn parse_arg_value(args: &[OsString], name: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            if i + 1 < args.len() {
                return args[i + 1].to_str().map(|s| s.to_string());
            }
            return None;
        }
        if let Some(s) = args[i].to_str() {
            let prefix = format!("{name}=");
            if s.starts_with(&prefix) {
                return Some(s[prefix.len()..].to_string());
            }
        }
        i += 1;
    }
    None
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

fn locate_assets_dir(cli_assets: Option<String>) -> Result<PathBuf> {
    if let Some(p) = cli_assets {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Ok(pb);
        }
        return Err(anyhow!("assets dir not found: {}", pb.display()));
    }
    if let Ok(env) = std::env::var("STEMSEP_ASSETS_DIR") {
        let pb = PathBuf::from(env);
        if pb.exists() {
            return Ok(pb);
        }
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    let mut candidates: Vec<PathBuf> = vec![cwd.join("StemSepApp").join("assets")];
    candidates.push(cwd.join("..").join("StemSepApp").join("assets"));
    if let Some(exe) = exe_dir {
        // target/(debug|release)/ -> crate -> repo
        candidates.push(
            exe.join("..")
                .join("..")
                .join("..")
                .join("StemSepApp")
                .join("assets"),
        );
        candidates.push(exe.join("..").join("..").join("StemSepApp").join("assets"));
        candidates.push(exe.join("StemSepApp").join("assets"));
    }

    for c in candidates {
        if c.exists() {
            return Ok(c);
        }
    }

    Err(anyhow!(
        "Unable to locate StemSepApp/assets. Provide --assets-dir or set STEMSEP_ASSETS_DIR."
    ))
}

fn locate_models_dir(cli_models: Option<String>) -> Result<PathBuf> {
    if let Some(p) = cli_models {
        return Ok(PathBuf::from(p));
    }
    if let Ok(env) = std::env::var("STEMSEP_MODELS_DIR") {
        return Ok(PathBuf::from(env));
    }
    let home = home_dir().ok_or_else(|| anyhow!("Unable to resolve home directory"))?;
    Ok(home.join(".stemsep").join("models"))
}

fn read_json_file(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn load_model_id_aliases(cfg: &BackendConfig) -> Value {
    // Best-effort: aliases are optional and should never crash backend if missing/malformed
    let p = cfg
        .assets_dir
        .join("registry")
        .join("model_id_aliases.json");
    read_json_file(&p).unwrap_or(Value::Null)
}

fn resolve_model_id_alias(cfg: &BackendConfig, model_id: &str) -> String {
    let aliases_json = load_model_id_aliases(cfg);
    let Some(map) = aliases_json.get("aliases").and_then(|v| v.as_object()) else {
        return model_id.to_string();
    };

    if let Some(v) = map.get(model_id).and_then(|v| v.as_str()) {
        if !v.trim().is_empty() {
            return v.to_string();
        }
    }

    model_id.to_string()
}

fn url_basename(url: &str) -> Option<String> {
    let trimmed = url.split('?').next().unwrap_or(url);
    trimmed.split('/').last().and_then(|s| {
        if s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    })
}

fn lookup_model_artifact_filenames(
    cfg: &BackendConfig,
    model_id: &str,
) -> Result<(Option<String>, Option<String>)> {
    let model_id = resolve_model_id_alias(cfg, model_id);
    let models_json = read_json_file(&cfg.assets_dir.join("models.json.bak"))?;
    let models = models_json
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    for m in models {
        let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if id != model_id {
            continue;
        }

        let artifacts = m.get("artifacts").and_then(|v| v.as_object());
        if artifacts.is_none() {
            return Ok((None, None));
        }
        let artifacts = artifacts.unwrap();

        let mut ckpt: Option<String> = None;
        let mut cfg_name: Option<String> = None;

        if let Some(primary) = artifacts.get("primary").and_then(|v| v.as_object()) {
            if let Some(fname) = primary.get("filename").and_then(|v| v.as_str()) {
                let f = fname.trim();
                if !f.is_empty() && !f.contains(".MISSING") {
                    ckpt = Some(f.to_string());
                }
            }
        }

        if let Some(c) = artifacts.get("config").and_then(|v| v.as_object()) {
            if let Some(fname) = c.get("filename").and_then(|v| v.as_str()) {
                let f = fname.trim();
                if !f.is_empty() && !f.contains(".MISSING") {
                    cfg_name = Some(f.to_string());
                }
            }
        }

        return Ok((ckpt, cfg_name));
    }

    Ok((None, None))
}

fn lookup_model_links(
    cfg: &BackendConfig,
    model_id: &str,
) -> Result<(Option<String>, Option<String>)> {
    let model_id = resolve_model_id_alias(cfg, model_id);
    let models_json = read_json_file(&cfg.assets_dir.join("models.json.bak"))?;
    let models = models_json
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    for m in models {
        let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if id == model_id {
            let mut ckpt: Option<String> = None;
            let mut cfg_url: Option<String> = None;
            if let Some(links) = m.get("links") {
                if let Some(u) = links.get("checkpoint").and_then(|v| v.as_str()) {
                    if !u.trim().is_empty() {
                        ckpt = Some(u.to_string());
                    }
                }
                if let Some(u) = links.get("config").and_then(|v| v.as_str()) {
                    if !u.trim().is_empty() {
                        cfg_url = Some(u.to_string());
                    }
                }
            }
            return Ok((ckpt, cfg_url));
        }
    }
    Ok((None, None))
}

fn download_with_progress_cancellable(
    url: &str,
    dest_path: &Path,
    stdout: Arc<Mutex<io::Stdout>>,
    model_id: String,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<()> {
    std::fs::create_dir_all(
        dest_path
            .parent()
            .ok_or_else(|| anyhow!("invalid destination path"))?,
    )
    .context("create models dir")?;

    let tmp_path = {
        let fname = dest_path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("invalid destination filename"))?;
        dest_path.with_file_name(format!("{fname}.part"))
    };

    let hf_token = std::env::var("STEMSEP_HF_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            std::env::var("HF_TOKEN")
                .ok()
                .filter(|s| !s.trim().is_empty())
        })
        .or_else(|| {
            std::env::var("HUGGINGFACE_HUB_TOKEN")
                .ok()
                .filter(|s| !s.trim().is_empty())
        });

    let client = reqwest::blocking::Client::builder()
        .user_agent("StemSep/stemsep-backend (model downloader)")
        .redirect(reqwest::redirect::Policy::limited(10))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .context("build http client")?;

    let existing_len: u64 = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);

    send_event(
        &stdout,
        serde_json::json!({
            "type": "progress",
            "model_id": model_id,
            "progress": 0,
            "message": if existing_len > 0 { "Resuming download" } else { "Starting download" }
        }),
    );

    let mut req = client.get(url);
    // Hugging Face uses gated repos/private files; bearer token enables downloads.
    if let Some(token) = &hf_token {
        if url.contains("huggingface.co") || url.contains("cdn-lfs.huggingface.co") {
            req = req.bearer_auth(token);
        }
    }

    // If we have a partial file, attempt HTTP Range resume.
    if existing_len > 0 {
        req = req.header(header::RANGE, format!("bytes={existing_len}-"));
    }

    let mut resp = req.send().with_context(|| format!("GET {url}"))?;

    // Range requested, but server says range is invalid. This can happen if we already
    // downloaded the full file but never renamed it into place.
    if existing_len > 0 && resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        // Best-effort: promote the partial file to final.
        if tmp_path.exists() {
            if dest_path.exists() {
                let _ = std::fs::remove_file(&tmp_path);
            } else {
                std::fs::rename(&tmp_path, dest_path)
                    .with_context(|| format!("move into place: {}", dest_path.display()))?;
            }

            return Ok(());
        }
    }

    // If server ignored Range (200 OK) but we attempted resume, restart from scratch.
    if existing_len > 0 && resp.status() == reqwest::StatusCode::OK {
        let _ = std::fs::remove_file(&tmp_path);
        let mut req2 = client.get(url);
        if let Some(token) = &hf_token {
            if url.contains("huggingface.co") || url.contains("cdn-lfs.huggingface.co") {
                req2 = req2.bearer_auth(token);
            }
        }
        resp = req2.send().with_context(|| format!("GET {url}"))?;
    }

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_else(|_| "".to_string());
        let body_snip = body.trim();
        let mut msg = if body_snip.is_empty() {
            format!("download failed: HTTP {status}")
        } else {
            // Keep message short for UI toast/logs.
            let clipped = if body_snip.len() > 500 {
                &body_snip[..500]
            } else {
                body_snip
            };
            format!("download failed: HTTP {status} - {clipped}")
        };

        if status == reqwest::StatusCode::UNAUTHORIZED {
            msg.push_str(
                "\nThis looks like a gated/private Hugging Face model. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN / STEMSEP_HF_TOKEN) and restart the app.",
            );
        }

        return Err(anyhow!(msg));
    }

    let status = resp.status();
    let resumed =
        existing_len > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT && tmp_path.exists();

    // When resuming via 206, content_length() is the remaining bytes; total becomes existing+remaining.
    let total = match (resumed, resp.content_length()) {
        (true, Some(remaining)) => Some(existing_len + remaining),
        (true, None) => None,
        (false, t) => t,
    };

    let mut out = if resumed {
        std::fs::OpenOptions::new()
            .append(true)
            .open(&tmp_path)
            .with_context(|| format!("append {}", tmp_path.display()))?
    } else {
        std::fs::File::create(&tmp_path)
            .with_context(|| format!("create {}", tmp_path.display()))?
    };

    let mut downloaded: u64 = if resumed { existing_len } else { 0 };
    let mut buf = [0u8; 1024 * 128];
    let mut last_pct: i64 = -1;

    let is_cancelled = || {
        cancel
            .as_ref()
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(false)
    };

    // Emit an initial progress based on existing bytes if we can compute it.
    if resumed {
        if let Some(total_bytes) = total {
            let pct = ((downloaded as f64 / total_bytes as f64) * 100.0).floor() as i64;
            send_event(
                &stdout,
                serde_json::json!({
                    "type": "progress",
                    "model_id": model_id,
                    "progress": pct,
                    "message": "Resumed"
                }),
            );
            last_pct = pct;
        }
    }

    loop {
        if is_cancelled() {
            out.flush().ok();
            drop(out);
            // Keep the .part file so we can resume later.
            send_event(
                &stdout,
                serde_json::json!({
                    "type": "paused",
                    "model_id": model_id,
                    "progress": total.map(|t| ((downloaded as f64 / t as f64) * 100.0).floor() as i64)
                }),
            );
            return Ok(());
        }
        let n = resp.read(&mut buf).context("read download stream")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).context("write download file")?;
        downloaded += n as u64;

        if let Some(total_bytes) = total {
            let pct = ((downloaded as f64 / total_bytes as f64) * 100.0).floor() as i64;
            // Emit every 2% to avoid spamming
            if pct >= 0 && pct <= 100 && pct / 2 != last_pct / 2 {
                last_pct = pct;
                send_event(
                    &stdout,
                    serde_json::json!({
                        "type": "progress",
                        "model_id": model_id,
                        "progress": pct,
                    }),
                );
            }
        }
    }

    out.flush().ok();
    drop(out);

    // Atomic-ish replace
    if dest_path.exists() {
        let _ = std::fs::remove_file(dest_path);
    }
    std::fs::rename(&tmp_path, dest_path)
        .with_context(|| format!("move into place: {}", dest_path.display()))?;

    Ok(())
}

#[derive(Clone)]
struct DownloadFile {
    url: String,
    dest_path: PathBuf,
}

#[derive(Clone)]
struct DownloadTask {
    files: Vec<DownloadFile>,
    cancel: Arc<AtomicBool>,
}

#[derive(Clone, Default)]
struct DownloadManager {
    tasks: Arc<Mutex<HashMap<String, DownloadTask>>>,
}

impl DownloadManager {
    fn start(&self, model_id: &str, files: Vec<DownloadFile>, stdout: Arc<Mutex<io::Stdout>>) {
        let cancel = Arc::new(AtomicBool::new(false));
        let task = DownloadTask {
            files: files.clone(),
            cancel: cancel.clone(),
        };
        {
            let mut map = self.tasks.lock().expect("downloads lock");
            map.insert(model_id.to_string(), task);
        }

        let mgr = self.clone();
        let model_id_clone = model_id.to_string();
        thread::spawn(move || {
            let mut primary_path: Option<String> = None;
            let mut completed_paths: Vec<String> = Vec::new();

            for (idx, f) in files.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    // download_with_progress_cancellable will have emitted "paused".
                    return;
                }

                if idx == 0 {
                    primary_path = Some(f.dest_path.to_string_lossy().to_string());
                }

                // Skip already-downloaded files.
                if f.dest_path.exists() {
                    completed_paths.push(f.dest_path.to_string_lossy().to_string());
                    continue;
                }

                let res = download_with_progress_cancellable(
                    &f.url,
                    &f.dest_path,
                    stdout.clone(),
                    model_id_clone.clone(),
                    Some(cancel.clone()),
                );

                match res {
                    Ok(()) => {
                        if cancel.load(Ordering::Relaxed) {
                            // paused
                            return;
                        }
                        if f.dest_path.exists() {
                            completed_paths.push(f.dest_path.to_string_lossy().to_string());
                        }
                    }
                    Err(e) => {
                        send_event(
                            &stdout,
                            serde_json::json!({
                                "type": "error",
                                "model_id": model_id_clone,
                                "error": format!("{e:#}")
                            }),
                        );
                        mgr.clear_task(&model_id_clone);
                        return;
                    }
                }
            }

            // All files done.
            send_event(
                &stdout,
                serde_json::json!({
                    "type": "progress",
                    "model_id": model_id_clone,
                    "progress": 100,
                }),
            );
            send_event(
                &stdout,
                serde_json::json!({
                    "type": "complete",
                    "model_id": model_id_clone,
                    "path": primary_path,
                    "paths": completed_paths,
                }),
            );
            mgr.clear_task(&model_id_clone);
        });
    }

    fn request_pause(&self, model_id: &str) -> bool {
        let map = self.tasks.lock().expect("downloads lock");
        if let Some(task) = map.get(model_id) {
            task.cancel.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    fn resume(&self, model_id: &str, stdout: Arc<Mutex<io::Stdout>>) -> bool {
        let files = {
            let mut map = self.tasks.lock().expect("downloads lock");
            if let Some(task) = map.get(model_id) {
                let files = task.files.clone();
                // Replace task with a fresh cancel flag.
                map.insert(
                    model_id.to_string(),
                    DownloadTask {
                        files: files.clone(),
                        cancel: Arc::new(AtomicBool::new(false)),
                    },
                );
                Some(files)
            } else {
                None
            }
        };

        if let Some(files) = files {
            self.start(model_id, files, stdout);
            true
        } else {
            false
        }
    }

    fn clear_task(&self, model_id: &str) {
        let mut map = self.tasks.lock().expect("downloads lock");
        map.remove(model_id);
    }
}

fn remove_model_files(cfg: &BackendConfig, model_id: &str) -> Result<usize> {
    let models_json = read_json_file(&cfg.assets_dir.join("models.json.bak"))?;
    let models = models_json
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut candidates: Vec<PathBuf> = Vec::new();

    // Basenames from registry links
    if let Some(m) = models
        .iter()
        .find(|m| m.get("id").and_then(|v| v.as_str()) == Some(model_id))
    {
        // Prefer explicit artifact filenames when present.
        if let Some(artifacts) = m.get("artifacts").and_then(|v| v.as_object()) {
            if let Some(primary) = artifacts.get("primary").and_then(|v| v.as_object()) {
                if let Some(fname) = primary.get("filename").and_then(|v| v.as_str()) {
                    let f = fname.trim();
                    if !f.is_empty() && !f.contains(".MISSING") {
                        candidates.push(cfg.models_dir.join(f));
                        candidates.push(cfg.models_dir.join(format!("{f}.part")));
                    }
                }
            }
            if let Some(c) = artifacts.get("config").and_then(|v| v.as_object()) {
                if let Some(fname) = c.get("filename").and_then(|v| v.as_str()) {
                    let f = fname.trim();
                    if !f.is_empty() && !f.contains(".MISSING") {
                        candidates.push(cfg.models_dir.join(f));
                        candidates.push(cfg.models_dir.join(format!("{f}.part")));
                    }
                }
            }
        }

        if let Some(links) = m.get("links") {
            for key in ["checkpoint", "config"] {
                if let Some(u) = links.get(key).and_then(|v| v.as_str()) {
                    if let Some(base) = url_basename(u) {
                        candidates.push(cfg.models_dir.join(&base));
                        candidates.push(cfg.models_dir.join(format!("{base}.part")));
                    }
                }
            }
        }
    }

    // Common id-based filenames
    for ext in [
        ".ckpt",
        ".pth",
        ".pt",
        ".onnx",
        ".safetensors",
        ".yaml",
        ".yml",
    ] {
        candidates.push(cfg.models_dir.join(format!("{model_id}{ext}")));
        candidates.push(cfg.models_dir.join(format!("{model_id}{ext}.part")));
    }

    candidates.sort();
    candidates.dedup();

    let mut removed = 0usize;
    for p in candidates {
        if p.exists() {
            // Retry loop to handle Windows file locking race conditions
            let mut success = false;
            for _ in 0..10 {
                match std::fs::remove_file(&p) {
                    Ok(_) => {
                        success = true;
                        break;
                    }
                    Err(_) => {
                        std::thread::sleep(Duration::from_millis(200));
                    }
                }
            }
            if success {
                removed += 1;
            }
        }
    }
    Ok(removed)
}

fn any_model_file_exists(models_dir: &Path, model_id: &str, model_obj: &Value) -> bool {
    // If registry declares explicit files, require checkpoint.
    // Config is optional for some models (e.g. standard ones), but checkpoint is mandatory.
    if let Some(artifacts) = model_obj.get("artifacts").and_then(|v| v.as_object()) {
        if let Some(primary) = artifacts.get("primary").and_then(|v| v.as_object()) {
            if let Some(fname) = primary.get("filename").and_then(|v| v.as_str()) {
                let f = fname.trim();
                if !f.is_empty() && !f.contains(".MISSING") {
                    if models_dir.join(f).exists() {
                        return true;
                    }
                }
            }
        }
    }

    if let Some(links) = model_obj.get("links") {
        let ok_ckpt;
        let mut ok_cfg = true;

        let ckpt_link = links.get("checkpoint").and_then(|v| v.as_str()).unwrap_or("");
        if !ckpt_link.trim().is_empty() {
            ok_ckpt = if ckpt_link.to_lowercase().contains(".onnx") {
                models_dir.join(format!("{model_id}.onnx")).exists()
            } else if let Some(base) = url_basename(ckpt_link) {
                models_dir.join(base).exists()
            } else {
                false
            };
        } else {
            // If checkpoint link is empty in registry, it can't be "installed" via registry path.
            ok_ckpt = false;
        }

        let cfg_link = links.get("config").and_then(|v| v.as_str()).unwrap_or("");
        if !cfg_link.trim().is_empty() {
            // Prefer stable per-model config filenames to avoid collisions like "config.yaml".
            // Still accept historical basename installs.
            let alias_yaml = models_dir.join(format!("{model_id}.yaml"));
            let alias_yml = models_dir.join(format!("{model_id}.yml"));
            let base_ok = url_basename(cfg_link)
                .map(|base| models_dir.join(base).exists())
                .unwrap_or(false);
            ok_cfg = alias_yaml.exists() || alias_yml.exists() || base_ok;
        }

        if ok_ckpt && ok_cfg {
            return true;
        }
        // If we had links and failed, don't fall back to heuristic (to avoid false positives)
        if !ckpt_link.trim().is_empty() || !cfg_link.trim().is_empty() {
            return false;
        }
    }

    // Fallback heuristic for models without explicit links or custom models.
    // MUST find at least one checkpoint-like file.
    for ext in [
        ".ckpt",
        ".pth",
        ".pt",
        ".onnx",
        ".safetensors",
    ] {
        if models_dir.join(format!("{model_id}{ext}")).exists() {
            return true;
        }
    }
    false
}

fn locate_nvidia_smi() -> Option<std::path::PathBuf> {
    // Prefer PATH first.
    if Command::new("nvidia-smi").arg("-h").output().is_ok() {
        return Some(std::path::PathBuf::from("nvidia-smi"));
    }

    // Common Windows install locations.
    if cfg!(target_os = "windows") {
        let candidates = [
            r"C:\\Windows\\System32\\nvidia-smi.exe",
            r"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
            r"C:\\Program Files (x86)\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe",
        ];
        for c in candidates {
            let p = std::path::PathBuf::from(c);
            if p.exists() {
                return Some(p);
            }
        }
    }

    // Common Linux location.
    if cfg!(target_os = "linux") {
        let candidates = ["/usr/bin/nvidia-smi", "/bin/nvidia-smi"];
        for c in candidates {
            let p = std::path::PathBuf::from(c);
            if p.exists() {
                return Some(p);
            }
        }
    }

    None
}

fn detect_gpus() -> Value {
    // Best-effort on Windows/Linux: try nvidia-smi if present.
    let nvidia_smi = locate_nvidia_smi();
    let output = match &nvidia_smi {
        Some(cmd) => Command::new(cmd)
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output(),
        None => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "nvidia-smi not found",
        )),
    };

    let mut gpus: Vec<Value> = Vec::new();
    if let Ok(out) = output {
        if out.status.success() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                for (idx, line) in s.lines().enumerate() {
                    let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
                    if parts.len() >= 2 {
                        let name = parts[0];
                        let mem_mb: f64 = parts[1].parse::<f64>().unwrap_or(0.0);
                        gpus.push(serde_json::json!({
                            "id": format!("cuda:{idx}"),
                            "index": idx,
                            "name": name,
                            "memory_gb": (mem_mb / 1024.0)
                        }));
                    }
                }
            }
        }
    }

    // Mark a single "recommended" GPU (highest VRAM) to match renderer expectations.
    let mut best_gpu: Option<Value> = None;
    if !gpus.is_empty() {
        let mut best_idx = 0usize;
        let mut best_mem = -1.0_f64;
        for (i, g) in gpus.iter().enumerate() {
            let mem = g.get("memory_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if mem > best_mem {
                best_mem = mem;
                best_idx = i;
            }
        }
        if let Some(obj) = gpus.get_mut(best_idx).and_then(|v| v.as_object_mut()) {
            obj.insert("recommended".to_string(), Value::Bool(true));
        }
        best_gpu = gpus.get(best_idx).cloned();
    }

    // System info used by the renderer's "System Health" panel.
    let system_info = {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_brand: Option<String> = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .filter(|s| !s.trim().is_empty());

        // sysinfo's memory unit has changed across versions; handle both KiB and bytes.
        let total_mem_raw = sys.total_memory() as f64;
        let mem_gb = if total_mem_raw > 1_000_000_000.0 {
            total_mem_raw / 1024.0 / 1024.0 / 1024.0
        } else {
            total_mem_raw / 1024.0 / 1024.0
        };
        let mem_gb_rounded: Option<f64> = if mem_gb.is_finite() && mem_gb > 0.0 {
            Some((mem_gb * 10.0).round() / 10.0)
        } else {
            None
        };

        serde_json::json!({
            "processor": cpu_brand,
            "cpu_count": sys.cpus().len(),
            "memory_total_gb": mem_gb_rounded
        })
    };

    let cuda_version: Option<String> = match &nvidia_smi {
        Some(cmd) => Command::new(cmd)
            .args(["--query-gpu=driver_version", "--format=csv,noheader"])
            .output()
            .ok()
            .and_then(|out| {
                if !out.status.success() {
                    return None;
                }
                String::from_utf8(out.stdout).ok()
            })
            .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
            .filter(|s| !s.is_empty()),
        None => None,
    };

    let has_cuda = !gpus.is_empty();
    let gpu_count = gpus.len();

    let recommended_profile = if let Some(best) = best_gpu.as_ref() {
        serde_json::json!({
            "profile_name": "cuda",
            "settings": {},
            "vram_gb": best.get("memory_gb").cloned().unwrap_or(Value::from(0))
        })
    } else {
        Value::Null
    };

    serde_json::json!({
        "gpus": gpus,
        "gpu_count": gpu_count,
        "has_cuda": has_cuda,
        "cuda_version": cuda_version,
        "system_info": system_info,
        "recommended_profile": recommended_profile
    })
}

fn write_json_line_locked<T: Serialize>(stdout: &Arc<Mutex<io::Stdout>>, value: &T) -> Result<()> {
    let line = serde_json::to_string(value).context("serialize json")?;
    let mut out = stdout.lock().expect("stdout lock");
    out.write_all(line.as_bytes())
        .and_then(|_| out.write_all(b"\n"))
        .context("write stdout")?;
    out.flush().ok();
    Ok(())
}

fn write_raw_line_locked(stdout: &Arc<Mutex<io::Stdout>>, line: &str) -> Result<()> {
    let mut out = stdout.lock().expect("stdout lock");
    out.write_all(line.as_bytes())
        .and_then(|_| out.write_all(b"\n"))
        .context("write stdout")?;
    out.flush().ok();
    Ok(())
}

fn env_true(key: &str, default_value: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default_value,
    }
}

fn env_u64(key: &str, default_value: u64) -> u64 {
    match std::env::var(key) {
        Ok(v) => v.trim().parse::<u64>().unwrap_or(default_value),
        Err(_) => default_value,
    }
}

fn send_event(stdout: &Arc<Mutex<io::Stdout>>, event: Value) {
    let _ = write_json_line_locked(stdout, &event);
}

#[allow(dead_code)]
fn write_wav_16bit(path: &Path, sr: u32, samples: &[f32]) -> Result<()> {
    // samples: interleaved stereo (L,R,L,R,...) or mono if len is frames
    let channels: u16 = 2;
    let spec = hound::WavSpec {
        channels,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).context("create wav")?;
    for s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(v).context("write sample")?;
    }
    writer.finalize().context("finalize wav")?;
    Ok(())
}

#[allow(dead_code)]
fn read_wav_to_f32_stereo(path: &Path) -> Result<(u32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path).context("open wav")?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels as usize;

    // Convert to f32 in [-1, 1]
    let mut mono: Vec<f32> = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                for s in reader.samples::<i16>() {
                    let v = s.context("read sample")? as f32 / i16::MAX as f32;
                    mono.push(v);
                }
            } else {
                for s in reader.samples::<i32>() {
                    let v = s.context("read sample")? as f32 / i32::MAX as f32;
                    mono.push(v);
                }
            }
        }
        hound::SampleFormat::Float => {
            for s in reader.samples::<f32>() {
                mono.push(s.context("read sample")?);
            }
        }
    }

    // Ensure stereo interleaved
    if ch == 1 {
        let mut stereo: Vec<f32> = Vec::with_capacity(mono.len() * 2);
        for v in mono {
            stereo.push(v);
            stereo.push(v);
        }
        Ok((sr, stereo))
    } else if ch == 2 {
        Ok((sr, mono))
    } else {
        Err(anyhow!("unsupported wav channels: {ch}"))
    }
}

#[cfg(feature = "tract")]
fn tract_introspect_onnx(path: &Path) -> Result<Value> {
    let model = tract::onnx()
        .model_for_path(path)
        .with_context(|| format!("load onnx model: {}", path.display()))?;
    let inputs: Vec<Value> = model
        .input_outlets()
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let fact = model.input_fact(idx).ok();
            let shape = fact.as_ref().map(|f| format!("{:?}", f.shape));
            let dtype = fact
                .as_ref()
                .map(|f| format!("{:?}", f.datum_type))
                .unwrap_or_else(|| "unknown".to_string());
            serde_json::json!({
                "index": idx,
                "dtype": dtype,
                "shape": shape
            })
        })
        .collect();

    let outputs: Vec<Value> = model
        .output_outlets()
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let fact = model.output_fact(idx).ok();
            let shape = fact.as_ref().map(|f| format!("{:?}", f.shape));
            let dtype = fact
                .as_ref()
                .map(|f| format!("{:?}", f.datum_type))
                .unwrap_or_else(|| "unknown".to_string());
            serde_json::json!({
                "index": idx,
                "dtype": dtype,
                "shape": shape
            })
        })
        .collect();

    Ok(serde_json::json!({
        "format": "onnx",
        "path": path.to_string_lossy(),
        "inputs": inputs,
        "outputs": outputs
    }))
}

fn send_ok(stdout: &Arc<Mutex<io::Stdout>>, id: Option<Value>, data: Value) -> Result<()> {
    let resp = ResponseEnvelope {
        id,
        success: true,
        data: Some(data),
        error: None,
    };
    write_json_line_locked(stdout, &resp)
}

fn send_err(
    stdout: &Arc<Mutex<io::Stdout>>,
    id: Option<Value>,
    code: &str,
    message: &str,
) -> Result<()> {
    let resp = ResponseEnvelope {
        id,
        success: false,
        data: None,
        error: Some(format!("{code}: {message}")),
    };
    write_json_line_locked(stdout, &resp)
}

#[derive(Debug)]
struct PythonProxy {
    _child: Child,
    stdin: Arc<Mutex<ChildStdin>>, // shared so main thread can forward requests
}

fn locate_python_bridge_script() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("STEMSEP_PYTHON_BRIDGE") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| cwd.clone());

    let candidates = vec![
        cwd.join("electron-poc").join("python-bridge.py"),
        cwd.join("python-bridge.py"),
        exe_dir.join("python-bridge.py"),
        exe_dir
            .join("..")
            .join("..")
            .join("..")
            .join("electron-poc")
            .join("python-bridge.py"),
    ];

    candidates.into_iter().find(|p| p.exists())
}

fn locate_inference_script() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("STEMSEP_INFERENCE_SCRIPT") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| cwd.clone());

    let candidates = vec![
        cwd.join("scripts").join("inference.py"),
        cwd.join("inference.py"),
        exe_dir.join("scripts").join("inference.py"),
        exe_dir.join("inference.py"),
        // Dev fallback
        exe_dir
            .join("..")
            .join("..")
            .join("..")
            .join("scripts")
            .join("inference.py"),
    ];

    candidates.into_iter().find(|p| p.exists())
}

fn locate_python_exe() -> String {
    if let Ok(p) = std::env::var("STEMSEP_PYTHON") {
        return p;
    }
    if cfg!(windows) {
        // Prefer the launcher if present.
        return "py".to_string();
    }
    "python3".to_string()
}

fn check_neuralop_available() -> Result<()> {
    let python = locate_python_exe();
    let code = r#"import sys
try:
    try:
        from neuralop.models import FNO1d  # noqa: F401
    except Exception:
        from neuralop.models import FNO  # noqa: F401
except Exception as e:
    sys.stderr.write("FNO models require the 'neuraloperator' (neuralop) dependency. ")
    sys.stderr.write("Install it in the Python runtime used by StemSep and restart. ")
    sys.stderr.write("Original import error: " + repr(e) + "\n")
    sys.exit(1)
"#;

    let out = Command::new(&python)
        .arg("-c")
        .arg(code)
        .output()
        .with_context(|| format!("spawn python for neuralop import check (python={python})"))?;

    if out.status.success() {
        return Ok(());
    }

    let mut msg = String::from_utf8_lossy(&out.stderr).to_string();
    msg = msg.trim().to_string();
    if msg.is_empty() {
        msg = String::from_utf8_lossy(&out.stdout).trim().to_string();
    }
    if msg.is_empty() {
        msg = "Failed to import neuralop.models.FNO1d/FNO (no stderr)".to_string();
    }

    Err(anyhow!(msg))
}

fn python_runtime_fingerprint(python: &str) -> Value {
    let code = r#"import json, sys, platform
data = {
  "executable": sys.executable,
  "version": sys.version,
  "version_info": list(sys.version_info),
  "platform": platform.platform(),
}
try:
  import torch
  data["torch"] = {
    "version": getattr(torch, "__version__", None),
    "cuda_available": bool(torch.cuda.is_available()),
  }
  if torch.cuda.is_available():
    try:
      data["torch"]["cuda_device_count"] = int(torch.cuda.device_count())
      data["torch"]["cuda_device_name_0"] = str(torch.cuda.get_device_name(0)) if torch.cuda.device_count() > 0 else None
    except Exception as e:
      data["torch"]["cuda_probe_error"] = repr(e)
except Exception as e:
  data["torch_error"] = repr(e)

try:
  import neuralop
  data["neuralop"] = {"version": getattr(neuralop, "__version__", None)}
  try:
    try:
      from neuralop.models import FNO1d  # noqa: F401
      data["neuralop"]["fno1d_import_ok"] = True
    except Exception as e:
      data["neuralop"]["fno1d_import_ok"] = False
      data["neuralop"]["fno1d_import_error"] = repr(e)

    try:
      from neuralop.models import FNO  # noqa: F401
      data["neuralop"]["fno_import_ok"] = True
    except Exception as e:
      data["neuralop"]["fno_import_ok"] = False
      data["neuralop"]["fno_import_error"] = repr(e)
  except Exception as e:
    data["neuralop"]["import_probe_error"] = repr(e)
except Exception as e:
  data["neuralop_error"] = repr(e)

print(json.dumps(data))
"#;

    let out = Command::new(python).arg("-c").arg(code).output();
    match out {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if let Ok(v) = serde_json::from_str::<Value>(&stdout) {
                return v;
            }
            serde_json::json!({
                "python": python,
                "status": if out.status.success() { "ok" } else { "error" },
                "stdout": stdout,
                "stderr": String::from_utf8_lossy(&out.stderr).trim().to_string()
            })
        }
        Err(e) => serde_json::json!({
            "python": python,
            "status": "spawn_failed",
            "error": e.to_string()
        }),
    }
}

fn spawn_python_proxy(cfg: &BackendConfig, stdout: Arc<Mutex<io::Stdout>>) -> Result<PythonProxy> {
    let script = locate_python_bridge_script()
        .ok_or_else(|| anyhow!("python-bridge.py not found (set STEMSEP_PYTHON_BRIDGE)"))?;
    let python = locate_python_exe();

    let mut child = Command::new(python)
        .arg(script)
        .arg("--models-dir")
        .arg(cfg.models_dir.to_string_lossy().to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn python bridge")?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("python stdin missing"))?;
    let child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("python stdout missing"))?;
    let child_stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("python stderr missing"))?;

    let stdin_arc = Arc::new(Mutex::new(stdin));

    // Forward python stdout -> our stdout (filter duplicate bridge_ready)
    {
        let out = stdout.clone();
        thread::spawn(move || {
            let reader = io::BufReader::new(child_stdout);
            for line in reader.lines().flatten() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if v.get("type").and_then(|t| t.as_str()) == Some("bridge_ready") {
                        continue;
                    }
                }
                let _ = write_raw_line_locked(&out, &line);
            }
        });
    }

    // Forward python stderr -> our stderr (best-effort)
    thread::spawn(move || {
        let reader = io::BufReader::new(child_stderr);
        for line in reader.lines().flatten() {
            eprintln!("[python-bridge] {line}");
        }
    });

    Ok(PythonProxy {
        _child: child,
        stdin: stdin_arc,
    })
}

struct SeparationJob {
    process: Child,
}

#[derive(Debug, Clone)]
struct RecipeContext {
    id: String,
}

fn maybe_recipe_context(cfg: &BackendConfig, model_id: Option<&str>) -> Option<RecipeContext> {
    let model_id = model_id?.trim();
    if model_id.is_empty() {
        return None;
    }
    let recipes_json = read_json_file(&cfg.assets_dir.join("recipes.json")).ok()?;
    let is_recipe = recipes_json
        .get("recipes")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|r| r.get("id").and_then(|v| v.as_str()) == Some(model_id))
        })
        .is_some();
    if is_recipe {
        Some(RecipeContext {
            id: model_id.to_string(),
        })
    } else {
        None
    }
}

fn parse_step_meta_from_message(message: &str) -> Option<Map<String, Value>> {
    let msg = message.trim();
    if msg.is_empty() {
        return None;
    }

    // Pipeline: "Running step 1: <step_name> (<model_id>)"
    if let Some(rest) = msg.strip_prefix("Running step ") {
        if let Some(colon) = rest.find(':') {
            let head = rest[..colon].trim();
            let step_num: i64 = head
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<i64>().ok())
                .unwrap_or(0);
            let tail = rest[colon + 1..].trim();

            let (step_name, model_id) = if tail.ends_with(')') {
                if let Some(open) = tail.rfind('(') {
                    let name = tail[..open].trim();
                    let mid = tail[open + 1..tail.len() - 1].trim();
                    (
                        name.to_string(),
                        if mid.is_empty() {
                            None
                        } else {
                            Some(mid.to_string())
                        },
                    )
                } else {
                    (tail.to_string(), None)
                }
            } else {
                (tail.to_string(), None)
            };

            let mut step = Map::new();
            if step_num > 0 {
                step.insert("number".to_string(), Value::from(step_num));
                step.insert("index".to_string(), Value::from(step_num - 1));
            }
            if !step_name.is_empty() {
                step.insert("name".to_string(), Value::String(step_name));
            }
            if let Some(mid) = model_id {
                step.insert("model_id".to_string(), Value::String(mid));
            }

            let mut meta = Map::new();
            meta.insert("phase".to_string(), Value::String("pipeline".to_string()));
            meta.insert("step".to_string(), Value::Object(step));
            return Some(meta);
        }
    }

    // Ensemble: "Running model 1/2: <model_id>"
    if let Some(rest) = msg.strip_prefix("Running model ") {
        if let Some(colon) = rest.find(':') {
            let head = rest[..colon].trim();
            let idx_total = head.split_once('/').and_then(|(a, b)| {
                Some((a.trim().parse::<i64>().ok()?, b.trim().parse::<i64>().ok()?))
            });
            let model_id = rest[colon + 1..].trim();

            let mut step = Map::new();
            if let Some((i, t)) = idx_total {
                if i > 0 {
                    step.insert("number".to_string(), Value::from(i));
                    step.insert("index".to_string(), Value::from(i - 1));
                }
                if t > 0 {
                    step.insert("total".to_string(), Value::from(t));
                }
            }
            if !model_id.is_empty() {
                step.insert("model_id".to_string(), Value::String(model_id.to_string()));
            }

            let mut meta = Map::new();
            meta.insert("phase".to_string(), Value::String("ensemble".to_string()));
            meta.insert("step".to_string(), Value::Object(step));
            return Some(meta);
        }
    }

    // Post-processing: best-effort model_id extraction from parentheses.
    if msg.starts_with("Post-processing:") {
        let mut meta = Map::new();
        meta.insert(
            "phase".to_string(),
            Value::String("post_processing".to_string()),
        );
        if msg.ends_with(')') {
            if let Some(open) = msg.rfind('(') {
                let mid = msg[open + 1..msg.len() - 1].trim();
                if !mid.is_empty() {
                    let mut step = Map::new();
                    step.insert("model_id".to_string(), Value::String(mid.to_string()));
                    meta.insert("step".to_string(), Value::Object(step));
                }
            }
        }
        return Some(meta);
    }

    None
}

#[derive(Debug, Clone)]
struct ActiveStep {
    key: String,
    started_ts: f64,
    meta: Value,
}

#[derive(Debug, Default)]
struct StepSynthesizer {
    active: Option<ActiveStep>,
}

impl StepSynthesizer {
    fn step_key_from_meta(meta: &Map<String, Value>) -> Option<String> {
        let phase = meta.get("phase").and_then(|v| v.as_str())?;
        let step = meta.get("step").and_then(|v| v.as_object());

        let idx = step.and_then(|s| s.get("index")).and_then(|v| v.as_i64());
        let name = step
            .and_then(|s| s.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let model_id = step
            .and_then(|s| s.get("model_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        Some(format!("{phase}|{idx:?}|{name}|{model_id}"))
    }

    fn emit_started(
        stdout: &Arc<Mutex<io::Stdout>>,
        job_id: &str,
        progress: Option<f64>,
        meta: Value,
    ) {
        let mut evt = Map::new();
        evt.insert(
            "type".to_string(),
            Value::String("separation_step_started".to_string()),
        );
        evt.insert("job_id".to_string(), Value::String(job_id.to_string()));
        evt.insert("ts".to_string(), Value::from(now_ts_seconds()));
        if let Some(p) = progress {
            evt.insert("progress".to_string(), Value::from(p));
        }
        evt.insert("meta".to_string(), meta);
        send_event(stdout, Value::Object(evt));
    }

    fn emit_completed(
        stdout: &Arc<Mutex<io::Stdout>>,
        job_id: &str,
        progress: Option<f64>,
        meta: Value,
        started_ts: f64,
        status: &str,
    ) {
        let now = now_ts_seconds();
        let mut evt = Map::new();
        evt.insert(
            "type".to_string(),
            Value::String("separation_step_completed".to_string()),
        );
        evt.insert("job_id".to_string(), Value::String(job_id.to_string()));
        evt.insert("ts".to_string(), Value::from(now));
        evt.insert(
            "duration_seconds".to_string(),
            Value::from((now - started_ts).max(0.0)),
        );
        evt.insert("status".to_string(), Value::String(status.to_string()));
        if let Some(p) = progress {
            evt.insert("progress".to_string(), Value::from(p));
        }
        evt.insert("meta".to_string(), meta);
        send_event(stdout, Value::Object(evt));
    }

    fn on_progress(&mut self, stdout: &Arc<Mutex<io::Stdout>>, job_id: &str, event: &Value) {
        let progress = event.get("progress").and_then(|v| v.as_f64());
        let meta_val = event.get("meta").cloned().unwrap_or(Value::Null);
        let meta_obj = meta_val.as_object();
        let key = meta_obj.and_then(Self::step_key_from_meta);
        let Some(key) = key else {
            return;
        };

        match &self.active {
            None => {
                Self::emit_started(stdout, job_id, progress, meta_val.clone());
                self.active = Some(ActiveStep {
                    key,
                    started_ts: now_ts_seconds(),
                    meta: meta_val,
                });
            }
            Some(active) if active.key != key => {
                // Close previous step and open new.
                let prev = self.active.take().unwrap();
                Self::emit_completed(
                    stdout,
                    job_id,
                    progress,
                    prev.meta,
                    prev.started_ts,
                    "completed",
                );
                Self::emit_started(stdout, job_id, progress, meta_val.clone());
                self.active = Some(ActiveStep {
                    key,
                    started_ts: now_ts_seconds(),
                    meta: meta_val,
                });
            }
            Some(_) => {
                // Same step; do nothing.
            }
        }
    }

    fn finish(
        &mut self,
        stdout: &Arc<Mutex<io::Stdout>>,
        job_id: &str,
        event: &Value,
        status: &str,
    ) {
        let progress = event.get("progress").and_then(|v| v.as_f64());
        if let Some(prev) = self.active.take() {
            Self::emit_completed(stdout, job_id, progress, prev.meta, prev.started_ts, status);
        }
    }
}

#[derive(Clone, Default)]
struct SeparationManager {
    jobs: Arc<Mutex<HashMap<String, SeparationJob>>>,
}

impl SeparationManager {
    fn start(
        &self,
        job_id: String,
        config: Value,
        cfg: &BackendConfig,
        stdout: Arc<Mutex<io::Stdout>>,
    ) -> Result<()> {
        let script = locate_inference_script()
            .ok_or_else(|| anyhow!("inference.py not found (set STEMSEP_INFERENCE_SCRIPT)"))?;
        let python = locate_python_exe();

        let config_str = serde_json::to_string(&config).context("serialize job config")?;

        let mut child = Command::new(python)
            .arg("-u")
            .arg(script)
            .arg(config_str)
            .env("STEMSEP_MODELS_DIR", &cfg.models_dir) // Ensure subprocess sees correct models dir
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped()) // Not really used but good practice
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawn inference process")?;

        let child_stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("inference stdout missing"))?;
        let child_stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("inference stderr missing"))?;

        // Store job process for cancellation
        {
            let mut map = self.jobs.lock().expect("jobs lock");
            map.insert(job_id.clone(), SeparationJob { process: child });
        }

        let mgr = self.clone();
        let job_id_clone = job_id.clone();
        let stdout_clone = stdout.clone();

        // Heartbeat: emit synthetic progress events if we haven't received any real
        // separation_progress updates for a while (Roformer/MDXC can be quiet for minutes).
        // Default interval: 3s (requested). Set STEMSEP_PROGRESS_HEARTBEAT_MS=0 to disable.
        #[derive(Clone, Default)]
        struct HeartbeatState {
            last_progress_ts: f64,
            last_progress_value: Option<f64>,
        }

        let heartbeat_ms = env_u64("STEMSEP_PROGRESS_HEARTBEAT_MS", 15000);
        let heartbeat_interval = if heartbeat_ms == 0 {
            None
        } else {
            Some(Duration::from_millis(heartbeat_ms))
        };

        let hb_state = Arc::new(Mutex::new(HeartbeatState {
            last_progress_ts: now_ts_seconds(),
            last_progress_value: None,
        }));
        let hb_running = Arc::new(AtomicBool::new(true));

        if let Some(interval) = heartbeat_interval {
            let hb_stdout = stdout.clone();
            let hb_job_id = job_id.clone();
            let hb_state_clone = hb_state.clone();
            let hb_running_clone = hb_running.clone();

            thread::spawn(move || {
                while hb_running_clone.load(Ordering::Relaxed) {
                    thread::sleep(interval);
                    if !hb_running_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    let now = now_ts_seconds();
                    let (silence_seconds, progress_value_opt) = {
                        let mut st = hb_state_clone.lock().expect("hb_state lock");
                        let silence = (now - st.last_progress_ts).max(0.0);
                        if silence + 1e-9 >= (interval.as_millis() as f64 / 1000.0) {
                            // Throttle to at most once per interval.
                            st.last_progress_ts = now;
                            (silence, st.last_progress_value)
                        } else {
                            (0.0, st.last_progress_value)
                        }
                    };

                    if silence_seconds <= 0.0 {
                        continue;
                    }

                    let progress_value = progress_value_opt.unwrap_or(0.0);
                    let msg = format!(
                        "Still running… (no progress update for {:.0}s)",
                        silence_seconds
                    );

                    send_event(
                        &hb_stdout,
                        serde_json::json!({
                            "type": "separation_progress",
                            "job_id": hb_job_id,
                            "ts": now,
                            "progress": progress_value,
                            "message": msg,
                            "meta": {
                                "heartbeat": true,
                                "silence_seconds": silence_seconds
                            }
                        }),
                    );
                }
            });
        }

        let job_model_id = config
            .get("model_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let recipe_ctx = maybe_recipe_context(cfg, job_model_id.as_deref());

        // Stdout reader thread
        thread::spawn(move || {
            let reader = io::BufReader::new(child_stdout);
            let mut synth = StepSynthesizer::default();
            for line in reader.lines().flatten() {
                if line.trim().is_empty() {
                    continue;
                }

                // Try parsing as JSON event
                if let Ok(mut event) = serde_json::from_str::<Value>(&line) {
                    // Track last-seen real progress so the heartbeat can keep the UI alive.
                    if event.get("type").and_then(|v| v.as_str()) == Some("separation_progress") {
                        let now = now_ts_seconds();
                        let progress = event.get("progress").and_then(|v| v.as_f64());
                        let mut st = hb_state.lock().expect("hb_state lock");
                        st.last_progress_ts = now;
                        if let Some(p) = progress {
                            st.last_progress_value = Some(p);
                        }
                    }

                    // IMPORTANT: Always force the Rust job_id onto forwarded events.
                    // The Python runner may emit its own job_id; the renderer typically filters by
                    // the job_id returned by the start response, so mismatches make the UI look stuck.
                    if let Some(obj) = event.as_object_mut() {
                        let prev_job_id = obj.get("job_id").cloned();
                        let prev_job_id_str = prev_job_id
                            .as_ref()
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());

                        // Preserve original job id for debugging (if different).
                        if prev_job_id_str.as_deref() != Some(job_id_clone.as_str()) {
                            let meta_entry = obj
                                .entry("meta".to_string())
                                .or_insert_with(|| Value::Object(Map::new()));
                            if let Some(meta_obj) = meta_entry.as_object_mut() {
                                if let Some(s) = prev_job_id_str {
                                    meta_obj
                                        .entry("python_job_id".to_string())
                                        .or_insert_with(|| Value::String(s));
                                } else if let Some(v) = prev_job_id {
                                    meta_obj.entry("python_job_id_raw".to_string()).or_insert(v);
                                }
                            }

                            if env_true("STEMSEP_DEBUG_JOB_ID_MISMATCH", false) {
                                eprintln!(
                                    "[job_id_mismatch] forcing job_id={} (python had {:?})",
                                    job_id_clone,
                                    obj.get("job_id")
                                );
                            }
                        }

                        obj.insert("job_id".to_string(), Value::String(job_id_clone.clone()));

                        // Attach recipe id (if applicable) and step/phase metadata (best-effort)
                        let meta_entry = obj
                            .entry("meta".to_string())
                            .or_insert_with(|| Value::Object(Map::new()));
                        if let Some(meta_obj) = meta_entry.as_object_mut() {
                            if let Some(rc) = &recipe_ctx {
                                meta_obj
                                    .entry("recipe_id".to_string())
                                    .or_insert_with(|| Value::String(rc.id.clone()));
                            }
                        }

                        if obj.get("type").and_then(|v| v.as_str()) == Some("separation_progress") {
                            let message_owned = obj
                                .get("message")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());

                            if let Some(message) = message_owned {
                                if let Some(step_meta) = parse_step_meta_from_message(&message) {
                                    // Merge parsed meta into event.meta
                                    let meta_entry = obj
                                        .entry("meta".to_string())
                                        .or_insert_with(|| Value::Object(Map::new()));
                                    if let Some(meta_obj) = meta_entry.as_object_mut() {
                                        for (k, v) in step_meta {
                                            meta_obj.insert(k, v);
                                        }
                                    }
                                }

                                // Make Python's one-shot output-validation retry explicit in logs.
                                // The runner emits a separation_progress message like:
                                // "Output validation failed; retrying once: ..." with meta.output_validation.
                                let meta_entry = obj
                                    .entry("meta".to_string())
                                    .or_insert_with(|| Value::Object(Map::new()));
                                if let Some(meta_obj) = meta_entry.as_object_mut() {
                                    let attempt = meta_obj.get("attempt").and_then(|v| v.as_i64());
                                    let validation = meta_obj
                                        .get("output_validation")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string());

                                    let looks_like_retry = message
                                        .contains("Output validation failed")
                                        || message.contains("retrying once")
                                        || (message.contains("retry") && validation.is_some());

                                    if looks_like_retry {
                                        meta_obj
                                            .entry("auto_retry".to_string())
                                            .or_insert_with(|| Value::Bool(true));
                                        if let Some(v) = &validation {
                                            meta_obj
                                                .entry("auto_retry_reason".to_string())
                                                .or_insert_with(|| Value::String(v.clone()));
                                        }

                                        eprintln!(
                                            "[auto_retry] job_id={} attempt={:?} reason={:?}",
                                            job_id_clone,
                                            attempt,
                                            validation.as_deref()
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Emit explicit step events (best-effort) based on parsed meta.
                    match event.get("type").and_then(|v| v.as_str()) {
                        Some("separation_progress") => {
                            synth.on_progress(&stdout_clone, &job_id_clone, &event);
                        }
                        Some("separation_complete") => {
                            hb_running.store(false, Ordering::Relaxed);
                            synth.finish(&stdout_clone, &job_id_clone, &event, "completed");
                        }
                        Some("separation_error") => {
                            hb_running.store(false, Ordering::Relaxed);
                            synth.finish(&stdout_clone, &job_id_clone, &event, "failed");
                        }
                        Some("separation_cancelled") => {
                            hb_running.store(false, Ordering::Relaxed);
                            synth.finish(&stdout_clone, &job_id_clone, &event, "cancelled");
                        }
                        _ => {}
                    }
                    send_event(&stdout_clone, event);
                } else {
                    // Fallback: log raw lines as debug/verbose progress
                    eprintln!("[inference-stdout] {}", line);
                }
            }

            hb_running.store(false, Ordering::Relaxed);

            // Cleanup when process exits
            mgr.clear_job(&job_id_clone);
        });

        // Stderr reader thread (logging)
        let job_id_log = job_id.clone();
        thread::spawn(move || {
            let reader = io::BufReader::new(child_stderr);
            for line in reader.lines().flatten() {
                eprintln!("[inference-stderr:{}] {}", job_id_log, line);
            }
        });

        Ok(())
    }

    fn cancel(&self, job_id: &str) -> bool {
        let mut map = self.jobs.lock().expect("jobs lock");
        if let Some(job) = map.get_mut(job_id) {
            let _ = job.process.kill();
            // We don't remove it here immediately; the stdout reader will detect exit and call clear_job.
            // But to be safe/responsive, we can assume it's gone.
            // Actually, best to let the cleanup happen naturally or force remove if we want to ensure immediate sync.
            // For now, killing is enough.
            true
        } else {
            false
        }
    }

    fn clear_job(&self, job_id: &str) {
        let mut map = self.jobs.lock().expect("jobs lock");
        map.remove(job_id);
    }
}

fn main() -> Result<()> {
    let args: Vec<OsString> = std::env::args_os().collect();
    let assets_dir = locate_assets_dir(parse_arg_value(&args, "--assets-dir"))?;
    let models_dir = locate_models_dir(parse_arg_value(&args, "--models-dir"))?;
    let cfg = BackendConfig {
        assets_dir,
        models_dir,
    };

    let stdin = io::stdin();
    let stdout = Arc::new(Mutex::new(io::stdout()));

    let downloads = DownloadManager::default();
    let separation_manager = SeparationManager::default();

    // Lazy python proxy (only spawned when needed)
    let mut py_proxy: Option<PythonProxy> = None;
    let enable_py_proxy = env_true("STEMSEP_PROXY_PYTHON", true);
    let prefer_rust_separation = env_true("STEMSEP_PREFER_RUST_SEPARATION", false);
    let _enable_dummy_separation = env_true("STEMSEP_ENABLE_DUMMY_SEPARATION", false);

    // Emit a bridge_ready event immediately so the Electron main process can
    // treat this as a drop-in replacement (even while separation is stubbed).
    let models_path = cfg.assets_dir.join("models.json.bak");
    let recipes_path = cfg.assets_dir.join("recipes.json");
    let models_count = read_json_file(&models_path)
        .ok()
        .and_then(|v| v.get("models").and_then(|m| m.as_array()).map(|a| a.len()))
        .unwrap_or(0);
    let recipes_count = read_json_file(&recipes_path)
        .ok()
        .and_then(|v| v.get("recipes").and_then(|m| m.as_array()).map(|a| a.len()))
        .unwrap_or(0);
    let gpu_info = detect_gpus();
    let has_gpu = gpu_info
        .get("gpus")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    let python = locate_python_exe();
    let runtime_fingerprint = python_runtime_fingerprint(&python);
    eprintln!(
        "[runtime_fingerprint] {}",
        serde_json::to_string(&runtime_fingerprint).unwrap_or_else(|_| "{}".to_string())
    );

    let mut caps: Vec<Value> = vec![];
    if models_count > 0 {
        caps.push(Value::from("models"));
    }
    if recipes_count > 0 {
        caps.push(Value::from("recipes"));
    }
    if has_gpu {
        caps.push(Value::from("gpu"));
    }
    // Separation is currently provided via Python proxy (until fully ported).
    if enable_py_proxy {
        caps.push(Value::from("separation"));
    }

    let ready = serde_json::json!({
        "type": "bridge_ready",
        "capabilities": caps,
        "models_count": models_count,
        "recipes_count": recipes_count,
        "backend": {
            "name": "stemsep-backend",
            "version": env!("CARGO_PKG_VERSION")
        }
    });
    write_json_line_locked(&stdout, &ready)?;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let parsed: Result<RequestEnvelope> = serde_json::from_str(&line).context("parse request");
        let req = match parsed {
            Ok(v) => v,
            Err(err) => {
                // No id available; ignore malformed line.
                let _ = write_json_line_locked(
                    &stdout,
                    &serde_json::json!({
                        "type": "backend_warning",
                        "message": format!("malformed request: {err:#}")
                    }),
                );
                continue;
            }
        };

        // Proxy most commands to Python until fully ported.
        // Rationale: keep UI unchanged and ensure end-to-end functionality while migrating.
        let should_proxy = enable_py_proxy
            && matches!(
                req.command.as_str(),
                "download_model"
                    | "pause_download"
                    | "resume_download"
                    | "remove_model"
                    | "import_custom_model"
                    | "resolve_youtube"
                    | "model_preflight"
                    | "separation_preflight"
                    // "separate_audio"  <-- Handled natively now
                    // "cancel_job"      <-- Handled natively now
                    | "pause_queue"
                    | "resume_queue"
                    | "reorder_queue"
                    | "check_memory"
                    | "export_output"
                    | "export_files"
                    | "discard_output"
            );

        // Allow opting into Rust-native separation while keeping other commands proxied.
        let should_proxy = if prefer_rust_separation {
            should_proxy
                && !matches!(
                    req.command.as_str(),
                    "separation_preflight"
                        | "separate_audio"
                        | "model_preflight"
                        | "download_model"
                )
        } else {
            should_proxy
        };

        if should_proxy {
            if py_proxy.is_none() {
                match spawn_python_proxy(&cfg, stdout.clone()) {
                    Ok(p) => py_proxy = Some(p),
                    Err(e) => {
                        // For id-based requests: send a response. For id-less commands, emit a generic failure response.
                        let msg = format!("Python proxy unavailable: {e:#}");
                        let _ = send_err(&stdout, req.id.clone(), "BACKEND_UNAVAILABLE", &msg);
                        continue;
                    }
                }
            }
            if let Some(proxy) = &py_proxy {
                let mut inw = proxy.stdin.lock().expect("python stdin lock");
                if let Err(e) = inw
                    .write_all(line.as_bytes())
                    .and_then(|_| inw.write_all(b"\n"))
                {
                    let msg = format!("Failed to write to python bridge: {e}");
                    let _ = send_err(&stdout, req.id.clone(), "BACKEND_IO", &msg);
                }
                inw.flush().ok();
            }
            continue;
        }

        match req.command.as_str() {
            "ping" => {
                send_ok(
                    &stdout,
                    req.id,
                    serde_json::json!({
                        "status": "ok",
                        "ok": true,
                        "rust_backend": true,
                        "has_gpu": has_gpu,
                    }),
                )?;
            }
            "get_runtime_fingerprint" => {
                send_ok(&stdout, req.id, runtime_fingerprint.clone())?;
            }
            "get_models" => {
                let models_json = read_json_file(&cfg.assets_dir.join("models.json.bak"))?;
                let models = models_json
                    .get("models")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();

                let mut out: Vec<Value> = Vec::with_capacity(models.len());
                for m in models {
                    let id = m.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let installed = if !id.is_empty() {
                        any_model_file_exists(&cfg.models_dir, id, &m)
                    } else {
                        false
                    };
                    let mut obj = m.as_object().cloned().unwrap_or_default();
                    obj.insert("installed".to_string(), Value::from(installed));
                    out.push(Value::Object(obj));
                }

                send_ok(&stdout, req.id, Value::Array(out))?;
            }
            "get_recipes" => {
                let recipes_json = read_json_file(&cfg.assets_dir.join("recipes.json"))?;
                let recipes = recipes_json
                    .get("recipes")
                    .cloned()
                    .unwrap_or(Value::Array(vec![]));
                send_ok(&stdout, req.id, recipes)?;
            }
            "get-gpu-devices" | "get_gpu_devices" => {
                send_ok(&stdout, req.id, gpu_info.clone())?;
            }
            "download_model" => {
                // Rust-native model download (background), emits {type:progress|complete|error} events.
                // IMPORTANT: never crash the backend on bad inputs; return an error response instead.
                let req_id = req.id.clone();
                let result: Result<()> = (|| {
                    let model_id = req
                        .extra
                        .get("model_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow!("model_id is required"))?
                        .to_string();

                    let (ckpt_url, cfg_url) = lookup_model_links(&cfg, &model_id)?;
                    let ckpt_url = ckpt_url.ok_or_else(|| {
                        anyhow!("No checkpoint URL found for model_id: {model_id}")
                    })?;

                    let (artifact_ckpt, _artifact_cfg) =
                        lookup_model_artifact_filenames(&cfg, &model_id).unwrap_or((None, None));

                    let ckpt_dest = if ckpt_url.to_lowercase().contains(".onnx") {
                        cfg.models_dir.join(format!("{model_id}.onnx"))
                    } else if let Some(fname) = &artifact_ckpt {
                        cfg.models_dir.join(fname)
                    } else if let Some(base) = url_basename(&ckpt_url) {
                        cfg.models_dir.join(base)
                    } else {
                        cfg.models_dir.join(format!("{model_id}.bin"))
                    };

                    let mut files: Vec<DownloadFile> = vec![DownloadFile {
                        url: ckpt_url.clone(),
                        dest_path: ckpt_dest.clone(),
                    }];
                    if let Some(u) = cfg_url {
                        let cfg_dest = match url_basename(&u) {
                            Some(base) if base.to_lowercase().ends_with(".yml") => {
                                cfg.models_dir.join(format!("{model_id}.yml"))
                            }
                            _ => cfg.models_dir.join(format!("{model_id}.yaml")),
                        };
                        files.push(DownloadFile {
                            url: u,
                            dest_path: cfg_dest,
                        });
                    }

                    downloads.start(&model_id, files.clone(), stdout.clone());

                    // Respond immediately so Electron IPC resolves; events continue in background.
                    send_ok(
                        &stdout,
                        req_id,
                        serde_json::json!({
                            "scheduled": true,
                            "model_id": model_id,
                            "dest": ckpt_dest.to_string_lossy(),
                            "files": files
                                .iter()
                                .map(|f| f.dest_path.to_string_lossy().to_string())
                                .collect::<Vec<_>>()
                        }),
                    )?;
                    Ok(())
                })();

                if let Err(e) = result {
                    let _ = send_err(&stdout, req.id, "DOWNLOAD_FAILED", &format!("{e:#}"));
                }
            }
            "pause_download" => {
                let model_id = req
                    .extra
                    .get("model_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow!("model_id is required"))?;
                let requested = downloads.request_pause(model_id);
                send_ok(
                    &stdout,
                    req.id,
                    serde_json::json!({
                        "model_id": model_id,
                        "requested": requested
                    }),
                )?;
            }
            "resume_download" => {
                // IMPORTANT: never crash the backend on bad inputs; return an error response instead.
                let req_id = req.id.clone();
                let result: Result<()> = (|| {
                    let model_id = req
                        .extra
                        .get("model_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow!("model_id is required"))?
                        .to_string();

                    let resumed = downloads.resume(&model_id, stdout.clone());
                    if !resumed {
                        // Fresh start
                        let (ckpt_url, cfg_url) = lookup_model_links(&cfg, &model_id)?;
                        let ckpt_url = ckpt_url.ok_or_else(|| {
                            anyhow!("No checkpoint URL found for model_id: {model_id}")
                        })?;

                        let (artifact_ckpt, _artifact_cfg) =
                            lookup_model_artifact_filenames(&cfg, &model_id).unwrap_or((None, None));
                        let ckpt_dest = if ckpt_url.to_lowercase().contains(".onnx") {
                            cfg.models_dir.join(format!("{model_id}.onnx"))
                        } else if let Some(fname) = &artifact_ckpt {
                            cfg.models_dir.join(fname)
                        } else if let Some(base) = url_basename(&ckpt_url) {
                            cfg.models_dir.join(base)
                        } else {
                            cfg.models_dir.join(format!("{model_id}.bin"))
                        };
                        let mut files: Vec<DownloadFile> = vec![DownloadFile {
                            url: ckpt_url.clone(),
                            dest_path: ckpt_dest,
                        }];
                        if let Some(u) = cfg_url {
                            let cfg_dest = match url_basename(&u) {
                                Some(base) if base.to_lowercase().ends_with(".yml") => {
                                    cfg.models_dir.join(format!("{model_id}.yml"))
                                }
                                _ => cfg.models_dir.join(format!("{model_id}.yaml")),
                            };
                            files.push(DownloadFile {
                                url: u,
                                dest_path: cfg_dest,
                            });
                        }
                        downloads.start(&model_id, files, stdout.clone());
                    }

                    send_ok(
                        &stdout,
                        req_id,
                        serde_json::json!({
                            "model_id": model_id,
                            "scheduled": true
                        }),
                    )?;
                    Ok(())
                })();

                if let Err(e) = result {
                    let _ = send_err(&stdout, req.id, "DOWNLOAD_FAILED", &format!("{e:#}"));
                }
            }
            "remove_model" => {
                // IMPORTANT: never crash the backend on bad inputs or filesystem errors.
                let req_id = req.id.clone();
                let result: Result<()> = (|| {
                    let model_id = req.extra.get("model_id").and_then(|v| v.as_str());
                    if model_id.is_none() {
                        send_err(&stdout, req_id, "INVALID", "model_id is required")?;
                        return Ok(());
                    }
                    let model_id = model_id.unwrap();

                    // Cancel any active download and drop task tracking.
                    let _ = downloads.request_pause(model_id);
                    downloads.clear_task(model_id);

                    let removed = remove_model_files(&cfg, model_id)?;
                    send_ok(
                        &stdout,
                        req_id,
                        serde_json::json!({
                            "model_id": model_id,
                            "removed_files": removed
                        }),
                    )?;
                    Ok(())
                })();

                if let Err(e) = result {
                    let _ = send_err(&stdout, req.id, "REMOVE_FAILED", &format!("{e:#}"));
                }
            }
            "introspect_model" => {
                let model_path = req
                    .extra
                    .get("model_path")
                    .and_then(|v| v.as_str())
                    .map(PathBuf::from)
                    .or_else(|| {
                        req.extra
                            .get("model_id")
                            .and_then(|v| v.as_str())
                            .map(|id| cfg.models_dir.join(format!("{id}.onnx")))
                    })
                    .ok_or_else(|| anyhow!("model_path or model_id is required"))?;

                if !model_path.exists() {
                    send_err(
                        &stdout,
                        req.id,
                        "NOT_FOUND",
                        &format!("Model file not found: {}", model_path.display()),
                    )?;
                    continue;
                }

                #[cfg(feature = "tract")]
                {
                    let info = tract_introspect_onnx(&model_path)?;
                    send_ok(&stdout, req.id, info)?;
                }

                #[cfg(not(feature = "tract"))]
                {
                    send_err(
                        &stdout,
                        req.id,
                        "NOT_SUPPORTED",
                        "introspect_model requires the `tract` feature (rebuild backend with --features tract)",
                    )?;
                }
            }
            "separation_preflight" => {
                // Contract-compatible preflight: returns a report object even when inputs are
                // missing/invalid. (Matches Python bridge behavior: success=true with can_proceed=false.)
                let file_path = req.extra.get("file_path").and_then(|v| v.as_str());
                let model_id = req.extra.get("model_id").and_then(|v| v.as_str());
                let stems_val = req.extra.get("stems").cloned().unwrap_or(Value::Null);
                let device_req = req
                    .extra
                    .get("device")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                // Flexible numeric parsing for overlap and segment_size (handle strings from UI)
                let overlap_req = req.extra.get("overlap").and_then(|v| {
                    if let Some(f) = v.as_f64() { Some(f) }
                    else if let Some(s) = v.as_str() { s.parse::<f64>().ok() }
                    else { None }
                });
                let segment_size_req = req.extra.get("segment_size").and_then(|v| {
                    if let Some(i) = v.as_i64() { Some(i) }
                    else if let Some(s) = v.as_str() { s.parse::<i64>().ok() }
                    else { None }
                });

                let batch_size_req = req.extra.get("batch_size").and_then(|v| v.as_i64());
                let tta_req = req
                    .extra
                    .get("tta")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let output_format_req = req
                    .extra
                    .get("output_format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("wav")
                    .to_string();
                let shifts_req = req.extra.get("shifts").and_then(|v| v.as_i64());
                let bitrate_req = req
                    .extra
                    .get("bitrate")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                // Ensemble/phase-fix fields are passed through in `resolved` so the UI has a single place
                // to read final values.
                let ensemble_cfg_val = req.extra.get("ensemble_config").or_else(|| req.extra.get("ensembleConfig"));

                let (ensemble_algorithm, phase_fix_enabled, phase_fix_params) =
                    match ensemble_cfg_val {
                        Some(Value::Object(o)) => {
                            let alg = o
                                .get("algorithm")
                                .and_then(|v| v.as_str())
                                .unwrap_or("average")
                                .to_string();
                            let phase_enabled = o
                                .get("phaseFixEnabled")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            let params = o.get("phaseFixParams").cloned().unwrap_or(Value::Null);
                            (alg, phase_enabled, params)
                        }
                        _ => {
                            let alg = req
                                .extra
                                .get("ensemble_algorithm")
                                .and_then(|v| v.as_str())
                                .unwrap_or("average")
                                .to_string();
                            // Legacy phase_params override
                            let (enabled, params) = match req.extra.get("phase_params") {
                                Some(Value::Object(p))
                                    if p.get("enabled").and_then(|v| v.as_bool()) == Some(true) =>
                                {
                                    (true, Value::Object(p.clone()))
                                }
                                _ => (false, Value::Null),
                            };
                            (alg, enabled, params)
                        }
                    };

                let mut errors: Vec<String> = Vec::new();
                let mut warnings: Vec<String> = Vec::new();
                let mut missing_models: Vec<Value> = Vec::new();

                // Validate audio path (minimal: existence + WAV support note)
                let audio_info = match file_path {
                    None => {
                        errors.push("Missing required parameter: file_path".to_string());
                        Value::Null
                    }
                    Some(p) => {
                        let fp = PathBuf::from(p);
                        if !fp.exists() {
                            errors.push(format!("Input file not found: {p}"));
                            Value::Null
                        } else {
                            let ext = fp
                                .extension()
                                .and_then(|e| e.to_str())
                                .unwrap_or("")
                                .to_lowercase();
                            serde_json::json!({
                                "valid": true,
                                "path": fp.to_string_lossy(),
                                "ext": ext
                            })
                        }
                    }
                };

                // Resolve model presence.
                // IMPORTANT: Separation is currently performed via the Python proxy; we should not
                // hard-require an ONNX artifact ("<model_id>.onnx"). Instead, we check whether the
                // requested model(s) appear installed in models_dir (checkpoint/config with common extensions).
                //
                // The UI may send `model_id: "ensemble"` as a virtual ID when `ensemble_config` is present.
                let models_json =
                    read_json_file(&cfg.assets_dir.join("models.json.bak")).unwrap_or(Value::Null);
                let recipes_json =
                    read_json_file(&cfg.assets_dir.join("recipes.json")).unwrap_or(Value::Null);

                let mut resolved_model_id: Option<String> = None;
                let mut requested_model_ids: Vec<String> = Vec::new();

                // If model_id refers to a recipe, preflight should validate the recipe's step model dependencies
                // rather than treating the recipe id itself as a downloadable model artifact.
                let mut recipe_step_model_ids: Vec<String> = Vec::new();
                let mut recipe_type: Option<String> = None;
                let mut recipe_plan: Option<Value> = None;
                if let Some(id) = model_id {
                    if let Some(recipe) = recipes_json
                        .get("recipes")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| {
                            arr.iter()
                                .find(|r| r.get("id").and_then(|v| v.as_str()) == Some(id))
                        })
                    {
                        recipe_type = recipe
                            .get("type")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        // Capture a lightweight plan for UI/debugging.
                        let mut steps_out: Vec<Value> = Vec::new();
                        if let Some(steps) = recipe.get("steps").and_then(|v| v.as_array()) {
                            for step in steps {
                                let step_name = step
                                    .get("step_name")
                                    .and_then(|v| v.as_str())
                                    .or_else(|| step.get("name").and_then(|v| v.as_str()))
                                    .unwrap_or("step");
                                let input_source = step
                                    .get("input_source")
                                    .and_then(|v| v.as_str())
                                    .or_else(|| step.get("input_from").and_then(|v| v.as_str()));
                                let output = step.get("output").cloned().unwrap_or(Value::Null);
                                let action = step.get("action").and_then(|v| v.as_str());
                                let model_id_step = step.get("model_id").and_then(|v| v.as_str());
                                let source_model = step
                                    .get("source_model")
                                    .and_then(|v| v.as_str())
                                    .or_else(|| step.get("sourceModel").and_then(|v| v.as_str()));

                                // Track referenced models (model_id + phase_fix source_model)
                                if let Some(mid) = model_id_step {
                                    let mid = mid.trim();
                                    if !mid.is_empty()
                                        && !recipe_step_model_ids.iter().any(|m| m == mid)
                                    {
                                        recipe_step_model_ids.push(mid.to_string());
                                    }
                                }
                                if let Some(mid) = source_model {
                                    let mid = mid.trim();
                                    if !mid.is_empty()
                                        && !recipe_step_model_ids.iter().any(|m| m == mid)
                                    {
                                        recipe_step_model_ids.push(mid.to_string());
                                    }
                                }

                                // Emit a lightweight plan row when the step references a model.
                                if model_id_step.is_some() || source_model.is_some() {
                                    let mut obj = Map::new();
                                    obj.insert(
                                        "step_name".to_string(),
                                        Value::String(step_name.to_string()),
                                    );
                                    if let Some(a) = action {
                                        obj.insert(
                                            "action".to_string(),
                                            Value::String(a.to_string()),
                                        );
                                    }
                                    if let Some(mid) = model_id_step {
                                        obj.insert(
                                            "model_id".to_string(),
                                            Value::String(mid.to_string()),
                                        );
                                    }
                                    if let Some(mid) = source_model {
                                        obj.insert(
                                            "source_model".to_string(),
                                            Value::String(mid.to_string()),
                                        );
                                    }
                                    if let Some(src) = input_source {
                                        obj.insert(
                                            "input_source".to_string(),
                                            Value::String(src.to_string()),
                                        );
                                    }
                                    if !output.is_null() {
                                        obj.insert("output".to_string(), output);
                                    }
                                    steps_out.push(Value::Object(obj));
                                }
                            }
                        }

                        let mut plan = Map::new();
                        plan.insert("id".to_string(), Value::String(id.to_string()));
                        if let Some(t) = &recipe_type {
                            plan.insert("type".to_string(), Value::String(t.clone()));
                        }
                        if let Some(defaults) = recipe.get("defaults") {
                            plan.insert("defaults".to_string(), defaults.clone());
                        }
                        if let Some(pp) = recipe.get("post_processing") {
                            plan.insert("post_processing".to_string(), pp.clone());
                        }
                        if !steps_out.is_empty() {
                            plan.insert("steps".to_string(), Value::Array(steps_out));
                        }
                        recipe_plan = Some(Value::Object(plan));
                    }
                }

                if let Some(Value::Object(o)) = ensemble_cfg_val {
                    // Expected shape from UI: { models: [{model_id, weight?}, ...], algorithm, ... }
                    if let Some(Value::Array(models)) = o.get("models") {
                        for m in models {
                            if let Some(mid) = m.get("model_id").and_then(|v| v.as_str()) {
                                requested_model_ids.push(mid.to_string());
                            }
                        }
                    }
                    // Keep `model_id` as provided by the caller if present, but treat "ensemble" as virtual.
                    if let Some(id) = model_id {
                        resolved_model_id = Some(id.to_string());
                    } else {
                        resolved_model_id = Some("ensemble".to_string());
                    }
                } else if let Some(id) = model_id {
                    resolved_model_id = Some(id.to_string());
                    if !recipe_step_model_ids.is_empty() {
                        // Treat as recipe id; validate referenced model_ids from steps.
                        requested_model_ids.extend(recipe_step_model_ids);
                        if let Some(t) = recipe_type {
                            warnings.push(format!("Resolved model_id '{id}' as recipe (type={t})"));
                        } else {
                            warnings.push(format!("Resolved model_id '{id}' as recipe"));
                        }
                    } else {
                        requested_model_ids.push(id.to_string());
                    }
                } else {
                    errors.push("Missing required parameter: model_id".to_string());
                }

                // Post-processing pipeline steps may reference additional models (e.g. Phase Fix reference).
                // UI payloads may use either snake_case or camelCase for this field; accept both.
                let post_steps_val = req
                    .extra
                    .get("post_processing_steps")
                    .or_else(|| req.extra.get("postProcessingSteps"));
                if let Some(Value::Array(steps)) = post_steps_val {
                    for step in steps {
                        if let Some(obj) = step.as_object() {
                            let mid = obj
                                .get("model_id")
                                .and_then(|v| v.as_str())
                                .or_else(|| obj.get("modelId").and_then(|v| v.as_str()));
                            if let Some(mid) = mid {
                                let mid = mid.trim();
                                if !mid.is_empty() && !requested_model_ids.iter().any(|m| m == mid)
                                {
                                    requested_model_ids.push(mid.to_string());
                                }
                            }
                        }
                    }
                }

                if requested_model_ids.is_empty() {
                    // If the UI used the virtual model_id but didn't include ensemble_config models, fail clearly.
                    if resolved_model_id.as_deref() == Some("ensemble") {
                        errors
                            .push("model_id=ensemble requires ensemble_config.models".to_string());
                    }
                }

                // For each referenced model_id, check if any expected local file exists.
                // This mirrors Python's ability to load checkpoints/configs (ckpt/pth/yaml/onnx/etc).
                let mut needs_fno_dependency = false;
                for mid in &requested_model_ids {
                    let model_obj = models_json
                        .get("models")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| {
                            arr.iter()
                                .find(|m| m.get("id").and_then(|v| v.as_str()) == Some(mid))
                        })
                        .cloned()
                        .unwrap_or(Value::Null);

                    // Detect FNO models by runtime.variant
                    if let Some(variant) = model_obj
                        .get("runtime")
                        .and_then(|v| v.as_object())
                        .and_then(|o| o.get("variant"))
                        .and_then(|v| v.as_str())
                    {
                        if variant.trim().eq_ignore_ascii_case("fno") {
                            needs_fno_dependency = true;
                        }
                    }

                    let exists = any_model_file_exists(&cfg.models_dir, mid, &model_obj);
                    if !exists {
                        missing_models.push(Value::String(mid.clone()));
                    }
                }

                if !missing_models.is_empty() {
                    errors.push(format!(
                        "Missing model file(s) for: {} (models_dir={})",
                        missing_models
                            .iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(", "),
                        cfg.models_dir.display()
                    ));
                }

                // FNO dependency preflight: fail early with a clear message when neuralop is missing.
                if needs_fno_dependency {
                    if let Err(e) = check_neuralop_available() {
                        errors.push(e.to_string());
                    }
                }

                let model_id_resolved = resolved_model_id.unwrap_or_default();

                // Resolve device: default to CUDA when available (auto), else CPU.
                // Accept UI values: "auto" | "cpu" | "cuda" | "cuda:0".
                let gpus = gpu_info
                    .get("gpus")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let default_device = if !gpus.is_empty() { "cuda:0" } else { "cpu" };

                let mut resolved_device = match device_req.as_deref() {
                    None => default_device.to_string(),
                    Some(d) if d.trim().is_empty() => default_device.to_string(),
                    Some(d) if d.trim().eq_ignore_ascii_case("auto") => default_device.to_string(),
                    Some(d) => d.trim().to_string(),
                };

                if resolved_device.eq_ignore_ascii_case("cuda") {
                    resolved_device = "cuda:0".to_string();
                }

                if resolved_device.starts_with("cuda") && gpus.is_empty() {
                    errors.push("CUDA device requested but no GPUs were detected".to_string());
                }

                // Apply defaults similar to Python bridge.
                let mut resolved_overlap = overlap_req.unwrap_or(0.25);
                if !(0.0..=0.99).contains(&resolved_overlap) {
                    // Accept guide-style integer overlaps (2/3/4/8...) used by MSST/UVR-style chunking.
                    if (2.0..=50.0).contains(&resolved_overlap) {
                        // Keep as-is; Python will normalize/interpret semantics per runtime.
                    }
                    // Special case: if UI sends percentage (e.g. 25 instead of 0.25)
                    else if (1.0..100.0).contains(&resolved_overlap) {
                        resolved_overlap /= 100.0;
                    } else {
                        warnings.push(format!(
                            "overlap out of expected range (ratio 0..1, divisor 2..50): {resolved_overlap}; using 0.25"
                        ));
                        resolved_overlap = 0.25;
                    }
                }

                let mut resolved_segment_size = segment_size_req.unwrap_or(352800);
                if resolved_segment_size <= 0 || resolved_segment_size == 256 {
                    // 0 = Auto, 256 = legacy/invalid. Use safe default.
                    resolved_segment_size = 352800;
                }

                // VRAM safety clamp (same breakpoints as Python bridge, best-effort).
                if resolved_device.starts_with("cuda") {
                    let idx = resolved_device
                        .split(':')
                        .nth(1)
                        .and_then(|s| s.parse::<i64>().ok())
                        .unwrap_or(0);
                    if let Some(gpus) = gpu_info.get("gpus").and_then(|v| v.as_array()) {
                        if let Some(gpu) = gpus
                            .iter()
                            .find(|g| g.get("index").and_then(|v| v.as_i64()) == Some(idx))
                        {
                            let vram_gb =
                                gpu.get("memory_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let max_safe = if vram_gb > 0.0 && vram_gb < 5.0 {
                                112455
                            } else if vram_gb > 0.0 && vram_gb < 8.0 {
                                352800
                            } else {
                                resolved_segment_size
                            };
                            if resolved_segment_size > max_safe {
                                warnings.push(format!("Reducing segment_size from {resolved_segment_size} to {max_safe} due to VRAM constraints ({vram_gb:.1} GB)"));
                                resolved_segment_size = max_safe;
                            }
                        }
                    }
                }

                // NOTE: We intentionally do not attempt ONNX introspection here.
                // Rust-native inference is not the current execution path; this preflight is a compatibility shim.

                let torch_available = check_torch_available();
                let can_proceed = errors.is_empty();
                let report = serde_json::json!({
                    "can_proceed": can_proceed,
                    "errors": errors,
                    "warnings": warnings,
                    "audio": audio_info,
                    "missing_models": missing_models,
                    "torch_available": torch_available,
                    "resolved": {
                        "model_id": if model_id_resolved.is_empty() { Value::Null } else { Value::String(model_id_resolved) },
                        "recipe": recipe_plan.unwrap_or(Value::Null),
                        "stems": stems_val,
                        "device": resolved_device,
                        "overlap": resolved_overlap,
                        "segment_size": resolved_segment_size,
                        "batch_size": batch_size_req.map(Value::from).unwrap_or(Value::Null),
                        "tta": tta_req,
                        "shifts": shifts_req.map(Value::from).unwrap_or(Value::Null),
                        "bitrate": bitrate_req.map(Value::from).unwrap_or(Value::Null),
                        "output_format": output_format_req,
                        "ensemble_algorithm": ensemble_algorithm,
                        "phase_fix_enabled": phase_fix_enabled,
                        "phase_fix_params": phase_fix_params
                    },
                    "memory": Value::Null
                });

                send_ok(&stdout, req.id, report)?;
            }
            "separate_audio" => {
                let job_id = Uuid::new_v4().to_string();

                // If this is a recipe, emit a lightweight plan event for UI/debugging.
                if let Some(model_id) = req.extra.get("model_id").and_then(|v| v.as_str()) {
                    let recipes_json =
                        read_json_file(&cfg.assets_dir.join("recipes.json")).unwrap_or(Value::Null);
                    if let Some(recipe) = recipes_json
                        .get("recipes")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| {
                            arr.iter()
                                .find(|r| r.get("id").and_then(|v| v.as_str()) == Some(model_id))
                        })
                    {
                        let mut plan = Map::new();
                        plan.insert("id".to_string(), Value::String(model_id.to_string()));
                        if let Some(t) = recipe.get("type").and_then(|v| v.as_str()) {
                            plan.insert("type".to_string(), Value::String(t.to_string()));
                        }
                        if let Some(defaults) = recipe.get("defaults") {
                            plan.insert("defaults".to_string(), defaults.clone());
                        }
                        if let Some(pp) = recipe.get("post_processing") {
                            plan.insert("post_processing".to_string(), pp.clone());
                        }
                        if let Some(steps) = recipe.get("steps").and_then(|v| v.as_array()) {
                            let mut steps_out: Vec<Value> = Vec::new();
                            for step in steps {
                                let step_name = step
                                    .get("step_name")
                                    .and_then(|v| v.as_str())
                                    .or_else(|| step.get("name").and_then(|v| v.as_str()))
                                    .unwrap_or("step");
                                let input_source = step
                                    .get("input_source")
                                    .and_then(|v| v.as_str())
                                    .or_else(|| step.get("input_from").and_then(|v| v.as_str()));
                                let output = step.get("output").cloned().unwrap_or(Value::Null);

                                if let Some(mid) = step.get("model_id").and_then(|v| v.as_str()) {
                                    let mut obj = Map::new();
                                    obj.insert(
                                        "step_name".to_string(),
                                        Value::String(step_name.to_string()),
                                    );
                                    obj.insert(
                                        "model_id".to_string(),
                                        Value::String(mid.to_string()),
                                    );
                                    if let Some(src) = input_source {
                                        obj.insert(
                                            "input_source".to_string(),
                                            Value::String(src.to_string()),
                                        );
                                    }
                                    if !output.is_null() {
                                        obj.insert("output".to_string(), output);
                                    }
                                    steps_out.push(Value::Object(obj));
                                }
                            }
                            if !steps_out.is_empty() {
                                plan.insert("steps".to_string(), Value::Array(steps_out));
                            }
                        }

                        send_event(
                            &stdout,
                            serde_json::json!({
                                "type": "recipe_plan",
                                "job_id": job_id,
                                "recipe": Value::Object(plan)
                            }),
                        );
                    }
                }

                // Prepare config object for inference script
                // We forward the entire `extra` map, plus ensure models_dir is set
                let mut config_obj = Value::Object(req.extra.clone());
                if let Some(obj) = config_obj.as_object_mut() {
                    obj.insert(
                        "models_dir".to_string(),
                        Value::String(cfg.models_dir.to_string_lossy().to_string()),
                    );
                    obj.insert("job_id".to_string(), Value::String(job_id.clone()));

                    // Resolve device server-side when the client does not specify one.
                    // This makes "auto" actually use CUDA when available.
                    let gpus = gpu_info
                        .get("gpus")
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let default_device = if !gpus.is_empty() { "cuda:0" } else { "cpu" };

                    let incoming = obj
                        .get("device")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let mut resolved =
                        if incoming.is_empty() || incoming.eq_ignore_ascii_case("auto") {
                            default_device.to_string()
                        } else {
                            incoming
                        };

                    if resolved.eq_ignore_ascii_case("cuda") {
                        resolved = "cuda:0".to_string();
                    }

                    obj.insert("device".to_string(), Value::String(resolved));
                }

                // Start background job
                match separation_manager.start(job_id.clone(), config_obj, &cfg, stdout.clone()) {
                    Ok(_) => {
                        send_ok(
                            &stdout,
                            req.id,
                            serde_json::json!({
                                "job_id": job_id,
                                "status": "started"
                            }),
                        )?;
                    }
                    Err(e) => {
                        send_err(
                            &stdout,
                            req.id,
                            "START_FAILED",
                            &format!("Failed to start separation: {e:#}"),
                        )?;
                    }
                }
            }
            "cancel_job" => {
                let job_id = req.extra.get("job_id").and_then(|v| v.as_str());
                if let Some(id) = job_id {
                    let cancelled = separation_manager.cancel(id);
                    if cancelled {
                        send_ok(
                            &stdout,
                            req.id,
                            serde_json::json!({
                                "job_id": id,
                                "status": "cancelled"
                            }),
                        )?;
                    } else {
                        // Attempt to forward to python proxy if not found in Rust manager?
                        // Or just report not found.
                        // Since we are taking over, we assume all valid jobs are in our manager.
                        send_err(&stdout, req.id, "NOT_FOUND", "Job not found or not running")?;
                    }
                } else {
                    send_err(&stdout, req.id, "INVALID", "job_id is required")?;
                }
            }
            "get_workflows" => {
                // Minimal parity: keep shape similar to Python ({workflows: [...]})
                send_ok(
                    &stdout,
                    req.id,
                    serde_json::json!({"workflows": ["live", "studio"]}),
                )?;
            }
            "check-preset-models" => {
                let preset_mappings = req
                    .extra
                    .get("preset_mappings")
                    .and_then(|v| v.as_object())
                    .cloned()
                    .unwrap_or_default();

                let mut availability = Map::new();
                for (preset_name, model_id_val) in preset_mappings {
                    let installed = model_id_val
                        .as_str()
                        .map(|id| any_model_file_exists(&cfg.models_dir, id, &Value::Null))
                        .unwrap_or(false);
                    availability.insert(preset_name, Value::from(installed));
                }
                send_ok(&stdout, req.id, Value::Object(availability))?;
            }
            "get_model_tech" => {
                let model_id = req.extra.get("model_id").and_then(|v| v.as_str());
                if model_id.is_none() {
                    send_err(&stdout, req.id, "INVALID", "model_id is required")?;
                    continue;
                }
                let model_id = model_id.unwrap();
                let models_json = read_json_file(&cfg.assets_dir.join("models.json.bak"))?;
                let models = models_json
                    .get("models")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let found = models
                    .into_iter()
                    .find(|m| m.get("id").and_then(|v| v.as_str()) == Some(model_id));
                if let Some(m) = found {
                    send_ok(
                        &stdout,
                        req.id,
                        serde_json::json!({
                            "ok": true,
                            "model_id": model_id,
                            "name": m.get("name").cloned().unwrap_or(Value::Null),
                            "architecture": m.get("architecture").cloned().unwrap_or(Value::Null),
                            "vram_required": m.get("vram_required").cloned().unwrap_or(Value::Null),
                            "sdr": m.get("sdr").cloned().unwrap_or(Value::Null),
                            "fullness": m.get("fullness").cloned().unwrap_or(Value::Null),
                            "bleedless": m.get("bleedless").cloned().unwrap_or(Value::Null),
                            "links": m.get("links").cloned().unwrap_or(Value::Null)
                        }),
                    )?;
                } else {
                    send_err(
                        &stdout,
                        req.id,
                        "NOT_FOUND",
                        &format!("Unknown model_id: {model_id}"),
                    )?;
                }
            }
            other => {
                send_err(
                    &stdout,
                    req.id,
                    "UNKNOWN_COMMAND",
                    &format!("Unknown command: {other}"),
                )?;
            }
        }
    }

    Ok(())
}
