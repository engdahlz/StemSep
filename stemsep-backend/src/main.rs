use anyhow::{anyhow, Context, Result};
use base64::Engine;
use chrono::Utc;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use sha2::{Digest, Sha256};
use uuid::Uuid;

use sysinfo::System;

use reqwest;
use reqwest::header;

#[cfg(target_os = "windows")]
use wasapi::{
    initialize_mta, DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat,
};

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
    catalog_cache_dir: PathBuf,
    catalog_bootstrap_path: PathBuf,
    catalog_remote_url: String,
    catalog_signature_url: String,
    catalog_public_key_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct CatalogStatus {
    active_revision: String,
    source_url: String,
    fetched_at: Option<String>,
    fallback_kind: String,
    stale: bool,
    signature_valid: bool,
    active_path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PlaybackDeviceInfo {
    id: String,
    label: String,
    kind: String,
    is_default: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PlaybackDeviceActivityProbe {
    device_id: String,
    detected: bool,
    peak_rms: f64,
    elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct PlaybackCaptureResultData {
    capture_id: String,
    file_path: String,
    capture_sample_rate: u32,
    capture_channels: u16,
    capture_bits_per_sample: u16,
    capture_sample_format: String,
    capture_start_at: String,
    capture_end_at: String,
    duration_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PlaybackCaptureJobState {
    capture_id: String,
    device_id: String,
    output_path: String,
    status: String,
    requested_at: String,
    started_at: Option<String>,
    finished_at: Option<String>,
    error: Option<String>,
}

#[derive(Debug)]
struct PlaybackCaptureJob {
    cancel: Arc<AtomicBool>,
    state: Arc<Mutex<PlaybackCaptureJobState>>,
}

static PLAYBACK_CAPTURE_JOBS: OnceLock<Mutex<HashMap<String, PlaybackCaptureJob>>> = OnceLock::new();

fn playback_capture_jobs() -> &'static Mutex<HashMap<String, PlaybackCaptureJob>> {
    PLAYBACK_CAPTURE_JOBS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_playback_capture_job(
    capture_id: &str,
    device_id: &str,
    output_path: &Path,
) -> Arc<AtomicBool> {
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mut registry = playback_capture_jobs().lock().expect("capture job registry");
    registry.insert(
        capture_id.to_string(),
        PlaybackCaptureJob {
            cancel: cancel_flag.clone(),
            state: Arc::new(Mutex::new(PlaybackCaptureJobState {
                capture_id: capture_id.to_string(),
                device_id: device_id.to_string(),
                output_path: output_path.display().to_string(),
                status: "queued".to_string(),
                requested_at: Utc::now().to_rfc3339(),
                started_at: None,
                finished_at: None,
                error: None,
            })),
        },
    );
    cancel_flag
}

fn set_playback_capture_job_status(capture_id: &str, status: &str, error: Option<String>) {
    let registry = playback_capture_jobs().lock().expect("capture job registry");
    if let Some(job) = registry.get(capture_id) {
        if let Ok(mut state) = job.state.lock() {
            state.status = status.to_string();
            if state.started_at.is_none()
                && matches!(status, "awaiting_audio" | "capturing" | "saving")
            {
                state.started_at = Some(Utc::now().to_rfc3339());
            }
            if matches!(status, "completed" | "cancelled" | "failed") {
                state.finished_at = Some(Utc::now().to_rfc3339());
            }
            state.error = error;
        }
    }
}

fn snapshot_playback_capture_jobs(capture_id: Option<&str>) -> Value {
    let registry = playback_capture_jobs().lock().expect("capture job registry");
    match capture_id {
        Some(id) => registry
            .get(id)
            .and_then(|job| job.state.lock().ok().map(|state| state.clone()))
            .map(|state| serde_json::to_value(state).unwrap_or(Value::Null))
            .unwrap_or(Value::Null),
        None => serde_json::to_value(
            registry
                .values()
                .filter_map(|job| job.state.lock().ok().map(|state| state.clone()))
                .collect::<Vec<_>>(),
        )
        .unwrap_or(Value::Array(vec![])),
    }
}

fn finish_playback_capture_job(capture_id: &str) {
    let mut registry = playback_capture_jobs().lock().expect("capture job registry");
    registry.remove(capture_id);
}

fn cancel_playback_capture_request(capture_id: Option<&str>) -> bool {
    let registry = playback_capture_jobs().lock().expect("capture job registry");
    match capture_id {
        Some(id) => registry
            .get(id)
            .map(|job| {
                job.cancel.store(true, Ordering::SeqCst);
                true
            })
            .unwrap_or(false),
        None => {
            for job in registry.values() {
                job.cancel.store(true, Ordering::SeqCst);
            }
            !registry.is_empty()
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ResolvedSourceLink {
    role: String,
    url: String,
    host: String,
    provider: String,
    resolver: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    locator: Option<Value>,
    manual: bool,
    channel: String,
    priority: usize,
    auth: String,
    verified: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ResolvedAvailability {
    #[serde(rename = "class")]
    class_name: String,
    reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ResolvedArtifactSource {
    url: String,
    host: String,
    provider: String,
    resolver: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    locator: Option<Value>,
    channel: String,
    priority: usize,
    auth: String,
    verified: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ResolvedArtifact {
    kind: String,
    filename: String,
    relative_path: String,
    required: bool,
    manual: bool,
    exists: bool,
    canonical_present: bool,
    legacy_present: bool,
    resolved_path: Option<String>,
    source: Option<String>,
    source_host: Option<String>,
    sha256: Option<String>,
    size_bytes: Option<u64>,
    sources: Vec<ResolvedArtifactSource>,
    verified: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ResolvedArtifactInstallation {
    kind: String,
    filename: String,
    relative_path: String,
    present: bool,
    verified: bool,
    canonical_present: bool,
    legacy_present: bool,
    resolved_path: Option<String>,
    required: bool,
    selected_source: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ResolvedInstallationStatus {
    installed: bool,
    canonical_ready: bool,
    legacy_fallback_used: bool,
    verified_hashes: bool,
    missing_artifacts: Vec<String>,
    relative_paths: Vec<String>,
    source_resolution: Vec<Value>,
    artifacts: Vec<ResolvedArtifactInstallation>,
}

#[derive(Debug, Clone, Serialize)]
struct ResolvedDownloadManifest {
    model_id: String,
    family: String,
    mode: String,
    strategy: String,
    install_mode: String,
    artifact_count: usize,
    downloadable_artifact_count: usize,
    source_policy: String,
    availability: ResolvedAvailability,
    sources: Vec<ResolvedSourceLink>,
    artifacts: Vec<ResolvedArtifact>,
    manual_instructions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartMetadata {
    model_id: String,
    filename: String,
    relative_path: String,
    source_url: String,
    sha256: Option<String>,
    size_bytes: Option<u64>,
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

const DEFAULT_REMOTE_CATALOG_URL: &str =
    "https://raw.githubusercontent.com/engdahlz/StemSep-catalog/main/catalog.runtime.remote.json";
const DEFAULT_REMOTE_CATALOG_SIG_URL: &str =
    "https://raw.githubusercontent.com/engdahlz/StemSep-catalog/main/catalog.runtime.remote.json.sig";

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

fn locate_catalog_cache_dir(cli_cache_dir: Option<String>) -> Result<PathBuf> {
    if let Some(path) = cli_cache_dir {
        return Ok(PathBuf::from(path));
    }
    if let Ok(env) = std::env::var("STEMSEP_CATALOG_CACHE_DIR") {
        return Ok(PathBuf::from(env));
    }
    let home = home_dir().ok_or_else(|| anyhow!("Unable to resolve home directory"))?;
    Ok(home.join(".stemsep").join("catalog"))
}

fn locate_catalog_bootstrap_path(assets_dir: &Path) -> PathBuf {
    let dedicated = assets_dir.join("catalog.runtime.bootstrap.json");
    if dedicated.exists() {
        return dedicated;
    }
    assets_dir.join("catalog.runtime.json")
}

fn locate_catalog_public_key_path(assets_dir: &Path, cli_value: Option<String>) -> PathBuf {
    if let Some(value) = cli_value {
        return PathBuf::from(value);
    }
    if let Ok(env) = std::env::var("STEMSEP_CATALOG_PUBLIC_KEY_PATH") {
        return PathBuf::from(env);
    }
    assets_dir
        .join("registry")
        .join("remote")
        .join("catalog.public.ed25519.txt")
}

fn configured_catalog_remote_url(cli_value: Option<String>) -> String {
    cli_value
        .or_else(|| std::env::var("STEMSEP_CATALOG_URL").ok())
        .unwrap_or_else(|| DEFAULT_REMOTE_CATALOG_URL.to_string())
}

fn configured_catalog_signature_url(cli_value: Option<String>) -> String {
    cli_value
        .or_else(|| std::env::var("STEMSEP_CATALOG_SIG_URL").ok())
        .unwrap_or_else(|| DEFAULT_REMOTE_CATALOG_SIG_URL.to_string())
}

fn ephemeral_backend_config(models_dir: &Path) -> BackendConfig {
    BackendConfig {
        assets_dir: PathBuf::new(),
        models_dir: models_dir.to_path_buf(),
        catalog_cache_dir: PathBuf::new(),
        catalog_bootstrap_path: PathBuf::new(),
        catalog_remote_url: String::new(),
        catalog_signature_url: String::new(),
        catalog_public_key_path: PathBuf::new(),
    }
}

fn read_json_file(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn catalog_cached_runtime_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("catalog.runtime.cached.json")
}

fn catalog_cached_signature_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("catalog.runtime.cached.json.sig")
}

fn catalog_status_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("catalog.status.json")
}

fn validate_runtime_catalog_bytes(bytes: &[u8], label: &str) -> Result<Value> {
    let value: Value = serde_json::from_slice(bytes).with_context(|| format!("parse {label}"))?;
    let schema_version = value
        .get("schema_version")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let has_models = value.get("models").and_then(|v| v.as_array()).is_some();
    let has_index = value
        .get("selection_index")
        .and_then(|v| v.as_array())
        .is_some();
    if !has_models || !has_index {
        return Err(anyhow!(
            "{label} is not a compatible StemSep runtime catalog (missing models[] or selection_index[])"
        ));
    }
    if !schema_version.starts_with("catalog-runtime-v") {
        return Err(anyhow!(
            "{label} is not a compatible StemSep runtime catalog (schema_version={schema_version})"
        ));
    }
    Ok(value)
}

fn read_catalog_status(cache_dir: &Path) -> Option<CatalogStatus> {
    let path = catalog_status_path(cache_dir);
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn write_catalog_status(cache_dir: &Path, status: &CatalogStatus) -> Result<()> {
    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("create {}", cache_dir.display()))?;
    let path = catalog_status_path(cache_dir);
    let payload = serde_json::to_vec_pretty(status)?;
    std::fs::write(&path, payload).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn read_public_key(cfg: &BackendConfig) -> Result<VerifyingKey> {
    let text = std::fs::read_to_string(&cfg.catalog_public_key_path)
        .with_context(|| format!("read {}", cfg.catalog_public_key_path.display()))?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(anyhow!(
            "catalog public key file is empty: {}",
            cfg.catalog_public_key_path.display()
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(trimmed)
        .with_context(|| format!("decode {}", cfg.catalog_public_key_path.display()))?;
    let key_bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow!("catalog public key must be 32 raw bytes (base64-encoded)"))?;
    VerifyingKey::from_bytes(&key_bytes).context("parse ed25519 public key")
}

fn verify_catalog_signature(
    cfg: &BackendConfig,
    payload: &[u8],
    signature_text: &str,
) -> Result<()> {
    let verifying_key = read_public_key(cfg)?;
    let trimmed = signature_text.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("catalog signature payload is empty"));
    }
    let signature_bytes = base64::engine::general_purpose::STANDARD
        .decode(trimmed)
        .context("decode catalog signature")?;
    let signature = Signature::from_slice(&signature_bytes)
        .map_err(|_| anyhow!("catalog signature is not a valid ed25519 signature"))?;
    verifying_key
        .verify(payload, &signature)
        .context("verify catalog signature")
}

fn write_cached_catalog(cfg: &BackendConfig, payload: &[u8], signature_text: &str) -> Result<()> {
    std::fs::create_dir_all(&cfg.catalog_cache_dir)
        .with_context(|| format!("create {}", cfg.catalog_cache_dir.display()))?;
    let runtime_path = catalog_cached_runtime_path(&cfg.catalog_cache_dir);
    let signature_path = catalog_cached_signature_path(&cfg.catalog_cache_dir);
    std::fs::write(&runtime_path, payload)
        .with_context(|| format!("write {}", runtime_path.display()))?;
    std::fs::write(&signature_path, signature_text)
        .with_context(|| format!("write {}", signature_path.display()))?;
    Ok(())
}

fn refresh_remote_catalog(cfg: &BackendConfig) -> Result<CatalogStatus> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("StemSep/remote-catalog-v4")
        .build()
        .context("build catalog HTTP client")?;

    let response = client
        .get(&cfg.catalog_remote_url)
        .send()
        .with_context(|| format!("fetch {}", cfg.catalog_remote_url))?
        .error_for_status()
        .with_context(|| format!("download {}", cfg.catalog_remote_url))?;
    let payload = response.bytes().context("read remote catalog bytes")?;
    validate_runtime_catalog_bytes(payload.as_ref(), &cfg.catalog_remote_url)?;

    let signature_text = client
        .get(&cfg.catalog_signature_url)
        .send()
        .with_context(|| format!("fetch {}", cfg.catalog_signature_url))?
        .error_for_status()
        .with_context(|| format!("download {}", cfg.catalog_signature_url))?
        .text()
        .context("read remote catalog signature")?;
    verify_catalog_signature(cfg, payload.as_ref(), &signature_text)?;

    write_cached_catalog(cfg, payload.as_ref(), &signature_text)?;

    let status = CatalogStatus {
        active_revision: sha256_hex_bytes(payload.as_ref()),
        source_url: cfg.catalog_remote_url.clone(),
        fetched_at: Some(Utc::now().to_rfc3339()),
        fallback_kind: "remote_current".to_string(),
        stale: false,
        signature_valid: true,
        active_path: catalog_cached_runtime_path(&cfg.catalog_cache_dir)
            .display()
            .to_string(),
    };
    write_catalog_status(&cfg.catalog_cache_dir, &status)?;
    Ok(status)
}

fn cached_catalog_status(cfg: &BackendConfig) -> Result<Option<CatalogStatus>> {
    let runtime_path = catalog_cached_runtime_path(&cfg.catalog_cache_dir);
    let signature_path = catalog_cached_signature_path(&cfg.catalog_cache_dir);
    if !runtime_path.exists() || !signature_path.exists() {
        return Ok(None);
    }
    let payload = std::fs::read(&runtime_path)
        .with_context(|| format!("read {}", runtime_path.display()))?;
    validate_runtime_catalog_bytes(&payload, &runtime_path.display().to_string())?;
    let signature_text = std::fs::read_to_string(&signature_path)
        .with_context(|| format!("read {}", signature_path.display()))?;
    verify_catalog_signature(cfg, &payload, &signature_text)?;
    let fetched_at = read_catalog_status(&cfg.catalog_cache_dir).and_then(|status| status.fetched_at);
    Ok(Some(CatalogStatus {
        active_revision: sha256_hex_bytes(&payload),
        source_url: cfg.catalog_remote_url.clone(),
        fetched_at,
        fallback_kind: "cached_fallback".to_string(),
        stale: true,
        signature_valid: true,
        active_path: runtime_path.display().to_string(),
    }))
}

fn bundled_catalog_status(cfg: &BackendConfig) -> Result<CatalogStatus> {
    let payload = std::fs::read(&cfg.catalog_bootstrap_path)
        .with_context(|| format!("read {}", cfg.catalog_bootstrap_path.display()))?;
    validate_runtime_catalog_bytes(&payload, &cfg.catalog_bootstrap_path.display().to_string())?;
    Ok(CatalogStatus {
        active_revision: sha256_hex_bytes(&payload),
        source_url: cfg.catalog_bootstrap_path.display().to_string(),
        fetched_at: None,
        fallback_kind: "bundled_fallback".to_string(),
        stale: true,
        signature_valid: false,
        active_path: cfg.catalog_bootstrap_path.display().to_string(),
    })
}

fn active_catalog_status(cfg: &BackendConfig) -> Option<CatalogStatus> {
    let status = read_catalog_status(&cfg.catalog_cache_dir)?;
    let active_path = PathBuf::from(&status.active_path);
    if active_path.exists() {
        Some(status)
    } else {
        None
    }
}

fn refresh_or_fallback_catalog_status(cfg: &BackendConfig) -> Result<CatalogStatus> {
    match refresh_remote_catalog(cfg) {
        Ok(status) => Ok(status),
        Err(remote_error) => {
            if let Some(status) = cached_catalog_status(cfg)? {
                write_catalog_status(&cfg.catalog_cache_dir, &status)?;
                Ok(status)
            } else {
                let status = bundled_catalog_status(cfg).with_context(|| {
                    format!(
                        "remote catalog refresh failed and no usable cache exists: {remote_error:#}"
                    )
                })?;
                write_catalog_status(&cfg.catalog_cache_dir, &status)?;
                Ok(status)
            }
        }
    }
}

fn ensure_catalog_runtime_status(cfg: &BackendConfig) -> Result<CatalogStatus> {
    if let Some(status) = active_catalog_status(cfg) {
        return Ok(status);
    }
    if let Some(status) = cached_catalog_status(cfg)? {
        write_catalog_status(&cfg.catalog_cache_dir, &status)?;
        return Ok(status);
    }
    let status = bundled_catalog_status(cfg)?;
    write_catalog_status(&cfg.catalog_cache_dir, &status)?;
    Ok(status)
}

fn load_models_with_guide_overrides(cfg: &BackendConfig) -> Result<Value> {
    load_catalog_runtime(cfg)
}

fn load_catalog_runtime(cfg: &BackendConfig) -> Result<Value> {
    let status = ensure_catalog_runtime_status(cfg)?;
    let runtime_path = PathBuf::from(&status.active_path);
    if !runtime_path.exists() {
        return Err(anyhow!(
            "runtime catalog is missing after fallback resolution ({})",
            runtime_path.display()
        ));
    }
    let payload = std::fs::read(&runtime_path)
        .with_context(|| format!("read {}", runtime_path.display()))?;
    validate_runtime_catalog_bytes(&payload, &runtime_path.display().to_string())?;
    serde_json::from_slice(&payload).with_context(|| format!("parse {}", runtime_path.display()))
}

fn load_catalog_recipes(cfg: &BackendConfig) -> Result<Value> {
    let runtime = load_catalog_runtime(cfg)?;
    if let Some(recipes) = runtime.get("recipes").and_then(|v| v.as_array()) {
        return Ok(serde_json::json!({ "recipes": recipes.clone() }));
    }
    Err(anyhow!("catalog.runtime.json does not contain recipes[]"))
}

fn load_catalog_workflows(cfg: &BackendConfig) -> Result<Value> {
    let runtime = load_catalog_runtime(cfg)?;
    if let Some(workflows) = runtime.get("workflows").and_then(|v| v.as_array()) {
        return Ok(serde_json::json!({ "workflows": workflows.clone() }));
    }
    Err(anyhow!("catalog.runtime.json does not contain workflows[]"))
}

fn normalize_selection_type(value: &str) -> Option<String> {
    let normalized = value.trim().to_lowercase();
    match normalized.as_str() {
        "model" | "recipe" | "workflow" => Some(normalized),
        _ => None,
    }
}

fn selection_descriptor_from_extra(extra: &Map<String, Value>) -> (Option<String>, Option<String>) {
    let envelope = extra
        .get("selection_envelope")
        .or_else(|| extra.get("selectionEnvelope"))
        .and_then(|v| v.as_object());
    let selection_type = extra
        .get("selection_type")
        .or_else(|| extra.get("selectionType"))
        .and_then(|v| v.as_str())
        .or_else(|| envelope.and_then(|v| v.get("selectionType").and_then(|v| v.as_str())))
        .and_then(normalize_selection_type);
    let selection_id = extra
        .get("selection_id")
        .or_else(|| extra.get("selectionId"))
        .and_then(|v| v.as_str())
        .or_else(|| envelope.and_then(|v| v.get("selectionId").and_then(|v| v.as_str())))
        .or_else(|| extra.get("model_id").and_then(|v| v.as_str()))
        .or_else(|| extra.get("recipe_id").and_then(|v| v.as_str()))
        .or_else(|| extra.get("workflow_id").and_then(|v| v.as_str()))
        .map(|s| s.to_string());
    (selection_type, selection_id)
}

fn selection_envelope_from_entry(entry: &Value, selection_type: &str, selection_id: &str) -> Value {
    entry
        .get("selection_envelope")
        .cloned()
        .unwrap_or_else(|| {
            serde_json::json!({
                "selectionType": selection_type,
                "selectionId": selection_id,
                "catalogTier": entry.get("catalog_tier").cloned().unwrap_or(Value::Null),
                "sourceKind": entry.get("source_kind").cloned().unwrap_or(Value::Null),
                "installPolicy": entry.get("install_policy").cloned().unwrap_or(Value::Null),
                "verification": entry.get("verification").cloned().unwrap_or(Value::Null)
            })
        })
}

fn catalog_selection_entry(
    runtime: &Value,
    selection_type: &str,
    selection_id: &str,
) -> Option<Value> {
    let key = match selection_type {
        "model" => "models",
        "recipe" => "recipes",
        "workflow" => "workflows",
        _ => return None,
    };

    runtime
        .get(key)
        .and_then(|v| v.as_array())
        .and_then(|entries| {
            entries
                .iter()
                .find(|entry| {
                    entry.get("selection_id").and_then(|v| v.as_str()) == Some(selection_id)
                        || entry.get("id").and_then(|v| v.as_str()) == Some(selection_id)
                        || entry.get("model_id").and_then(|v| v.as_str()) == Some(selection_id)
                })
        })
        .cloned()
}

fn catalog_selection_required_model_ids(selection_type: &str, selection: &Value) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    let mut push_id = |candidate: Option<&str>| {
        if let Some(id) = candidate {
            let trimmed = id.trim();
            if !trimmed.is_empty() && !ids.iter().any(|existing| existing == trimmed) {
                ids.push(trimmed.to_string());
            }
        }
    };

    if let Some(explicit) = selection.get("required_model_ids").and_then(|v| v.as_array()) {
        for value in explicit {
            push_id(value.as_str());
        }
    }

    match selection_type {
        "model" => {
            push_id(selection.get("model_id").and_then(|v| v.as_str()));
            push_id(selection.get("id").and_then(|v| v.as_str()));
        }
        "recipe" => {
            if ids.is_empty() {
                ids.extend(collect_recipe_model_ids(selection));
            }
        }
        "workflow" => {
            if ids.is_empty() {
                ids.extend(collect_workflow_model_ids(selection));
            }
        }
        _ => {}
    }

    ids
}

fn catalog_selection_bundle(
    cfg: &BackendConfig,
    runtime: &Value,
    selection_type: &str,
    selection_id: &str,
) -> Result<Value> {
    let selection = catalog_selection_entry(runtime, selection_type, selection_id)
        .ok_or_else(|| anyhow!("Unknown {selection_type}: {selection_id}"))?;

    let selection_envelope = selection_envelope_from_entry(&selection, selection_type, selection_id);
    let required_model_ids = catalog_selection_required_model_ids(selection_type, &selection);
    let mut required_models: Vec<Value> = Vec::new();
    let mut required_model_installations: Vec<Value> = Vec::new();
    let mut missing_model_ids: Vec<String> = Vec::new();
    let mut missing_artifacts: Vec<String> = Vec::new();
    let mut verified_model_ids: Vec<String> = Vec::new();

    for model_id in required_model_ids.iter() {
        if let Some(model_obj) = find_model_object(runtime, model_id) {
            let manifest = resolve_model_download_manifest(cfg, model_id, &model_obj);
            let installation = resolve_installation_status(cfg, &manifest);
            if installation.installed {
                verified_model_ids.push(model_id.clone());
            } else {
                missing_artifacts.extend(installation.missing_artifacts.clone());
            }
            required_models.push(serde_json::json!({
                "model_id": model_id,
                "model": model_obj,
                "download": manifest,
                "installation": installation,
            }));
            required_model_installations.push(serde_json::json!({
                "model_id": model_id,
                "installed": installation.installed,
                "missing_artifacts": installation.missing_artifacts,
                "relative_paths": installation.relative_paths,
                "artifacts": installation.artifacts,
            }));
        } else {
            missing_model_ids.push(model_id.clone());
            required_models.push(serde_json::json!({
                "model_id": model_id,
                "error": "NOT_FOUND"
            }));
            required_model_installations.push(serde_json::json!({
                "model_id": model_id,
                "installed": false,
                "error": "NOT_FOUND"
            }));
        }
    }

    let installed = missing_model_ids.is_empty()
        && required_models
            .iter()
            .all(|entry| entry.get("installation").and_then(|v| v.get("installed")).and_then(|v| v.as_bool()) == Some(true));
    let canonical_ready = missing_model_ids.is_empty()
        && required_models.iter().all(|entry| {
            entry.get("installation")
                .and_then(|v| v.get("canonical_ready"))
                .and_then(|v| v.as_bool())
                == Some(true)
        });
    let legacy_fallback_used = required_models.iter().any(|entry| {
        entry.get("installation")
            .and_then(|v| v.get("legacy_fallback_used"))
            .and_then(|v| v.as_bool())
            == Some(true)
    });

    Ok(serde_json::json!({
        "selection_type": selection_type,
        "selection_id": selection_id,
        "selection": selection,
        "selection_envelope": selection_envelope,
        "catalog_tier": selection.get("catalog_tier").cloned().unwrap_or(Value::Null),
        "source_kind": selection.get("source_kind").cloned().unwrap_or(Value::Null),
        "install_policy": selection.get("install_policy").cloned().unwrap_or(Value::Null),
        "runtime_adapter": selection.get("runtime_adapter").cloned().unwrap_or(Value::Null),
        "verification": selection.get("verification").cloned().unwrap_or(Value::Null),
        "required_model_ids": required_model_ids,
        "required_models": required_models,
        "required_model_installations": required_model_installations,
        "missing_model_ids": missing_model_ids,
        "missing_artifacts": missing_artifacts,
        "verified_model_ids": verified_model_ids,
        "installed": installed,
        "canonical_ready": canonical_ready,
        "legacy_fallback_used": legacy_fallback_used,
    }))
}

fn catalog_selection_installation(cfg: &BackendConfig, runtime: &Value, selection_type: &str, selection_id: &str) -> Result<Value> {
    let bundle = catalog_selection_bundle(cfg, runtime, selection_type, selection_id)?;
    let required_model_installations = bundle
        .get("required_model_installations")
        .cloned()
        .unwrap_or(Value::Array(vec![]));
    let installed = bundle
        .get("installed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok(serde_json::json!({
        "selection_type": bundle.get("selection_type").cloned().unwrap_or(Value::Null),
        "selection_id": bundle.get("selection_id").cloned().unwrap_or(Value::Null),
        "selection_envelope": bundle.get("selection_envelope").cloned().unwrap_or(Value::Null),
        "catalog_tier": bundle.get("catalog_tier").cloned().unwrap_or(Value::Null),
        "source_kind": bundle.get("source_kind").cloned().unwrap_or(Value::Null),
        "install_policy": bundle.get("install_policy").cloned().unwrap_or(Value::Null),
        "runtime_adapter": bundle.get("runtime_adapter").cloned().unwrap_or(Value::Null),
        "verification": bundle.get("verification").cloned().unwrap_or(Value::Null),
        "required_model_ids": bundle.get("required_model_ids").cloned().unwrap_or(Value::Null),
        "required_model_installations": required_model_installations,
        "missing_model_ids": bundle.get("missing_model_ids").cloned().unwrap_or(Value::Null),
        "missing_artifacts": bundle.get("missing_artifacts").cloned().unwrap_or(Value::Null),
        "installed": installed,
        "canonical_ready": bundle.get("canonical_ready").cloned().unwrap_or(Value::Bool(false)),
        "legacy_fallback_used": bundle.get("legacy_fallback_used").cloned().unwrap_or(Value::Bool(false)),
    }))
}

fn catalog_selection_install_plan(
    cfg: &BackendConfig,
    runtime: &Value,
    selection_type: &str,
    selection_id: &str,
) -> Result<Value> {
    let bundle = catalog_selection_bundle(cfg, runtime, selection_type, selection_id)?;
    Ok(serde_json::json!({
        "selection_type": bundle.get("selection_type").cloned().unwrap_or(Value::Null),
        "selection_id": bundle.get("selection_id").cloned().unwrap_or(Value::Null),
        "selection": bundle.get("selection").cloned().unwrap_or(Value::Null),
        "selection_envelope": bundle.get("selection_envelope").cloned().unwrap_or(Value::Null),
        "catalog_tier": bundle.get("catalog_tier").cloned().unwrap_or(Value::Null),
        "source_kind": bundle.get("source_kind").cloned().unwrap_or(Value::Null),
        "install_policy": bundle.get("install_policy").cloned().unwrap_or(Value::Null),
        "runtime_adapter": bundle.get("runtime_adapter").cloned().unwrap_or(Value::Null),
        "verification": bundle.get("verification").cloned().unwrap_or(Value::Null),
        "required_model_ids": bundle.get("required_model_ids").cloned().unwrap_or(Value::Null),
        "required_models": bundle.get("required_models").cloned().unwrap_or(Value::Null),
        "installations": bundle.get("required_model_installations").cloned().unwrap_or(Value::Null),
        "missing_model_ids": bundle.get("missing_model_ids").cloned().unwrap_or(Value::Null),
        "missing_artifacts": bundle.get("missing_artifacts").cloned().unwrap_or(Value::Null),
        "installed": bundle.get("installed").cloned().unwrap_or(Value::Bool(false)),
        "canonical_ready": bundle.get("canonical_ready").cloned().unwrap_or(Value::Bool(false)),
        "legacy_fallback_used": bundle.get("legacy_fallback_used").cloned().unwrap_or(Value::Bool(false)),
    }))
}

fn catalog_selection_verification(
    cfg: &BackendConfig,
    runtime: &Value,
    selection_type: &str,
    selection_id: &str,
) -> Result<Value> {
    let bundle = catalog_selection_bundle(cfg, runtime, selection_type, selection_id)?;
    let mut artifact_checks: Vec<Value> = Vec::new();
    for required_model in bundle
        .get("required_models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
    {
        let model_id = required_model
            .get("model_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let installation = required_model.get("installation").cloned().unwrap_or(Value::Null);
        let download = required_model.get("download").cloned().unwrap_or(Value::Null);
        let mut per_artifact: Vec<Value> = Vec::new();
        if let Some(manifest_artifacts) = download.get("artifacts").and_then(|v| v.as_array()) {
            for artifact in manifest_artifacts {
                per_artifact.push(serde_json::json!({
                    "kind": artifact.get("kind").cloned().unwrap_or(Value::Null),
                    "filename": artifact.get("filename").cloned().unwrap_or(Value::Null),
                    "relative_path": artifact.get("relative_path").cloned().unwrap_or(Value::Null),
                    "exists": artifact.get("exists").cloned().unwrap_or(Value::Null),
                    "verified": artifact.get("verified").cloned().unwrap_or(Value::Null),
                    "canonical_present": artifact.get("canonical_present").cloned().unwrap_or(Value::Null),
                    "legacy_present": artifact.get("legacy_present").cloned().unwrap_or(Value::Null),
                    "resolved_path": artifact.get("resolved_path").cloned().unwrap_or(Value::Null),
                    "manual": artifact.get("manual").cloned().unwrap_or(Value::Null),
                    "source": artifact.get("source").cloned().unwrap_or(Value::Null),
                }));
            }
        }
        artifact_checks.push(serde_json::json!({
            "model_id": model_id,
            "installed": installation.get("installed").cloned().unwrap_or(Value::Bool(false)),
            "canonical_ready": installation.get("canonical_ready").cloned().unwrap_or(Value::Bool(false)),
            "legacy_fallback_used": installation.get("legacy_fallback_used").cloned().unwrap_or(Value::Bool(false)),
            "missing_artifacts": installation.get("missing_artifacts").cloned().unwrap_or(Value::Null),
            "artifacts": per_artifact,
        }));
    }

    Ok(serde_json::json!({
        "selection_type": bundle.get("selection_type").cloned().unwrap_or(Value::Null),
        "selection_id": bundle.get("selection_id").cloned().unwrap_or(Value::Null),
        "selection_envelope": bundle.get("selection_envelope").cloned().unwrap_or(Value::Null),
        "required_model_ids": bundle.get("required_model_ids").cloned().unwrap_or(Value::Null),
        "artifact_checks": artifact_checks,
        "installed": bundle.get("installed").cloned().unwrap_or(Value::Bool(false)),
        "canonical_ready": bundle.get("canonical_ready").cloned().unwrap_or(Value::Bool(false)),
        "legacy_fallback_used": bundle.get("legacy_fallback_used").cloned().unwrap_or(Value::Bool(false)),
        "missing_model_ids": bundle.get("missing_model_ids").cloned().unwrap_or(Value::Null),
        "missing_artifacts": bundle.get("missing_artifacts").cloned().unwrap_or(Value::Null),
        "verification": bundle.get("verification").cloned().unwrap_or(Value::Null),
    }))
}

fn catalog_execution_plan(
    cfg: &BackendConfig,
    runtime: &Value,
    selection_type: &str,
    selection_id: &str,
) -> Result<Value> {
    let bundle = catalog_selection_bundle(cfg, runtime, selection_type, selection_id)?;
    let selection = bundle.get("selection").cloned().unwrap_or(Value::Null);
    let required_model_ids = bundle
        .get("required_model_ids")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let runtime_adapter = bundle.get("runtime_adapter").cloned().unwrap_or(Value::Null);

    Ok(serde_json::json!({
        "selection_type": bundle.get("selection_type").cloned().unwrap_or(Value::Null),
        "selection_id": bundle.get("selection_id").cloned().unwrap_or(Value::Null),
        "selection_envelope": bundle.get("selection_envelope").cloned().unwrap_or(Value::Null),
        "selection": selection,
        "required_model_ids": required_model_ids,
        "required_models": bundle.get("required_models").cloned().unwrap_or(Value::Null),
        "install_plan": catalog_selection_install_plan(cfg, runtime, selection_type, selection_id)?,
        "verification": bundle.get("verification").cloned().unwrap_or(Value::Null),
        "runtime_adapter": runtime_adapter,
        "execution_constraints": serde_json::json!({
            "manual_install": bundle.get("install_policy").and_then(|v| v.as_str()).map(|v| v != "direct").unwrap_or(false),
            "catalog_tier": bundle.get("catalog_tier").cloned().unwrap_or(Value::Null),
            "source_kind": bundle.get("source_kind").cloned().unwrap_or(Value::Null),
            "canonical_ready": bundle.get("canonical_ready").cloned().unwrap_or(Value::Bool(false)),
            "legacy_fallback_used": bundle.get("legacy_fallback_used").cloned().unwrap_or(Value::Bool(false)),
        }),
        "resolved_bundle": bundle,
    }))
}

fn detect_runtime_adapter(model_obj: &Value) -> Option<String> {
    let preferred = model_obj
        .get("runtime")
        .and_then(|v| v.get("preferred"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let allowed = model_obj
        .get("runtime")
        .and_then(|v| v.get("allowed"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(|entry| entry.to_string()))
        .collect::<Vec<_>>();
    let engine = model_obj
        .get("runtime")
        .and_then(|v| v.get("engine"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let variant = model_obj
        .get("runtime")
        .and_then(|v| v.get("variant"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    match engine.as_deref() {
        Some("msst_builtin") => Some("msst_builtin".to_string()),
        Some("demucs_native") => Some("demucs_native".to_string()),
        Some("custom_builtin_variant") => Some("custom_builtin_variant".to_string()),
        Some("native_stemsep") => Some("native_stemsep".to_string()),
        Some(other) => Some(other.to_string()),
        None => match variant.as_deref() {
            Some("demucs") => Some("demucs_native".to_string()),
            Some("fno") => Some("custom_builtin_variant".to_string()),
            Some(_) => Some("native_stemsep".to_string()),
            None => match preferred.as_deref() {
                Some("msst") => Some("msst_builtin".to_string()),
                Some("demucs") | Some("demucs_native") => Some("demucs_native".to_string()),
                Some("stemsep") | Some("native_stemsep") => Some("native_stemsep".to_string()),
                Some(other) => Some(other.to_string()),
                None => {
                    if allowed.iter().any(|entry| entry == "msst") {
                        Some("msst_builtin".to_string())
                    } else if allowed
                        .iter()
                        .any(|entry| matches!(entry.as_str(), "demucs" | "demucs_native"))
                    {
                        Some("demucs_native".to_string())
                    } else if allowed
                        .iter()
                        .any(|entry| matches!(entry.as_str(), "stemsep" | "native_stemsep"))
                    {
                        Some("native_stemsep".to_string())
                    } else {
                        None
                    }
                }
            },
        },
    }
}

fn collect_missing_runtime_assets(models_dir: &Path, model_id: &str, model_obj: &Value) -> Vec<String> {
    let cfg = ephemeral_backend_config(models_dir);
    let manifest = resolve_model_download_manifest(&cfg, model_id, model_obj);
    resolve_installation_status(&cfg, &manifest).missing_artifacts
}

fn collect_workflow_model_ids(workflow: &Value) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();

    let mut push_id = |candidate: Option<&str>| {
        if let Some(id) = candidate {
            let trimmed = id.trim();
            if !trimmed.is_empty() && !ids.iter().any(|existing| existing == trimmed) {
                ids.push(trimmed.to_string());
            }
        }
    };

    if let Some(models) = workflow.get("models").and_then(|v| v.as_array()) {
        for model in models {
            push_id(model.get("model_id").and_then(|v| v.as_str()));
        }
    }

    if let Some(steps) = workflow.get("steps").and_then(|v| v.as_array()) {
        for step in steps {
            push_id(step.get("model_id").and_then(|v| v.as_str()));
            push_id(step.get("source_model").and_then(|v| v.as_str()));
        }
    }

    ids
}

fn collect_recipe_model_ids(recipe: &Value) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();

    let mut push_id = |candidate: Option<&str>| {
        if let Some(id) = candidate {
            let trimmed = id.trim();
            if !trimmed.is_empty() && !ids.iter().any(|existing| existing == trimmed) {
                ids.push(trimmed.to_string());
            }
        }
    };

    if let Some(steps) = recipe.get("steps").and_then(|v| v.as_array()) {
        for step in steps {
            push_id(step.get("model_id").and_then(|v| v.as_str()));
            push_id(step.get("source_model").and_then(|v| v.as_str()));
        }
    }

    ids
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

fn url_host(url: &str) -> Option<String> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let after_scheme = trimmed
        .split("://")
        .nth(1)
        .unwrap_or(trimmed);
    let host = after_scheme
        .split('/')
        .next()
        .unwrap_or("")
        .trim()
        .trim_end_matches(':');
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

fn find_model_object(models_json: &Value, model_id: &str) -> Option<Value> {
    models_json
        .get("models")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|m| m.get("id").and_then(|v| v.as_str()) == Some(model_id))
        })
        .cloned()
}

fn find_recipe_object(recipes_json: &Value, recipe_id: &str) -> Option<Value> {
    recipes_json
        .get("recipes")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .find(|recipe| recipe.get("id").and_then(|v| v.as_str()) == Some(recipe_id))
        })
        .cloned()
}

fn has_complete_audio_quality_thresholds(value: Option<&Value>) -> bool {
    let Some(obj) = value.and_then(|v| v.as_object()) else {
        return false;
    };

    [
        "min_correlation",
        "min_snr_db",
        "min_si_sdr_db",
        "max_gain_delta_db",
        "max_clipped_samples",
    ]
    .iter()
    .all(|key| obj.contains_key(*key))
}

fn extract_recipe_id_from_manifest(manifest: &Value) -> Option<String> {
    manifest
        .get("recipe_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            manifest
                .get("config")
                .and_then(|v| v.get("recipe_id").or_else(|| v.get("recipeId")))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

fn extract_golden_set_id_from_manifest(manifest: &Value) -> Option<String> {
    manifest
        .get("golden_set_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            manifest
                .get("config")
                .and_then(|v| v.get("golden_set_id").or_else(|| v.get("goldenSetId")))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
}

fn model_family(model_obj: &Value) -> String {
    let engine = model_obj
        .get("runtime")
        .and_then(|v| v.get("engine"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    let model_type = model_obj
        .get("runtime")
        .and_then(|v| v.get("model_type"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    let architecture = model_obj
        .get("architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();

    if engine == "demucs_native" || architecture.contains("demucs") || model_type.contains("demucs") {
        return "demucs".to_string();
    }
    if architecture == "vr" || model_type == "vr" {
        return "vr".to_string();
    }
    if architecture.contains("mdx") || model_type.contains("mdx") {
        return "mdx".to_string();
    }
    if engine == "msst_builtin"
        || engine == "custom_builtin_variant"
        || architecture.contains("roformer")
        || architecture.contains("scnet")
        || architecture.contains("apollo")
        || architecture.contains("bandit")
        || model_type.contains("roformer")
        || model_type.contains("scnet")
        || model_type.contains("apollo")
        || model_type.contains("bandit")
    {
        return "msst".to_string();
    }
    "other".to_string()
}

fn artifact_relative_path(model_obj: &Value, model_id: &str, filename: &str) -> String {
    if let Some(download_artifacts) = model_obj
        .get("download")
        .and_then(|v| v.get("artifacts"))
        .and_then(|v| v.as_array())
    {
        for item in download_artifacts {
            let Some(item_filename) = item.get("filename").and_then(|v| v.as_str()) else {
                continue;
            };
            if item_filename != filename {
                continue;
            }
            if let Some(relative) = item.get("relative_path").and_then(|v| v.as_str()) {
                let trimmed = relative.trim();
                if !trimmed.is_empty() {
                    return trimmed.replace('\\', "/");
                }
            }
        }
    }

    format!("{}/{}/{}", model_family(model_obj), model_id, filename).replace('\\', "/")
}

fn is_stale_legacy_mirror_url(url: &str) -> bool {
    url.to_ascii_lowercase()
        .contains("github.com/engdahlz/stemsep-models/releases/download/models-v2-2026-01-01/")
}

fn legacy_artifact_paths(
    models_dir: &Path,
    model_id: &str,
    kind: &str,
    filename: &str,
) -> Vec<PathBuf> {
    let mut candidates = vec![models_dir.join(filename)];
    let ext = Path::new(filename)
        .extension()
        .and_then(|v| v.to_str())
        .map(|value| format!(".{value}"));
    if let Some(ext) = ext {
        candidates.push(models_dir.join(format!("{model_id}{ext}")));
    }
    if kind == "config" {
        candidates.push(models_dir.join(format!("{model_id}.yaml")));
        candidates.push(models_dir.join(format!("{model_id}.yml")));
    }
    candidates.sort();
    candidates.dedup();
    candidates
}

fn verify_local_artifact_hash(path: &Path, expected_sha256: &str) -> Result<bool> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open {} for sha256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 128];
    loop {
        let read = file.read(&mut buffer).context("read artifact for sha256")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    Ok(actual.eq_ignore_ascii_case(expected_sha256))
}

fn part_metadata_path(tmp_path: &Path) -> PathBuf {
    let file_name = tmp_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("download.part");
    tmp_path.with_file_name(format!("{file_name}.json"))
}

fn read_part_metadata(path: &Path) -> Option<PartMetadata> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<PartMetadata>(&text).ok()
}

fn write_part_metadata(path: &Path, metadata: &PartMetadata) -> Result<()> {
    let payload = serde_json::to_vec_pretty(metadata).context("serialize part metadata")?;
    std::fs::write(path, payload).with_context(|| format!("write {}", path.display()))
}

fn remove_part_metadata(path: &Path) {
    let _ = std::fs::remove_file(path);
}

fn resolve_model_download_manifest(
    cfg: &BackendConfig,
    model_id: &str,
    model_obj: &Value,
) -> ResolvedDownloadManifest {
    let links = model_obj.get("links").and_then(|v| v.as_object());
    let runtime = model_obj.get("runtime").and_then(|v| v.as_object());
    let install = model_obj.get("install").and_then(|v| v.as_object());
    let availability_obj = model_obj.get("availability").and_then(|v| v.as_object());
    let install_mode = install
        .and_then(|v| v.get("mode"))
        .and_then(|v| v.as_str())
        .or_else(|| runtime.and_then(|v| v.get("install_mode")).and_then(|v| v.as_str()))
        .or_else(|| {
            model_obj
                .get("download")
                .and_then(|v| v.get("install_mode"))
                .and_then(|v| v.as_str())
        })
        .unwrap_or("direct")
        .to_string();
    let availability = ResolvedAvailability {
        class_name: availability_obj
            .and_then(|v| v.get("class"))
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                if install_mode == "manual"
                    || runtime
                        .and_then(|v| v.get("requires_manual_assets"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                {
                    "manual_import"
                } else if model_obj.get("catalog_status").and_then(|v| v.as_str()) == Some("blocked")
                {
                    "blocked_non_public"
                } else {
                    "direct"
                }
            })
            .to_string(),
        reason: availability_obj
            .and_then(|v| v.get("reason"))
            .and_then(|v| v.as_str())
            .map(|v| v.to_string()),
    };
    let manual_registry = availability.class_name == "manual_import"
        || install_mode == "manual"
        || runtime
            .and_then(|v| v.get("requires_manual_assets"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

    let mut sources: Vec<ResolvedSourceLink> = Vec::new();
    let mut seen_source_urls: Vec<String> = Vec::new();
    let mut push_source = |role: &str, source_value: &Value, manual: bool| {
        let trimmed = if let Some(url) = source_value.as_str() {
            url.trim().to_string()
        } else {
            source_value
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string()
        };
        if trimmed.is_empty()
            || seen_source_urls
                .iter()
                .any(|existing| existing.as_str() == trimmed.as_str())
        {
            return;
        }
        seen_source_urls.push(trimmed.to_string());
        let (provider, resolver, locator) = infer_source_provider_resolver(&trimmed);
        sources.push(ResolvedSourceLink {
            role: role.to_string(),
            url: trimmed.to_string(),
            host: source_value
                .get("host")
                .and_then(|v| v.as_str())
                .map(|v| v.to_string())
                .or_else(|| url_host(&trimmed))
                .unwrap_or_else(|| "unknown".to_string()),
            provider: source_value
                .get("provider")
                .and_then(|v| v.as_str())
                .unwrap_or(&provider)
                .to_string(),
            resolver: source_value
                .get("resolver")
                .and_then(|v| v.as_str())
                .unwrap_or(&resolver)
                .to_string(),
            locator: source_value.get("locator").cloned().or(locator),
            manual,
            channel: source_value
                .get("channel")
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| {
                    if trimmed.contains("github.com/engdahlz/stemsep-models") {
                        "mirror"
                    } else {
                        "upstream"
                    }
                })
                .to_string(),
            priority: source_value
                .get("priority")
                .and_then(|v| v.as_u64())
                .unwrap_or((sources.len() + 1) as u64) as usize,
            auth: source_value
                .get("auth")
                .and_then(|v| v.as_str())
                .unwrap_or("none")
                .to_string(),
            verified: source_value
                .get("verified")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        });
    };

    if let Some(links_obj) = links {
        if let Some(url) = links_obj.get("checkpoint") {
            push_source("checkpoint", url, manual_registry);
        }
        if let Some(url) = links_obj.get("config") {
            push_source("config", url, manual_registry);
        }
        if let Some(url) = links_obj.get("homepage") {
            push_source("homepage", url, true);
        }
    }
    if let Some(download_sources) = model_obj
        .get("download")
        .and_then(|v| v.get("sources"))
        .and_then(|v| v.as_array())
    {
        for source in download_sources {
            let role = source
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("source");
            let manual = source
                .get("manual")
                .and_then(|v| v.as_bool())
                .unwrap_or(manual_registry);
            push_source(role, source, manual);
        }
    }

    fn push_artifact_entry(
        models_dir: &Path,
        model_obj: &Value,
        model_id: &str,
        artifacts: &mut Vec<ResolvedArtifact>,
        seen_paths: &mut Vec<String>,
        kind: &str,
        filename: &str,
        source: Option<String>,
        required: bool,
        manual: bool,
        sha256: Option<String>,
        size_bytes: Option<u64>,
        explicit_relative_path: Option<String>,
        artifact_sources: Vec<ResolvedArtifactSource>,
    ) {
        let trimmed = filename.trim();
        if trimmed.is_empty() || trimmed.contains(".MISSING") {
            return;
        }
        let relative_path = explicit_relative_path
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| artifact_relative_path(model_obj, model_id, trimmed));
        if seen_paths.iter().any(|existing| existing == &relative_path) {
            return;
        }
        seen_paths.push(relative_path.clone());
        let source_host = source.as_deref().and_then(url_host);
        let stale_legacy_mirror = source
            .as_deref()
            .map(is_stale_legacy_mirror_url)
            .unwrap_or(false);
        let effective_sources = artifact_sources;
        let canonical_path = models_dir.join(relative_path.replace('/', "\\"));
        let probe = ResolvedArtifact {
            kind: kind.to_string(),
            filename: trimmed.to_string(),
            relative_path: relative_path.clone(),
            required,
            manual: manual || stale_legacy_mirror,
            exists: false,
            canonical_present: false,
            legacy_present: false,
            resolved_path: None,
            source: source.clone(),
            source_host: source_host.clone(),
            sha256: sha256.clone(),
            size_bytes,
            sources: effective_sources.clone(),
            verified: false,
        };
        let canonical_exists = canonical_path.exists();
        let legacy_path = legacy_artifact_paths(models_dir, model_id, &probe.kind, &probe.filename)
            .into_iter()
            .find(|path| path.exists());
        let legacy_present = legacy_path.is_some();
        let existing_path = if canonical_exists {
            Some(canonical_path.clone())
        } else {
            legacy_path.clone()
        };
        let exists = canonical_exists;
        let verified = if let (true, Some(expected_sha256)) = (canonical_exists, sha256.as_deref()) {
            verify_local_artifact_hash(&canonical_path, expected_sha256).unwrap_or(false)
        } else if canonical_exists {
            true
        } else {
            false
        };
        artifacts.push(ResolvedArtifact {
            kind: kind.to_string(),
            filename: trimmed.to_string(),
            relative_path,
            required,
            manual: manual || stale_legacy_mirror,
            exists,
            canonical_present: canonical_exists,
            legacy_present,
            resolved_path: existing_path.as_ref().map(|path| path.display().to_string()),
            source,
            source_host,
            sha256,
            size_bytes,
            sources: effective_sources,
            verified,
        });
    }

    let mut artifacts: Vec<ResolvedArtifact> = Vec::new();
    let mut seen_paths: Vec<String> = Vec::new();

    if let Some(download_artifacts) = model_obj
        .get("download")
        .and_then(|v| v.get("artifacts"))
        .and_then(|v| v.as_array())
    {
        for item in download_artifacts {
            let Some(filename) = item.get("filename").and_then(|v| v.as_str()) else {
                continue;
            };
            let kind = item.get("kind").and_then(|v| v.as_str()).unwrap_or("aux");
            let required = item
                .get("required")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let mut artifact_sources: Vec<ResolvedArtifactSource> = item
                .get("sources")
                .and_then(|v| v.as_array())
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| {
                            let url = entry.get("url").and_then(|v| v.as_str())?.trim().to_string();
                            if url.is_empty() {
                                return None;
                            }
                            let (provider, resolver, locator) =
                                infer_source_provider_resolver(&url);
                            Some(ResolvedArtifactSource {
                                url: url.clone(),
                                host: entry
                                    .get("host")
                                    .and_then(|v| v.as_str())
                                    .map(|v| v.to_string())
                                    .or_else(|| url_host(&url))
                                    .unwrap_or_else(|| "unknown".to_string()),
                                provider: entry
                                    .get("provider")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(&provider)
                                    .to_string(),
                                resolver: entry
                                    .get("resolver")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(&resolver)
                                    .to_string(),
                                locator: entry.get("locator").cloned().or(locator),
                                channel: entry
                                    .get("channel")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_else(|| {
                                        if url.contains("github.com/engdahlz/stemsep-models") {
                                            "mirror"
                                        } else {
                                            "upstream"
                                        }
                                    })
                                    .to_string(),
                                priority: entry
                                    .get("priority")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(1) as usize,
                                auth: entry
                                    .get("auth")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("none")
                                    .to_string(),
                                verified: entry
                                    .get("verified")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(true),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let legacy_source = item
                .get("source")
                .and_then(|v| v.as_str())
                .map(|v| v.to_string());
            if artifact_sources.is_empty() {
                if let Some(url) = legacy_source.clone() {
                    let (provider, resolver, locator) = infer_source_provider_resolver(&url);
                    artifact_sources.push(ResolvedArtifactSource {
                        url: url.clone(),
                        host: url_host(&url).unwrap_or_else(|| "unknown".to_string()),
                        provider,
                        resolver,
                        locator,
                        channel: item
                            .get("channel")
                            .and_then(|v| v.as_str())
                            .unwrap_or_else(|| {
                                if url.contains("github.com/engdahlz/stemsep-models") {
                                    "mirror"
                                } else {
                                    "upstream"
                                }
                            })
                            .to_string(),
                        priority: 1,
                        auth: item
                            .get("auth")
                            .and_then(|v| v.as_str())
                            .unwrap_or("none")
                            .to_string(),
                        verified: item
                            .get("verified")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true),
                    });
                }
            }
            artifact_sources.sort_by_key(|entry| entry.priority);
            let source = artifact_sources.first().map(|entry| entry.url.clone());
            let manual = item
                .get("manual")
                .and_then(|v| v.as_bool())
                .unwrap_or(manual_registry || artifact_sources.is_empty());
            let relative_path = item
                .get("relative_path")
                .and_then(|v| v.as_str())
                .map(|v| v.replace('\\', "/"));
            let sha256 = item
                .get("sha256")
                .and_then(|v| v.as_str())
                .map(|v| v.to_string());
            let size_bytes = item.get("size_bytes").and_then(|v| v.as_u64());
            push_artifact_entry(
                &cfg.models_dir,
                model_obj,
                model_id,
                &mut artifacts,
                &mut seen_paths,
                kind,
                filename,
                source,
                required,
                manual,
                sha256,
                size_bytes,
                relative_path,
                artifact_sources,
            );
        }
    } else {
        if let Some(artifacts_obj) = model_obj.get("artifacts").and_then(|v| v.as_object()) {
            if let Some(primary) = artifacts_obj.get("primary").and_then(|v| v.as_object()) {
                if let Some(filename) = primary.get("filename").and_then(|v| v.as_str()) {
                    let source = links
                        .and_then(|v| v.get("checkpoint"))
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_string());
                    let kind = primary.get("kind").and_then(|v| v.as_str()).unwrap_or("checkpoint");
                    let sha256 = primary.get("sha256").and_then(|v| v.as_str()).map(|v| v.to_string());
                    let manual = manual_registry && source.is_none();
                    push_artifact_entry(
                        &cfg.models_dir,
                        model_obj,
                        model_id,
                        &mut artifacts,
                        &mut seen_paths,
                        kind,
                        filename,
                        source,
                        true,
                        manual,
                        sha256,
                        primary.get("size_bytes").and_then(|v| v.as_u64()),
                        None,
                        Vec::new(),
                    );
                }
            }
            if let Some(config_obj) = artifacts_obj.get("config").and_then(|v| v.as_object()) {
                if let Some(filename) = config_obj.get("filename").and_then(|v| v.as_str()) {
                    let source = links
                        .and_then(|v| v.get("config"))
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_string());
                    let kind = config_obj.get("kind").and_then(|v| v.as_str()).unwrap_or("config");
                    let sha256 = config_obj.get("sha256").and_then(|v| v.as_str()).map(|v| v.to_string());
                    let manual = manual_registry && source.is_none();
                    push_artifact_entry(
                        &cfg.models_dir,
                        model_obj,
                        model_id,
                        &mut artifacts,
                        &mut seen_paths,
                        kind,
                        filename,
                        source,
                        true,
                        manual,
                        sha256,
                        config_obj.get("size_bytes").and_then(|v| v.as_u64()),
                        None,
                        Vec::new(),
                    );
                }
            }
            if let Some(additional) = artifacts_obj.get("additional").and_then(|v| v.as_array()) {
                for entry in additional {
                    let Some(filename) = entry.get("filename").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let source = entry.get("url").and_then(|v| v.as_str()).map(|v| v.to_string());
                    let kind = entry.get("kind").and_then(|v| v.as_str()).unwrap_or("aux");
                    let sha256 = entry.get("sha256").and_then(|v| v.as_str()).map(|v| v.to_string());
                    let manual = manual_registry && source.is_none();
                    push_artifact_entry(
                        &cfg.models_dir,
                        model_obj,
                        model_id,
                        &mut artifacts,
                        &mut seen_paths,
                        kind,
                        filename,
                        source,
                        true,
                        manual,
                        sha256,
                        entry.get("size_bytes").and_then(|v| v.as_u64()),
                        None,
                        Vec::new(),
                    );
                }
            }
        }

        if artifacts.is_empty() {
            if let Some(filename) = runtime
                .and_then(|v| v.get("checkpoint_ref"))
                .and_then(|v| v.as_str())
            {
                let source = links
                    .and_then(|v| v.get("checkpoint"))
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string());
                let manual = manual_registry || source.is_none();
                push_artifact_entry(
                    &cfg.models_dir,
                    model_obj,
                    model_id,
                    &mut artifacts,
                    &mut seen_paths,
                    "checkpoint",
                    filename,
                    source,
                    true,
                    manual,
                    None,
                    None,
                    None,
                    Vec::new(),
                );
            } else if let Some(url) = links
                .and_then(|v| v.get("checkpoint"))
                .and_then(|v| v.as_str())
            {
                let filename = url_basename(url).unwrap_or_else(|| format!("{model_id}.bin"));
                push_artifact_entry(
                    &cfg.models_dir,
                    model_obj,
                    model_id,
                    &mut artifacts,
                    &mut seen_paths,
                    "checkpoint",
                    &filename,
                    Some(url.to_string()),
                    true,
                    false,
                    None,
                    None,
                    None,
                    Vec::new(),
                );
            }
        }

        if let Some(filename) = runtime
            .and_then(|v| v.get("config_ref"))
            .and_then(|v| v.as_str())
        {
            let source = links
                .and_then(|v| v.get("config"))
                .and_then(|v| v.as_str())
                .map(|v| v.to_string());
            let manual = manual_registry || source.is_none();
            push_artifact_entry(
                &cfg.models_dir,
                model_obj,
                model_id,
                &mut artifacts,
                &mut seen_paths,
                "config",
                filename,
                source,
                true,
                manual,
                None,
                None,
                None,
                Vec::new(),
            );
        } else if let Some(url) = links
            .and_then(|v| v.get("config"))
            .and_then(|v| v.as_str())
        {
            if let Some(filename) = url_basename(url) {
                push_artifact_entry(
                    &cfg.models_dir,
                    model_obj,
                    model_id,
                    &mut artifacts,
                    &mut seen_paths,
                    "config",
                    &filename,
                    Some(url.to_string()),
                    true,
                    false,
                    None,
                    None,
                    None,
                    Vec::new(),
                );
            }
        }
    }

    if let Some(required_files) = runtime
        .and_then(|v| v.get("required_files"))
        .and_then(|v| v.as_array())
    {
        for filename in required_files.iter().filter_map(|v| v.as_str()) {
            let already_known = artifacts.iter().any(|artifact| artifact.filename == filename);
            if already_known {
                continue;
            }
            push_artifact_entry(
                &cfg.models_dir,
                model_obj,
                model_id,
                &mut artifacts,
                &mut seen_paths,
                "aux",
                filename,
                None,
                true,
                true,
                None,
                None,
                None,
                Vec::new(),
            );
        }
    }

    let manual_instructions = if let Some(notes) = install.and_then(|v| v.get("notes")).and_then(|v| v.as_array()) {
        notes
            .iter()
            .filter_map(|value| value.as_str())
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut manual_instructions = manual_instructions;
    let needs_manual_guidance = manual_registry
        || sources
            .iter()
            .any(|source| is_stale_legacy_mirror_url(&source.url));
    if needs_manual_guidance {
        if !manual_instructions.iter().any(|entry| entry.contains("Place")) {
            let folder = format!("{}/{}/", model_family(model_obj), model_id);
            manual_instructions.push(format!(
                "Place the required files under {folder} inside the configured models directory."
            ));
        }
        if sources.iter().any(|source| is_stale_legacy_mirror_url(&source.url)) {
            manual_instructions.push(
                "Legacy StemSep mirror links for this model are stale. Use the listed source references or install the files manually until a verified direct upstream is curated."
                    .to_string(),
            );
        }
        if sources.is_empty() {
            manual_instructions.push(
                "This model does not expose verified direct-download URLs yet. Use the listed required filenames and install notes for manual setup."
                    .to_string(),
            );
        }
    }

    let downloadable_artifact_count = artifacts
        .iter()
        .filter(|artifact| !artifact.manual && artifact.source.is_some())
        .count();
    let mode = if artifacts.is_empty() {
        "unavailable".to_string()
    } else if downloadable_artifact_count == 0 {
        "manual".to_string()
    } else if artifacts.len() > 1 {
        "multi_artifact_direct".to_string()
    } else {
        "direct".to_string()
    };
    let source_policy = if sources
        .iter()
        .any(|source| is_stale_legacy_mirror_url(&source.url))
    {
        "legacy_mirror_manual".to_string()
    } else if sources
        .iter()
        .any(|source| source.host.contains("github.com/engdahlz/stemsep-models"))
    {
        "mirror_fallback".to_string()
    } else if manual_registry {
        "manual_verified".to_string()
    } else {
        "upstream_direct".to_string()
    };

    ResolvedDownloadManifest {
        model_id: model_id.to_string(),
        family: model_family(model_obj),
        mode,
        strategy: availability.class_name.clone(),
        install_mode,
        artifact_count: artifacts.len(),
        downloadable_artifact_count,
        source_policy,
        availability,
        sources,
        artifacts,
        manual_instructions,
    }
}

fn parse_url(url: &str) -> Option<reqwest::Url> {
    reqwest::Url::parse(url).ok()
}

fn github_release_locator(url: &str) -> Option<Value> {
    let parsed = parse_url(url)?;
    let segments = parsed.path_segments()?.collect::<Vec<_>>();
    if segments.len() < 6 || segments.get(2) != Some(&"releases") || segments.get(3) != Some(&"download") {
        return None;
    }
    Some(serde_json::json!({
        "owner": segments[0],
        "repo": segments[1],
        "tag": segments[4],
        "asset_name": segments[5..].join("/"),
    }))
}

fn github_raw_locator(url: &str) -> Option<Value> {
    let parsed = parse_url(url)?;
    let segments = parsed.path_segments()?.collect::<Vec<_>>();
    match parsed.host_str()? {
        "raw.githubusercontent.com" if segments.len() >= 4 => Some(serde_json::json!({
            "owner": segments[0],
            "repo": segments[1],
            "revision": segments[2],
            "file_path": segments[3..].join("/"),
        })),
        "github.com" if segments.len() >= 5 && segments.get(2) == Some(&"raw") => Some(serde_json::json!({
            "owner": segments[0],
            "repo": segments[1],
            "revision": segments[3],
            "file_path": segments[4..].join("/"),
        })),
        _ => None,
    }
}

fn huggingface_resolve_locator(url: &str) -> Option<Value> {
    let parsed = parse_url(url)?;
    if parsed.host_str()? != "huggingface.co" {
        return None;
    }
    let segments = parsed.path_segments()?.collect::<Vec<_>>();
    if segments.len() < 5 || segments.get(2) != Some(&"resolve") {
        return None;
    }
    Some(serde_json::json!({
        "repo_id": format!("{}/{}", segments[0], segments[1]),
        "revision": segments[3],
        "file_path": segments[4..].join("/"),
    }))
}

fn google_drive_file_id_from_url(url: &str) -> Option<String> {
    let parsed = parse_url(url)?;
    let host = parsed.host_str()?.to_lowercase();
    if !host.contains("drive.google.com") && !host.contains("drive.usercontent.google.com") {
        return None;
    }
    if let Some(id) = parsed
        .query_pairs()
        .find(|(key, _)| key == "id")
        .map(|(_, value)| value.to_string())
        .filter(|value| !value.trim().is_empty())
    {
        return Some(id);
    }
    let segments = parsed.path_segments()?.collect::<Vec<_>>();
    for index in 0..segments.len() {
        if segments[index] == "d" && index + 1 < segments.len() {
            return Some(segments[index + 1].to_string());
        }
        if segments[index] == "folders" && index + 1 < segments.len() {
            return Some(segments[index + 1].to_string());
        }
    }
    None
}

fn proton_share_locator(url: &str) -> Option<Value> {
    let parsed = parse_url(url)?;
    let host = parsed.host_str()?.to_lowercase();
    if !host.contains("drive.proton.me") {
        return None;
    }
    let token = parsed
        .fragment()
        .map(|fragment| fragment.to_string())
        .filter(|fragment| !fragment.trim().is_empty());
    Some(serde_json::json!({
        "share_url": url,
        "share_token": token,
    }))
}

fn infer_source_provider_resolver(url: &str) -> (String, String, Option<Value>) {
    if let Some(locator) = huggingface_resolve_locator(url) {
        return (
            "huggingface".to_string(),
            "huggingface_resolve".to_string(),
            Some(locator),
        );
    }
    if let Some(locator) = github_release_locator(url) {
        return (
            "github".to_string(),
            "github_release_asset".to_string(),
            Some(locator),
        );
    }
    if let Some(locator) = github_raw_locator(url) {
        return ("github".to_string(), "github_raw".to_string(), Some(locator));
    }
    if let Some(file_id) = google_drive_file_id_from_url(url) {
        let resolver = if url.contains("/folders/") {
            "google_drive_folder_entry"
        } else {
            "google_drive_file"
        };
        return (
            "google_drive".to_string(),
            resolver.to_string(),
            Some(serde_json::json!({ "file_id": file_id })),
        );
    }
    if let Some(locator) = proton_share_locator(url) {
        return (
            "proton_drive".to_string(),
            "proton_share_entry".to_string(),
            Some(locator),
        );
    }
    ("static".to_string(), "static_url".to_string(), None)
}

fn resolve_installation_status(
    _cfg: &BackendConfig,
    manifest: &ResolvedDownloadManifest,
) -> ResolvedInstallationStatus {
    let relative_paths = manifest
        .artifacts
        .iter()
        .map(|artifact| artifact.relative_path.clone())
        .collect::<Vec<_>>();
    let artifacts = manifest
        .artifacts
        .iter()
        .map(|artifact| ResolvedArtifactInstallation {
            kind: artifact.kind.clone(),
            filename: artifact.filename.clone(),
            relative_path: artifact.relative_path.clone(),
            present: artifact.exists,
            verified: artifact.verified,
            canonical_present: artifact.canonical_present,
            legacy_present: artifact.legacy_present,
            resolved_path: artifact.resolved_path.clone(),
            required: artifact.required,
            selected_source: artifact.source.clone(),
        })
        .collect::<Vec<_>>();
    let missing_artifacts = manifest
        .artifacts
        .iter()
        .filter(|artifact| artifact.required && (!artifact.exists || !artifact.verified))
        .map(|artifact| artifact.relative_path.clone())
        .collect::<Vec<_>>();
    let source_resolution = manifest
        .artifacts
        .iter()
        .map(|artifact| {
            let selected = artifact_download_sources(artifact)
                .into_iter()
                .next()
                .map(|source| {
                    serde_json::json!({
                        "url": source.url,
                        "provider": source.provider,
                        "resolver": source.resolver,
                        "locator": source.locator,
                        "priority": source.priority,
                        "channel": source.channel,
                        "verified": source.verified,
                    })
                })
                .unwrap_or(Value::Null);
            serde_json::json!({
                "relative_path": artifact.relative_path,
                "selected_source": selected,
            })
        })
        .collect::<Vec<_>>();
    let canonical_ready = manifest
        .artifacts
        .iter()
        .filter(|artifact| artifact.required)
        .all(|artifact| artifact.canonical_present && artifact.verified)
        && !manifest.artifacts.is_empty();
    let verified_hashes = manifest
        .artifacts
        .iter()
        .filter(|artifact| artifact.required)
        .all(|artifact| artifact.sha256.is_none() || artifact.verified);
    let legacy_fallback_used = manifest
        .artifacts
        .iter()
        .any(|artifact| artifact.required && artifact.legacy_present && !artifact.canonical_present);
    ResolvedInstallationStatus {
        installed: missing_artifacts.is_empty() && !manifest.artifacts.is_empty(),
        canonical_ready,
        legacy_fallback_used,
        verified_hashes,
        missing_artifacts,
        relative_paths,
        source_resolution,
        artifacts,
    }
}

fn artifact_download_sources(artifact: &ResolvedArtifact) -> Vec<ResolvedArtifactSource> {
    let mut sources = artifact.sources.clone();
    if sources.is_empty() {
        if let Some(url) = artifact.source.clone() {
            let (provider, resolver, locator) = infer_source_provider_resolver(&url);
            sources.push(ResolvedArtifactSource {
                url: url.clone(),
                host: artifact
                    .source_host
                    .clone()
                    .or_else(|| url_host(&url))
                    .unwrap_or_else(|| "unknown".to_string()),
                provider,
                resolver,
                locator,
                channel: if url.contains("github.com/engdahlz/stemsep-models") {
                    "mirror".to_string()
                } else {
                    "upstream".to_string()
                },
                priority: 1,
                auth: "none".to_string(),
                verified: true,
            });
        }
    }
    sources.sort_by_key(|entry| {
        (
            if entry.channel == "upstream" { 0usize } else { 1usize },
            entry.priority,
        )
    });
    sources
}

fn source_locator_string(locator: &Option<Value>, key: &str) -> Option<String> {
    locator
        .as_ref()
        .and_then(|value| value.get(key))
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn resolve_download_source_url(source: &ResolvedArtifactSource) -> Result<String> {
    match source.resolver.as_str() {
        "static_url" | "huggingface_resolve" | "github_release_asset" | "github_raw" => {
            Ok(source.url.clone())
        }
        "google_drive_file" => {
            let file_id = source_locator_string(&source.locator, "file_id")
                .or_else(|| google_drive_file_id_from_url(&source.url))
                .ok_or_else(|| {
                    anyhow!(
                        "google_drive_file source is missing locator.file_id: {}",
                        source.url
                    )
                })?;
            let mut url = reqwest::Url::parse("https://drive.usercontent.google.com/download")
                .context("build google drive download URL")?;
            url.query_pairs_mut()
                .append_pair("id", &file_id)
                .append_pair("export", "download")
                .append_pair("confirm", "t");
            Ok(url.to_string())
        }
        "google_drive_folder_entry" => {
            if let Some(download_url) = source_locator_string(&source.locator, "download_url") {
                return Ok(download_url);
            }
            let file_id = source_locator_string(&source.locator, "file_id")
                .or_else(|| google_drive_file_id_from_url(&source.url))
                .ok_or_else(|| {
                    anyhow!(
                        "google_drive_folder_entry requires locator.download_url or locator.file_id: {}",
                        source.url
                    )
                })?;
            let mut url = reqwest::Url::parse("https://drive.usercontent.google.com/download")
                .context("build google drive folder-entry URL")?;
            url.query_pairs_mut()
                .append_pair("id", &file_id)
                .append_pair("export", "download")
                .append_pair("confirm", "t");
            Ok(url.to_string())
        }
        "proton_share_file" | "proton_share_entry" => {
            if let Some(download_url) = source_locator_string(&source.locator, "download_url") {
                return Ok(download_url);
            }
            Err(anyhow!(
                "{} source requires locator.download_url; raw share pages are not auto-downloadable",
                source.resolver
            ))
        }
        other => Err(anyhow!(
            "Unsupported artifact source resolver '{other}' for {}",
            source.url
        )),
    }
}

fn download_with_progress_cancellable(
    url: &str,
    dest_path: &Path,
    stdout: Arc<Mutex<io::Stdout>>,
    model_id: String,
    artifact_index: usize,
    artifact_count: usize,
    current_file: String,
    current_relative_path: String,
    current_source: String,
    expected_sha256: Option<String>,
    expected_size: Option<u64>,
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
    let metadata_path = part_metadata_path(&tmp_path);
    let enforce_size = expected_sha256.is_none();

    let part_metadata = PartMetadata {
        model_id: model_id.clone(),
        filename: current_file.clone(),
        relative_path: current_relative_path.clone(),
        source_url: url.to_string(),
        sha256: expected_sha256.clone(),
        size_bytes: expected_size,
    };
    if let Some(existing_meta) = read_part_metadata(&metadata_path) {
        if existing_meta.source_url != part_metadata.source_url
            || existing_meta.sha256 != part_metadata.sha256
            || existing_meta.size_bytes != part_metadata.size_bytes
        {
            let _ = std::fs::remove_file(&tmp_path);
            remove_part_metadata(&metadata_path);
        }
    }
    write_part_metadata(&metadata_path, &part_metadata)?;

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
            "artifactIndex": artifact_index,
            "artifactCount": artifact_count,
            "currentFile": current_file,
            "currentRelativePath": current_relative_path,
            "currentSource": current_source,
            "stage": "downloading",
            "verified": false,
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
            if enforce_size {
                if let Some(expected) = expected_size {
                    let actual = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);
                    if actual != expected {
                        return Err(anyhow!(
                            "partial file size mismatch for {}: expected {expected} bytes, got {actual}",
                            current_relative_path
                        ));
                    }
                }
            }
            if let Some(expected_sha256) = expected_sha256.as_deref() {
                if !verify_local_artifact_hash(&tmp_path, expected_sha256)
                    .context("verify sha256 for resumed partial file")?
                {
                    return Err(anyhow!(
                        "sha256 mismatch for {} after resume",
                        current_relative_path
                    ));
                }
            }
            if dest_path.exists() {
                let _ = std::fs::remove_file(&tmp_path);
            } else {
                std::fs::rename(&tmp_path, dest_path)
                    .with_context(|| format!("move into place: {}", dest_path.display()))?;
            }
            remove_part_metadata(&metadata_path);

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
                    "artifactIndex": artifact_index,
                    "artifactCount": artifact_count,
                    "currentFile": current_file,
                    "currentRelativePath": current_relative_path,
                    "currentSource": current_source,
                    "stage": "downloading",
                    "verified": false,
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
                    "artifactIndex": artifact_index,
                    "artifactCount": artifact_count,
                    "currentFile": current_file,
                    "currentRelativePath": current_relative_path,
                    "currentSource": current_source,
                    "stage": "paused",
                    "verified": false,
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
                        "artifactIndex": artifact_index,
                        "artifactCount": artifact_count,
                        "currentFile": current_file,
                        "currentRelativePath": current_relative_path,
                        "currentSource": current_source,
                        "stage": "downloading",
                        "verified": false,
                    }),
                );
            }
        }
    }

    out.flush().ok();
    drop(out);

    send_event(
        &stdout,
        serde_json::json!({
            "type": "progress",
            "model_id": model_id,
            "progress": 100,
            "artifactIndex": artifact_index,
            "artifactCount": artifact_count,
            "currentFile": current_file,
            "currentRelativePath": current_relative_path,
            "currentSource": current_source,
            "stage": "verifying",
            "verified": false,
        }),
    );

    if enforce_size {
        if let Some(expected) = expected_size {
            let actual = std::fs::metadata(&tmp_path)
                .with_context(|| format!("stat {}", tmp_path.display()))?
                .len();
            if actual != expected {
                let _ = std::fs::remove_file(&tmp_path);
                remove_part_metadata(&metadata_path);
                return Err(anyhow!(
                    "size mismatch for {}: expected {expected} bytes, got {actual}",
                    current_relative_path
                ));
            }
        }
    }
    if let Some(expected_sha256) = expected_sha256.as_deref() {
        if !verify_local_artifact_hash(&tmp_path, expected_sha256)
            .with_context(|| format!("verify sha256 for {}", current_relative_path))?
        {
            let _ = std::fs::remove_file(&tmp_path);
            remove_part_metadata(&metadata_path);
            return Err(anyhow!("sha256 mismatch for {}", current_relative_path));
        }
    }

    // Atomic-ish replace
    if dest_path.exists() {
        let _ = std::fs::remove_file(dest_path);
    }
    std::fs::rename(&tmp_path, dest_path)
        .with_context(|| format!("move into place: {}", dest_path.display()))?;
    remove_part_metadata(&metadata_path);

    Ok(())
}

#[derive(Clone)]
struct DownloadFile {
    dest_path: PathBuf,
    relative_path: String,
    filename: String,
    sources: Vec<ResolvedArtifactSource>,
    sha256: Option<String>,
    size_bytes: Option<u64>,
}

#[derive(Clone)]
struct DownloadTask {
    cancel: Arc<AtomicBool>,
}

#[derive(Clone, Default)]
struct DownloadManager {
    tasks: Arc<Mutex<HashMap<String, DownloadTask>>>,
}

impl DownloadManager {
    fn start(&self, model_id: &str, files: Vec<DownloadFile>, stdout: Arc<Mutex<io::Stdout>>) -> bool {
        let cancel = Arc::new(AtomicBool::new(false));
        let task = DownloadTask { cancel: cancel.clone() };
        {
            let mut map = self.tasks.lock().expect("downloads lock");
            if map.contains_key(model_id) {
                return false;
            }
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

                let mut source_errors: Vec<String> = Vec::new();
                let mut completed = false;
                for source in &f.sources {
                    let resolved_url = match resolve_download_source_url(source) {
                        Ok(url) => url,
                        Err(error) => {
                            source_errors.push(format!(
                                "{} [{}:{}]: {error:#}",
                                source.url, source.provider, source.resolver
                            ));
                            continue;
                        }
                    };
                    let res = download_with_progress_cancellable(
                        &resolved_url,
                        &f.dest_path,
                        stdout.clone(),
                        model_id_clone.clone(),
                        idx + 1,
                        files.len(),
                        f.filename.clone(),
                        f.relative_path.clone(),
                        resolved_url.clone(),
                        f.sha256.clone(),
                        f.size_bytes,
                        Some(cancel.clone()),
                    );

                    match res {
                        Ok(()) => {
                            if cancel.load(Ordering::Relaxed) {
                                return;
                            }
                            if f.dest_path.exists() {
                                completed_paths.push(f.dest_path.to_string_lossy().to_string());
                            }
                            completed = true;
                            break;
                        }
                        Err(e) => {
                            source_errors.push(format!(
                                "{} [{}:{}]: {e:#}",
                                source.url, source.provider, source.resolver
                            ));
                        }
                    }
                }

                if !completed {
                    send_event(
                        &stdout,
                        serde_json::json!({
                            "type": "error",
                            "model_id": model_id_clone,
                            "artifactIndex": idx + 1,
                            "artifactCount": files.len(),
                            "currentFile": f.filename,
                            "currentRelativePath": f.relative_path,
                            "stage": "failed",
                            "verified": false,
                            "error": if source_errors.is_empty() {
                                "No valid download source succeeded".to_string()
                            } else {
                                format!("All download sources failed:\n{}", source_errors.join("\n"))
                            }
                        }),
                    );
                    mgr.clear_task(&model_id_clone);
                    return;
                }
            }

            // All files done.
            send_event(
                &stdout,
                serde_json::json!({
                    "type": "progress",
                    "model_id": model_id_clone,
                    "progress": 100,
                    "stage": "verifying",
                    "verified": true,
                }),
            );
            send_event(
                &stdout,
                serde_json::json!({
                    "type": "complete",
                    "model_id": model_id_clone,
                    "path": primary_path,
                    "paths": completed_paths,
                    "artifactCount": files.len(),
                    "stage": "installed",
                    "verified": true,
                }),
            );
            mgr.clear_task(&model_id_clone);
        });
        true
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

    fn clear_task(&self, model_id: &str) {
        let mut map = self.tasks.lock().expect("downloads lock");
        map.remove(model_id);
    }
}

fn remove_model_files(cfg: &BackendConfig, model_id: &str) -> Result<usize> {
    let models_json = load_models_with_guide_overrides(cfg)?;
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(model_obj) = find_model_object(&models_json, model_id) {
        let manifest = resolve_model_download_manifest(cfg, model_id, &model_obj);
        for artifact in &manifest.artifacts {
            let target = cfg.models_dir.join(artifact.relative_path.replace('/', "\\"));
            candidates.push(target.clone());
            candidates.push(target.with_file_name(format!("{}.part", artifact.filename)));
            for legacy in legacy_artifact_paths(&cfg.models_dir, model_id, &artifact.kind, &artifact.filename) {
                candidates.push(legacy.clone());
                if let Some(name) = legacy.file_name().and_then(|value| value.to_str()) {
                    candidates.push(legacy.with_file_name(format!("{name}.part")));
                }
            }
        }
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

fn copy_or_link_file(src: &Path, dest: &Path, allow_copy: bool) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    if dest.exists() {
        std::fs::remove_file(dest).with_context(|| format!("remove {}", dest.display()))?;
    }
    match std::fs::hard_link(src, dest) {
        Ok(()) => Ok(()),
        Err(link_error) => {
            if !allow_copy {
                return Err(link_error).with_context(|| {
                    format!("hard-link {} -> {}", src.display(), dest.display())
                });
            }
            std::fs::copy(src, dest)
                .with_context(|| format!("copy {} -> {}", src.display(), dest.display()))?;
            Ok(())
        }
    }
}

fn import_model_artifacts_from_registry(
    cfg: &BackendConfig,
    registry: &Value,
    model_id: &str,
    files: &[Value],
    allow_copy: bool,
) -> Result<(ResolvedDownloadManifest, ResolvedInstallationStatus)> {
    let model_obj = find_model_object(registry, model_id)
        .ok_or_else(|| anyhow!("Unknown model_id: {model_id}"))?;
    let manifest = resolve_model_download_manifest(cfg, model_id, &model_obj);
    if manifest.availability.class_name == "blocked_non_public" {
        return Err(anyhow!(
            "{}",
            manifest
                .availability
                .reason
                .clone()
                .unwrap_or_else(|| format!("Model {model_id} is blocked and cannot be imported"))
        ));
    }
    if files.is_empty() {
        return Err(anyhow!("No files supplied for import"));
    }

    let mut used_targets: Vec<String> = Vec::new();
    let mut matched_any = false;
    for file in files {
        let path = file
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Each import file must include a path"))?;
        let src = PathBuf::from(path);
        if !src.exists() {
            return Err(anyhow!("Import source does not exist: {}", src.display()));
        }
        let kind = file
            .get("kind")
            .and_then(|v| v.as_str())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let filename = src
            .file_name()
            .and_then(|value| value.to_str())
            .map(|value| value.to_string())
            .unwrap_or_default();

        let mut candidates: Vec<&ResolvedArtifact> = manifest
            .artifacts
            .iter()
            .filter(|artifact| artifact.filename == filename)
            .collect();
        if candidates.is_empty() {
            if let Some(kind_name) = kind.as_deref() {
                candidates = manifest
                    .artifacts
                    .iter()
                    .filter(|artifact| artifact.kind == kind_name)
                    .collect();
            }
        }
        if candidates.is_empty() {
            return Err(anyhow!(
                "Could not match imported file '{}' to any artifact for model {}",
                filename,
                model_id
            ));
        }
        if candidates.len() > 1 {
            return Err(anyhow!(
                "Imported file '{}' matched multiple artifacts for model {}; specify a unique filename",
                filename,
                model_id
            ));
        }
        let artifact = candidates.remove(0);
        if used_targets.iter().any(|target| target == &artifact.relative_path) {
            return Err(anyhow!(
                "Multiple import files map to the same artifact {}",
                artifact.relative_path
            ));
        }
        used_targets.push(artifact.relative_path.clone());
        let dest = cfg
            .models_dir
            .join(artifact.relative_path.replace('/', "\\"));
        copy_or_link_file(&src, &dest, allow_copy)?;
        if let Some(expected_size) = artifact.size_bytes {
            let actual_size = std::fs::metadata(&dest)
                .with_context(|| format!("stat {}", dest.display()))?
                .len();
            if actual_size != expected_size {
                let _ = std::fs::remove_file(&dest);
                return Err(anyhow!(
                    "Imported file size mismatch for {}: expected {expected_size} bytes, got {actual_size}",
                    artifact.relative_path
                ));
            }
        }
        if let Some(expected_sha256) = artifact.sha256.as_deref() {
            if !verify_local_artifact_hash(&dest, expected_sha256)
                .with_context(|| format!("verify sha256 for {}", dest.display()))?
            {
                let _ = std::fs::remove_file(&dest);
                return Err(anyhow!(
                    "Imported file failed sha256 verification for {}",
                    artifact.relative_path
                ));
            }
        }
        matched_any = true;
    }

    if !matched_any {
        return Err(anyhow!("No import files were matched to manifest artifacts"));
    }

    let refreshed_manifest = resolve_model_download_manifest(cfg, model_id, &model_obj);
    let installation = resolve_installation_status(cfg, &refreshed_manifest);
    Ok((refreshed_manifest, installation))
}

fn import_model_artifacts(
    cfg: &BackendConfig,
    model_id: &str,
    files: &[Value],
    allow_copy: bool,
) -> Result<(ResolvedDownloadManifest, ResolvedInstallationStatus)> {
    let models_json = load_models_with_guide_overrides(cfg)?;
    import_model_artifacts_from_registry(cfg, &models_json, model_id, files, allow_copy)
}

fn resolve_selection_request(runtime: &Value, extra: &Map<String, Value>) -> Result<(String, String)> {
    let (selection_type, selection_id) = selection_descriptor_from_extra(extra);
    let selection_type = selection_type
        .or_else(|| {
            selection_id.as_ref().and_then(|id| {
                runtime
                    .get("selection_index")
                    .and_then(|v| v.as_array())
                    .and_then(|entries| {
                        entries.iter().find(|entry| {
                            entry.get("selection_id").and_then(|v| v.as_str()) == Some(id.as_str())
                        })
                    })
                    .and_then(|entry| entry.get("selection_type").and_then(|v| v.as_str()))
                    .and_then(normalize_selection_type)
            })
        })
        .or_else(|| {
            extra.get("model_id")
                .and_then(|v| v.as_str())
                .map(|_| "model".to_string())
        })
        .ok_or_else(|| anyhow!("selection_type is required"))?;
    let selection_id = selection_id.ok_or_else(|| anyhow!("selection_id is required"))?;
    Ok((selection_type, selection_id))
}

fn schedule_model_download_for_object(
    cfg: &BackendConfig,
    downloads: &DownloadManager,
    stdout: Arc<Mutex<io::Stdout>>,
    model_id: &str,
    model_obj: &Value,
) -> Result<Value> {
    let manifest = resolve_model_download_manifest(cfg, model_id, model_obj);
    let installation = resolve_installation_status(cfg, &manifest);
    let files: Vec<DownloadFile> = manifest
        .artifacts
        .iter()
        .filter(|artifact| {
            artifact.required && !artifact.manual && artifact.source.is_some() && !artifact.exists
        })
        .map(|artifact| DownloadFile {
            dest_path: cfg.models_dir.join(artifact.relative_path.replace('/', "\\")),
            relative_path: artifact.relative_path.clone(),
            filename: artifact.filename.clone(),
            sources: artifact_download_sources(artifact),
            sha256: artifact.sha256.clone(),
            size_bytes: artifact.size_bytes,
        })
        .collect();

    if files.is_empty() {
        if installation.installed {
            return Ok(serde_json::json!({
                "scheduled": false,
                "model_id": model_id,
                "already_installed": true,
                "download": serde_json::to_value(&manifest).unwrap_or(Value::Null),
                "installation": serde_json::to_value(&installation).unwrap_or(Value::Null)
            }));
        }
        if manifest.mode == "manual" {
            return Err(anyhow!(
                "Model {model_id} requires manual setup. Open Model Details for required files and source links."
            ));
        }
        return Err(anyhow!("No downloadable artifacts found for model_id: {model_id}"));
    }

    let primary_path = files
        .first()
        .map(|file| file.dest_path.to_string_lossy().to_string());
    let scheduled = downloads.start(model_id, files.clone(), stdout);

    Ok(serde_json::json!({
        "scheduled": scheduled,
        "model_id": model_id,
        "dest": primary_path,
        "files": files
            .iter()
            .map(|f| f.dest_path.to_string_lossy().to_string())
            .collect::<Vec<_>>(),
        "already_in_flight": !scheduled,
        "download": serde_json::to_value(&manifest).unwrap_or(Value::Null),
        "installation": serde_json::to_value(&installation).unwrap_or(Value::Null)
    }))
}

fn install_selection(
    cfg: &BackendConfig,
    downloads: &DownloadManager,
    stdout: Arc<Mutex<io::Stdout>>,
    extra: &Map<String, Value>,
) -> Result<Value> {
    let runtime = load_catalog_runtime(cfg)?;
    let (selection_type, selection_id) = resolve_selection_request(&runtime, extra)?;
    let plan = catalog_selection_install_plan(cfg, &runtime, &selection_type, &selection_id)?;
    let bundle = catalog_selection_bundle(cfg, &runtime, &selection_type, &selection_id)?;
    let required_models = bundle
        .get("required_models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut results: Vec<Value> = Vec::new();
    let mut skipped: Vec<Value> = Vec::new();
    let mut failures: Vec<Value> = Vec::new();

    for required_model in required_models {
        let model_id = required_model
            .get("model_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if model_id.is_empty() {
            continue;
        }

        let model_obj = required_model.get("model").cloned().unwrap_or(Value::Null);
        let installation = required_model
            .get("installation")
            .cloned()
            .unwrap_or(Value::Null);
        let download = required_model.get("download").cloned().unwrap_or(Value::Null);
        let install_mode = download
            .get("install_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let mode = download.get("mode").and_then(|v| v.as_str()).unwrap_or("");
        let availability_class = download
            .get("availability")
            .and_then(|v| v.get("class"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if required_model.get("error").and_then(|v| v.as_str()) == Some("NOT_FOUND") {
            failures.push(serde_json::json!({
                "model_id": model_id,
                "error": "NOT_FOUND",
                "reason": "Model is required by the selection but missing from the runtime catalog."
            }));
            continue;
        }

        if installation.get("installed").and_then(|v| v.as_bool()) == Some(true) {
            skipped.push(serde_json::json!({
                "model_id": model_id,
                "reason": "Model is already installed.",
                "installation": installation,
            }));
            continue;
        }

        if matches!(install_mode, "manual" | "custom_runtime")
            || matches!(mode, "manual" | "unavailable")
            || availability_class == "blocked_non_public"
        {
            skipped.push(serde_json::json!({
                "model_id": model_id,
                "reason": if availability_class == "blocked_non_public" {
                    "Model is blocked and cannot be installed automatically."
                } else if matches!(install_mode, "manual" | "custom_runtime") {
                    "Model requires manual or custom-runtime setup."
                } else {
                    "Model does not expose downloadable artifacts."
                },
                "download": download,
                "installation": installation,
            }));
            continue;
        }

        match schedule_model_download_for_object(cfg, downloads, stdout.clone(), &model_id, &model_obj) {
            Ok(result) => results.push(result),
            Err(error) => failures.push(serde_json::json!({
                "model_id": model_id,
                "error": "DOWNLOAD_FAILED",
                "reason": format!("{error:#}"),
            })),
        }
    }

    let installation = catalog_selection_installation(cfg, &runtime, &selection_type, &selection_id)?;
    Ok(serde_json::json!({
        "success": failures.is_empty(),
        "selection_type": selection_type,
        "selection_id": selection_id,
        "plan": plan,
        "results": results,
        "skipped": skipped,
        "failures": failures,
        "installation": installation,
    }))
}

fn hydrated_catalog_runtime(cfg: &BackendConfig) -> Result<Value> {
    let runtime = load_catalog_runtime(cfg)?;
    let catalog_status = ensure_catalog_runtime_status(cfg)?;
    let models = runtime
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut hydrated_models: Vec<Value> = Vec::with_capacity(models.len());
    for model in models {
        let id = model.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let (installed, download_value, installation_value) = if !id.is_empty() {
            let manifest = resolve_model_download_manifest(cfg, id, &model);
            let installation = resolve_installation_status(cfg, &manifest);
            (
                installation.installed,
                serde_json::to_value(&manifest).unwrap_or(Value::Null),
                serde_json::to_value(&installation).unwrap_or(Value::Null),
            )
        } else {
            (false, Value::Null, Value::Null)
        };
        let mut obj = model.as_object().cloned().unwrap_or_default();
        obj.insert("installed".to_string(), Value::from(installed));
        obj.insert("download".to_string(), download_value);
        obj.insert("installation".to_string(), installation_value);
        hydrated_models.push(Value::Object(obj));
    }

    let mut object = runtime.as_object().cloned().unwrap_or_default();
    object.insert("models".to_string(), Value::Array(hydrated_models));
    object.insert(
        "catalog_status".to_string(),
        serde_json::to_value(catalog_status).unwrap_or(Value::Null),
    );
    Ok(Value::Object(object))
}

fn import_selection_artifacts(
    cfg: &BackendConfig,
    extra: &Map<String, Value>,
) -> Result<Value> {
    let runtime = load_catalog_runtime(cfg)?;
    let (selection_type, selection_id) = resolve_selection_request(&runtime, extra)?;
    let plan = catalog_selection_install_plan(cfg, &runtime, &selection_type, &selection_id)?;
    let bundle = catalog_selection_bundle(cfg, &runtime, &selection_type, &selection_id)?;
    let required_model_ids = bundle
        .get("required_model_ids")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(|entry| entry.trim().to_string()))
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();
    if required_model_ids.is_empty() {
        return Err(anyhow!("Selection does not resolve to any importable models."));
    }

    let files = extra
        .get("files")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("files[] is required"))?;
    let allow_copy = extra
        .get("allow_copy")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let mut grouped: HashMap<String, Vec<Value>> = HashMap::new();
    for file in files {
        let model_id = file
            .get("model_id")
            .or_else(|| file.get("modelId"))
            .and_then(|v| v.as_str())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let resolved_model_id = match (&model_id, required_model_ids.len()) {
            (Some(model_id), _) => {
                if !required_model_ids.iter().any(|candidate| candidate == model_id) {
                    return Err(anyhow!(
                        "Import file targets model_id '{}' which is not part of selection {}:{}",
                        model_id,
                        selection_type,
                        selection_id
                    ));
                }
                model_id.clone()
            }
            (None, 1) => required_model_ids[0].clone(),
            (None, _) => {
                return Err(anyhow!(
                    "Selection {}:{} requires multiple models; each import file must include model_id",
                    selection_type,
                    selection_id
                ));
            }
        };
        grouped.entry(resolved_model_id).or_default().push(file.clone());
    }

    let mut model_results: Vec<Value> = Vec::new();
    for model_id in required_model_ids.iter() {
        if let Some(model_files) = grouped.get(model_id) {
            let (manifest, installation) =
                import_model_artifacts_from_registry(cfg, &runtime, model_id, model_files, allow_copy)?;
            model_results.push(serde_json::json!({
                "model_id": model_id,
                "download": serde_json::to_value(&manifest).unwrap_or(Value::Null),
                "installation": serde_json::to_value(&installation).unwrap_or(Value::Null),
                "artifacts": serde_json::to_value(&installation.artifacts).unwrap_or(Value::Null),
            }));
        }
    }

    if model_results.is_empty() {
        return Err(anyhow!("No import files were matched to the selection."));
    }

    let installation = catalog_selection_installation(cfg, &runtime, &selection_type, &selection_id)?;
    let verification = catalog_selection_verification(cfg, &runtime, &selection_type, &selection_id)?;
    Ok(serde_json::json!({
        "success": installation.get("installed").and_then(|v| v.as_bool()).unwrap_or(false),
        "selection_type": selection_type,
        "selection_id": selection_id,
        "plan": plan,
        "model_results": model_results,
        "installation": installation,
        "verification": verification,
    }))
}

fn any_model_file_exists(models_dir: &Path, model_id: &str, model_obj: &Value) -> bool {
    let cfg = ephemeral_backend_config(models_dir);
    let manifest = resolve_model_download_manifest(&cfg, model_id, model_obj);
    if !manifest.artifacts.is_empty() {
        return resolve_installation_status(&cfg, &manifest).installed;
    }

    for ext in [".ckpt", ".pth", ".pt", ".onnx", ".safetensors"] {
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

#[cfg(target_os = "windows")]
fn detect_playback_devices_native() -> Result<Vec<PlaybackDeviceInfo>> {
    let _ = initialize_mta().ok();
    let enumerator = DeviceEnumerator::new().context("create device enumerator")?;
    let default_id = enumerator
        .get_default_device(&Direction::Render)
        .ok()
        .and_then(|device| device.get_id().ok());
    let collection = enumerator
        .get_device_collection(&Direction::Render)
        .context("enumerate render devices")?;

    let mut devices = Vec::new();
    let count = collection.get_nbr_devices().context("count render devices")?;
    for index in 0..count {
        let device = collection
            .get_device_at_index(index)
            .with_context(|| format!("read render device at index {index}"))?;
        let id = device.get_id().context("read render device id")?;
        let label = device
            .get_friendlyname()
            .unwrap_or_else(|_| format!("Playback Device {}", index + 1));
        devices.push(PlaybackDeviceInfo {
            is_default: default_id.as_deref() == Some(id.as_str()),
            id,
            label,
            kind: "render_endpoint".to_string(),
        });
    }

    Ok(devices)
}

#[cfg(not(target_os = "windows"))]
fn detect_playback_devices_native() -> Result<Vec<PlaybackDeviceInfo>> {
    Err(anyhow!(
        "Native playback capture is only available on Windows in this build."
    ))
}

#[cfg(target_os = "windows")]
fn probe_playback_device_activity_native(
    device_id: &str,
    timeout_ms: u64,
    min_active_rms: f64,
) -> Result<PlaybackDeviceActivityProbe> {
    let _ = initialize_mta().ok();
    let started = std::time::Instant::now();

    let enumerator = DeviceEnumerator::new().context("create device enumerator")?;
    let device = enumerator
        .get_device(device_id)
        .with_context(|| format!("resolve playback device {device_id}"))?;
    let device_format = device
        .get_device_format()
        .context("read playback device format")?;
    let sample_rate = device_format.get_samplespersec().max(44_100);
    let channels = 2usize;
    let desired_format = WaveFormat::new(
        32,
        32,
        &SampleType::Float,
        sample_rate as usize,
        channels,
        None,
    );
    let frame_bytes = desired_format.get_blockalign() as usize;

    let mut audio_client = device
        .get_iaudioclient()
        .context("activate playback audio client")?;
    let (_default_period, min_period) = audio_client
        .get_device_period()
        .context("read playback device period")?;
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: min_period.max(0),
    };
    audio_client
        .initialize_client(&desired_format, &Direction::Capture, &mode)
        .context("initialize render loopback probe")?;
    let event_handle = audio_client
        .set_get_eventhandle()
        .context("create loopback event handle")?;
    let capture_client = audio_client
        .get_audiocaptureclient()
        .context("create loopback capture client")?;
    let mut sample_queue: VecDeque<u8> = VecDeque::with_capacity(frame_bytes * 4096);

    audio_client.start_stream().context("start loopback probe")?;

    let mut detected = false;
    let mut peak_rms = 0.0f64;
    while started.elapsed() < Duration::from_millis(timeout_ms) {
        let new_frames = capture_client
            .get_next_packet_size()
            .context("read next loopback packet size")?
            .unwrap_or(0);
        if new_frames > 0 {
            let additional = (new_frames as usize * frame_bytes)
                .saturating_sub(sample_queue.capacity().saturating_sub(sample_queue.len()));
            sample_queue.reserve(additional);
            capture_client
                .read_from_device_to_deque(&mut sample_queue)
                .context("read loopback probe packet")?;
        }

        let available_frames = sample_queue.len() / frame_bytes;
        if available_frames > 0 {
            let mut chunk = vec![0u8; available_frames * frame_bytes];
            for byte in &mut chunk {
                *byte = sample_queue.pop_front().unwrap_or(0);
            }

            let mut rms_sum = 0.0f64;
            let mut rms_count = 0usize;
            for frame in 0..available_frames {
                for channel in 0..channels {
                    let offset = frame * frame_bytes + channel * 4;
                    let sample = f32::from_le_bytes([
                        chunk[offset],
                        chunk[offset + 1],
                        chunk[offset + 2],
                        chunk[offset + 3],
                    ]);
                    rms_sum += f64::from(sample * sample);
                    rms_count += 1;
                }
            }

            if rms_count > 0 {
                let rms = (rms_sum / rms_count as f64).sqrt();
                peak_rms = peak_rms.max(rms);
                if rms >= min_active_rms {
                    detected = true;
                    break;
                }
            }
        }

        let _ = event_handle.wait_for_event(200);
    }

    audio_client.stop_stream().ok();
    Ok(PlaybackDeviceActivityProbe {
        device_id: device_id.to_string(),
        detected,
        peak_rms,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

#[cfg(not(target_os = "windows"))]
fn probe_playback_device_activity_native(
    _device_id: &str,
    _timeout_ms: u64,
    _min_active_rms: f64,
) -> Result<PlaybackDeviceActivityProbe> {
    Err(anyhow!(
        "Native playback activity probing is only available on Windows in this build."
    ))
}

#[cfg(target_os = "windows")]
fn capture_playback_loopback_native(
    stdout: &Arc<Mutex<io::Stdout>>,
    capture_id: &str,
    cancel_flag: Arc<AtomicBool>,
    device_id: &str,
    output_path: &Path,
    expected_duration_sec: Option<f64>,
    start_timeout_ms: u64,
    trailing_silence_ms: u64,
    min_active_rms: f64,
) -> Result<PlaybackCaptureResultData> {
    let _ = initialize_mta().ok();
    let capture_start_at = Utc::now().to_rfc3339();
    let start_instant = std::time::Instant::now();

    let finalize_with_cleanup = |writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
                                 success: bool|
     -> Result<()> {
        writer.finalize().context("finalize capture wav")?;
        if !success && output_path.exists() {
            let _ = std::fs::remove_file(output_path);
        }
        Ok(())
    };

    let capture_result = (|| -> Result<PlaybackCaptureResultData> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }

        let enumerator = DeviceEnumerator::new().context("create device enumerator")?;
        let device = enumerator
            .get_device(device_id)
            .with_context(|| format!("resolve playback device {device_id}"))?;
        let device_format = device
            .get_device_format()
            .context("read playback device format")?;
        let sample_rate = device_format.get_samplespersec().max(44_100);
        let channels = 2usize;
        let desired_format = WaveFormat::new(
            32,
            32,
            &SampleType::Float,
            sample_rate as usize,
            channels,
            None,
        );
        let frame_bytes = desired_format.get_blockalign() as usize;

        let mut audio_client = device
            .get_iaudioclient()
            .context("activate playback audio client")?;
        let (_default_period, min_period) = audio_client
            .get_device_period()
            .context("read playback device period")?;
        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns: min_period.max(0),
        };
        audio_client
            .initialize_client(&desired_format, &Direction::Capture, &mode)
            .context("initialize render loopback capture")?;
        let event_handle = audio_client
            .set_get_eventhandle()
            .context("create loopback event handle")?;
        let buffer_frame_count = audio_client
            .get_buffer_size()
            .context("read loopback buffer size")?;
        let capture_client = audio_client
            .get_audiocaptureclient()
            .context("create loopback capture client")?;
        let mut sample_queue: VecDeque<u8> = VecDeque::with_capacity(
            100 * frame_bytes * (1024 + 2 * buffer_frame_count as usize),
        );

        let wav_spec = hound::WavSpec {
            channels: channels as u16,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let writer = hound::WavWriter::create(output_path, wav_spec)
            .with_context(|| format!("create capture wav at {}", output_path.display()))?;
        let mut writer = writer;

        send_playback_capture_progress(
            stdout,
            capture_id,
            "awaiting_audio",
            Some("Waiting for Qobuz playback to begin..."),
            Some(0.0),
            Some(0.0),
            None,
        )?;
        set_playback_capture_job_status(capture_id, "awaiting_audio", None);

        audio_client.start_stream().context("start loopback stream")?;

        let mut audio_detected = false;
        let mut first_active_at: Option<std::time::Instant> = None;
        let mut last_active_at = std::time::Instant::now();
        let mut last_progress_emit = std::time::Instant::now();

        loop {
            if cancel_flag.load(Ordering::SeqCst) {
                audio_client.stop_stream().ok();
                finalize_with_cleanup(writer, false)?;
                return Err(anyhow!("Capture cancelled"));
            }

            let new_frames = capture_client
                .get_next_packet_size()
                .context("read next loopback packet size")?
                .unwrap_or(0);
            if new_frames > 0 {
                let additional = (new_frames as usize * frame_bytes)
                    .saturating_sub(sample_queue.capacity().saturating_sub(sample_queue.len()));
                sample_queue.reserve(additional);
                capture_client
                    .read_from_device_to_deque(&mut sample_queue)
                    .context("read loopback audio packet")?;
            }

            let available_frames = sample_queue.len() / frame_bytes;
            if available_frames > 0 {
                let mut chunk = vec![0u8; available_frames * frame_bytes];
                for byte in &mut chunk {
                    *byte = sample_queue.pop_front().unwrap_or(0);
                }

                let mut rms_sum = 0.0f64;
                let mut rms_count = 0usize;
                let mut pcm_samples = Vec::with_capacity(available_frames * channels);
                for frame in 0..available_frames {
                    for channel in 0..channels {
                        let offset = frame * frame_bytes + channel * 4;
                        let sample = f32::from_le_bytes([
                            chunk[offset],
                            chunk[offset + 1],
                            chunk[offset + 2],
                            chunk[offset + 3],
                        ]);
                        rms_sum += f64::from(sample * sample);
                        rms_count += 1;
                        pcm_samples.push(sample.clamp(-1.0, 1.0));
                    }
                }

                let rms = if rms_count > 0 {
                    (rms_sum / rms_count as f64).sqrt()
                } else {
                    0.0
                };

                if rms >= min_active_rms {
                    last_active_at = std::time::Instant::now();
                    if !audio_detected {
                        audio_detected = true;
                        first_active_at = Some(std::time::Instant::now());
                        send_playback_capture_progress(
                            stdout,
                            capture_id,
                            "capturing",
                            Some("Audio activity detected. Capturing playback..."),
                            Some(0.0),
                            Some(0.0),
                            None,
                        )?;
                        set_playback_capture_job_status(capture_id, "capturing", None);
                    }
                }

                if audio_detected {
                    for sample in pcm_samples {
                        writer
                            .write_sample(sample)
                            .context("write capture wav sample")?;
                    }

                    if last_progress_emit.elapsed() >= Duration::from_millis(700) {
                        let elapsed = first_active_at
                            .map(|started| started.elapsed().as_secs_f64())
                            .unwrap_or(0.0);
                        let progress =
                            expected_duration_sec.map(|expected| (elapsed / expected).clamp(0.0, 1.0));
                        send_playback_capture_progress(
                            stdout,
                            capture_id,
                            "capturing",
                            Some("Recording Qobuz playback to WAV..."),
                            progress,
                            Some(elapsed),
                            None,
                        )?;
                        last_progress_emit = std::time::Instant::now();
                    }

                    let elapsed = first_active_at
                        .map(|started| started.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    if let Some(expected) = expected_duration_sec {
                        if elapsed >= expected + 2.0 {
                            break;
                        }
                    }
                    if elapsed > 2.0
                        && last_active_at.elapsed() >= Duration::from_millis(trailing_silence_ms)
                    {
                        break;
                    }
                } else if start_instant.elapsed() >= Duration::from_millis(start_timeout_ms) {
                    audio_client.stop_stream().ok();
                    finalize_with_cleanup(writer, false)?;
                    return Err(anyhow!(
                        "No audio activity was detected on the selected playback device."
                    ));
                }
            } else if !audio_detected
                && start_instant.elapsed() >= Duration::from_millis(start_timeout_ms)
            {
                audio_client.stop_stream().ok();
                finalize_with_cleanup(writer, false)?;
                return Err(anyhow!(
                    "No audio activity was detected on the selected playback device."
                ));
            }

            if event_handle.wait_for_event(250).is_err()
                && !audio_detected
                && start_instant.elapsed() >= Duration::from_millis(start_timeout_ms)
            {
                audio_client.stop_stream().ok();
                finalize_with_cleanup(writer, false)?;
                return Err(anyhow!(
                    "Playback did not start before the capture timeout elapsed."
                ));
            }
        }

        audio_client.stop_stream().ok();

        if first_active_at.is_none() {
            finalize_with_cleanup(writer, false)?;
            return Err(anyhow!(
                "Playback capture did not produce any WAV samples."
            ));
        }

        send_playback_capture_progress(
            stdout,
            capture_id,
            "saving",
            Some("Finalizing captured WAV..."),
            Some(1.0),
            first_active_at.map(|started| started.elapsed().as_secs_f64()),
            None,
        )?;
        set_playback_capture_job_status(capture_id, "saving", None);

        finalize_with_cleanup(writer, true)?;

        let capture_end_at = Utc::now().to_rfc3339();
        let duration_sec = first_active_at
            .map(|started| started.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        Ok(PlaybackCaptureResultData {
            capture_id: capture_id.to_string(),
            file_path: output_path.to_string_lossy().to_string(),
            capture_sample_rate: sample_rate,
            capture_channels: channels as u16,
            capture_bits_per_sample: 32,
            capture_sample_format: "float32".to_string(),
            capture_start_at,
            capture_end_at,
            duration_sec,
        })
    })();

    capture_result
}

#[cfg(not(target_os = "windows"))]
fn capture_playback_loopback_native(
    _stdout: &Arc<Mutex<io::Stdout>>,
    _capture_id: &str,
    _cancel_flag: Arc<AtomicBool>,
    _device_id: &str,
    _output_path: &Path,
    _expected_duration_sec: Option<f64>,
    _start_timeout_ms: u64,
    _trailing_silence_ms: u64,
    _min_active_rms: f64,
) -> Result<PlaybackCaptureResultData> {
    Err(anyhow!(
        "Native playback capture is only available on Windows in this build."
    ))
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

fn send_quality_progress(
    stdout: &Arc<Mutex<io::Stdout>>,
    stage: &str,
    progress: f64,
    message: &str,
    meta: Option<Value>,
) {
    let mut evt = serde_json::json!({
        "type": "quality_progress",
        "stage": stage,
        "progress": progress,
        "message": message,
        "ts": now_ts_seconds()
    });
    if let Some(m) = meta {
        if let Some(obj) = evt.as_object_mut() {
            obj.insert("meta".to_string(), m);
        }
    }
    send_event(stdout, evt);
}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Object(obj) => {
            let mut keys: Vec<&String> = obj.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                if let Some(v) = obj.get(k) {
                    out.insert(k.clone(), canonicalize_json(v));
                }
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize_json).collect()),
        _ => value.clone(),
    }
}

fn canonical_json_string(value: &Value) -> Result<String> {
    serde_json::to_string(&canonicalize_json(value)).context("serialize canonical json")
}

fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 64];
    let mut total = 0u64;
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        hasher.update(&buf[..n]);
    }
    Ok((format!("{:x}", hasher.finalize()), total))
}

fn parse_string_array(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect(),
        Some(Value::String(s)) if !s.trim().is_empty() => vec![s.trim().to_string()],
        _ => vec![],
    }
}

fn parse_output_items(extra: &Map<String, Value>) -> Vec<(String, String)> {
    // Preferred shape: output_files: { stem: "path.wav", ... }
    if let Some(Value::Object(obj)) = extra
        .get("output_files")
        .or_else(|| extra.get("outputFiles"))
    {
        let mut out: Vec<(String, String)> = obj
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|p| (k.clone(), p.to_string())))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        return out;
    }

    // Alternative: output_paths: ["a.wav", "b.wav"]
    if let Some(Value::Array(arr)) = extra
        .get("output_paths")
        .or_else(|| extra.get("outputPaths"))
    {
        let mut out: Vec<(String, String)> = arr
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                v.as_str()
                    .map(|p| (format!("output_{:03}", i + 1), p.to_string()))
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        return out;
    }

    vec![]
}

fn collect_model_candidate_paths(
    model_obj: &Value,
    models_dir: &Path,
    model_id: &str,
) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(artifacts) = model_obj.get("artifacts").and_then(|v| v.as_object()) {
        if let Some(primary) = artifacts.get("primary").and_then(|v| v.as_object()) {
            if let Some(fname) = primary.get("filename").and_then(|v| v.as_str()) {
                let f = fname.trim();
                if !f.is_empty() && !f.contains(".MISSING") {
                    candidates.push(models_dir.join(f));
                }
            }
        }
        if let Some(cfg_obj) = artifacts.get("config").and_then(|v| v.as_object()) {
            if let Some(fname) = cfg_obj.get("filename").and_then(|v| v.as_str()) {
                let f = fname.trim();
                if !f.is_empty() && !f.contains(".MISSING") {
                    candidates.push(models_dir.join(f));
                }
            }
        }
    }
    if let Some(links) = model_obj.get("links").and_then(|v| v.as_object()) {
        for key in ["checkpoint", "config"] {
            if let Some(u) = links.get(key).and_then(|v| v.as_str()) {
                if let Some(base) = url_basename(u) {
                    candidates.push(models_dir.join(base));
                }
            }
        }
    }
    for ext in [
        ".ckpt",
        ".pth",
        ".pt",
        ".onnx",
        ".safetensors",
        ".yaml",
        ".yml",
    ] {
        candidates.push(models_dir.join(format!("{model_id}{ext}")));
    }
    candidates.sort();
    candidates.dedup();
    candidates
}

fn build_quality_manifest(
    cfg: &BackendConfig,
    extra: &Map<String, Value>,
    stdout: &Arc<Mutex<io::Stdout>>,
) -> Result<Value> {
    let name = extra
        .get("name")
        .or_else(|| extra.get("baseline_name"))
        .or_else(|| extra.get("baselineName"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let model_ids = parse_string_array(
        extra
            .get("model_ids")
            .or_else(|| extra.get("modelIds"))
            .or_else(|| extra.get("models")),
    );
    let config = extra
        .get("config")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    let recipe_id = extra
        .get("recipe_id")
        .or_else(|| extra.get("recipeId"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            config
                .get("recipe_id")
                .or_else(|| config.get("recipeId"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });
    let golden_set_id = extra
        .get("golden_set_id")
        .or_else(|| extra.get("goldenSetId"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            config
                .get("golden_set_id")
                .or_else(|| config.get("goldenSetId"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });
    let metrics = extra.get("metrics").cloned().unwrap_or(Value::Null);
    let output_items = parse_output_items(extra);

    let config_hash = sha256_hex_bytes(canonical_json_string(&config)?.as_bytes());

    send_quality_progress(
        stdout,
        "resolve_models",
        10.0,
        "Resolving model artifacts",
        Some(serde_json::json!({ "model_count": model_ids.len() })),
    );

    let models_json = load_models_with_guide_overrides(cfg).unwrap_or(Value::Null);
    let registry_models = models_json
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut model_rows: Vec<Value> = Vec::new();
    for (idx, model_id) in model_ids.iter().enumerate() {
        let m_obj = registry_models
            .iter()
            .find(|m| m.get("id").and_then(|v| v.as_str()) == Some(model_id.as_str()))
            .cloned()
            .unwrap_or(Value::Null);
        let candidates = collect_model_candidate_paths(&m_obj, &cfg.models_dir, model_id);
        let existing = candidates.into_iter().find(|p| p.exists());
        match existing {
            Some(path) => {
                let (hash, bytes) = sha256_file(&path)?;
                model_rows.push(serde_json::json!({
                    "id": model_id,
                    "found": true,
                    "path": path.to_string_lossy(),
                    "sha256": hash,
                    "bytes": bytes
                }));
            }
            None => {
                model_rows.push(serde_json::json!({
                    "id": model_id,
                    "found": false,
                    "path": Value::Null,
                    "sha256": Value::Null,
                    "bytes": Value::Null
                }));
            }
        }
        let p = if model_ids.is_empty() {
            35.0
        } else {
            10.0 + ((idx + 1) as f64 / model_ids.len() as f64) * 25.0
        };
        send_quality_progress(
            stdout,
            "hash_models",
            p,
            &format!("Processed model {}", model_id),
            None,
        );
    }
    model_rows.sort_by(|a, b| {
        let aid = a.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let bid = b.get("id").and_then(|v| v.as_str()).unwrap_or("");
        aid.cmp(bid)
    });

    let mut output_rows: Vec<Value> = Vec::new();
    let mut output_hashes_obj = Map::new();
    for (idx, (label, out_path)) in output_items.iter().enumerate() {
        let p = PathBuf::from(out_path);
        if p.exists() {
            let (hash, bytes) = sha256_file(&p)?;
            output_hashes_obj.insert(label.clone(), Value::String(hash.clone()));
            output_rows.push(serde_json::json!({
                "label": label,
                "path": out_path,
                "exists": true,
                "sha256": hash,
                "bytes": bytes
            }));
        } else {
            output_rows.push(serde_json::json!({
                "label": label,
                "path": out_path,
                "exists": false,
                "sha256": Value::Null,
                "bytes": Value::Null
            }));
        }
        let p = if output_items.is_empty() {
            70.0
        } else {
            35.0 + ((idx + 1) as f64 / output_items.len() as f64) * 35.0
        };
        send_quality_progress(
            stdout,
            "hash_outputs",
            p,
            &format!("Processed output {}", label),
            None,
        );
    }
    output_rows.sort_by(|a, b| {
        let al = a.get("label").and_then(|v| v.as_str()).unwrap_or("");
        let bl = b.get("label").and_then(|v| v.as_str()).unwrap_or("");
        al.cmp(bl)
    });

    let created_at = now_ts_seconds();
    let mut manifest_obj = Map::new();
    manifest_obj.insert(
        "manifest_version".to_string(),
        Value::String("1.0".to_string()),
    );
    manifest_obj.insert("created_at".to_string(), Value::from(created_at));
    manifest_obj.insert(
        "name".to_string(),
        name.map(Value::String).unwrap_or(Value::Null),
    );
    manifest_obj.insert(
        "model_ids".to_string(),
        Value::Array(model_ids.into_iter().map(Value::String).collect()),
    );
    if let Some(recipe_id) = recipe_id {
        manifest_obj.insert("recipe_id".to_string(), Value::String(recipe_id));
    }
    if let Some(golden_set_id) = golden_set_id {
        manifest_obj.insert("golden_set_id".to_string(), Value::String(golden_set_id));
    }
    manifest_obj.insert("models".to_string(), Value::Array(model_rows));
    manifest_obj.insert("config".to_string(), config);
    manifest_obj.insert("config_hash".to_string(), Value::String(config_hash));
    manifest_obj.insert("outputs".to_string(), Value::Array(output_rows));
    manifest_obj.insert(
        "output_hashes".to_string(),
        Value::Object(output_hashes_obj),
    );
    manifest_obj.insert("metrics".to_string(), metrics);

    let mut deterministic_obj = manifest_obj.clone();
    deterministic_obj.remove("created_at");
    let manifest_hash =
        sha256_hex_bytes(canonical_json_string(&Value::Object(deterministic_obj))?.as_bytes());
    manifest_obj.insert("manifest_hash".to_string(), Value::String(manifest_hash));

    send_quality_progress(stdout, "finalize", 95.0, "Finalizing manifest", None);
    Ok(Value::Object(manifest_obj))
}

fn load_manifest_from_value(value: &Value) -> Result<Value> {
    if let Some(path) = value.as_str() {
        return read_json_file(&PathBuf::from(path))
            .with_context(|| format!("read manifest path {path}"));
    }
    if value.is_object() {
        return Ok(value.clone());
    }
    Err(anyhow!("manifest input must be a path string or object"))
}

fn parse_manifest_input(
    cfg: &BackendConfig,
    extra: &Map<String, Value>,
    stdout: &Arc<Mutex<io::Stdout>>,
    prefix: &str,
) -> Result<Value> {
    // Accepted forms:
    // - "<prefix>_manifest_path": "path.json"
    // - "<prefix>_manifest": { ... } | "path.json"
    // - "<prefix>": { model_ids/config/output_files... } (spec for inline build)
    let key_manifest_path = format!("{prefix}_manifest_path");
    let key_manifest = format!("{prefix}_manifest");
    let key_manifest_path_camel = format!("{prefix}ManifestPath");
    let key_manifest_camel = format!("{prefix}Manifest");
    if let Some(v) = extra
        .get(&key_manifest_path)
        .or_else(|| extra.get(&key_manifest_path_camel))
    {
        return load_manifest_from_value(v);
    }
    if let Some(v) = extra
        .get(&key_manifest)
        .or_else(|| extra.get(&key_manifest_camel))
    {
        return load_manifest_from_value(v);
    }
    if let Some(Value::Object(spec)) = extra.get(prefix) {
        return build_quality_manifest(cfg, spec, stdout);
    }
    // Fallback for candidate: if caller passed top-level fields for candidate creation.
    if prefix == "candidate" {
        return build_quality_manifest(cfg, extra, stdout);
    }
    Err(anyhow!("missing {prefix} manifest input"))
}

fn compare_quality_manifests(baseline: &Value, candidate: &Value) -> Value {
    let mut differences: Vec<Value> = Vec::new();

    let b_cfg = baseline
        .get("config_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let c_cfg = candidate
        .get("config_hash")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if b_cfg != c_cfg {
        differences.push(serde_json::json!({
            "type": "config_hash",
            "baseline": b_cfg,
            "candidate": c_cfg
        }));
    }

    let b_models = baseline
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let c_models = candidate
        .get("models")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let b_model_map: HashMap<String, String> = b_models
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(|v| v.as_str())?;
            let hash = m
                .get("sha256")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Some((id.to_string(), hash))
        })
        .collect();
    let c_model_map: HashMap<String, String> = c_models
        .iter()
        .filter_map(|m| {
            let id = m.get("id").and_then(|v| v.as_str())?;
            let hash = m
                .get("sha256")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Some((id.to_string(), hash))
        })
        .collect();

    let mut all_model_ids: Vec<String> = b_model_map
        .keys()
        .chain(c_model_map.keys())
        .cloned()
        .collect();
    all_model_ids.sort();
    all_model_ids.dedup();
    for id in all_model_ids {
        let b = b_model_map.get(&id).cloned().unwrap_or_default();
        let c = c_model_map.get(&id).cloned().unwrap_or_default();
        if b != c {
            differences.push(serde_json::json!({
                "type": "model_hash",
                "id": id,
                "baseline": b,
                "candidate": c
            }));
        }
    }

    let b_outputs = baseline
        .get("output_hashes")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let c_outputs = candidate
        .get("output_hashes")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let mut labels: Vec<String> = b_outputs.keys().chain(c_outputs.keys()).cloned().collect();
    labels.sort();
    labels.dedup();
    let mut output_mismatch = 0usize;
    for label in labels {
        let b = b_outputs.get(&label).and_then(|v| v.as_str()).unwrap_or("");
        let c = c_outputs.get(&label).and_then(|v| v.as_str()).unwrap_or("");
        if b != c {
            output_mismatch += 1;
            differences.push(serde_json::json!({
                "type": "output_hash",
                "label": label,
                "baseline": b,
                "candidate": c
            }));
        }
    }

    let mut metrics_delta = Map::new();
    let b_metrics = baseline
        .get("metrics")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let c_metrics = candidate
        .get("metrics")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let mut metric_keys: Vec<String> = b_metrics.keys().chain(c_metrics.keys()).cloned().collect();
    metric_keys.sort();
    metric_keys.dedup();
    for key in metric_keys {
        let b = b_metrics.get(&key).and_then(|v| v.as_f64());
        let c = c_metrics.get(&key).and_then(|v| v.as_f64());
        if let (Some(bv), Some(cv)) = (b, c) {
            metrics_delta.insert(key.clone(), Value::from(cv - bv));
        }
    }

    let diff_count = differences.len() as i64;
    let quality_score = (100 - diff_count * 10).max(0);
    serde_json::json!({
        "compatible": differences.is_empty(),
        "quality_score": quality_score,
        "difference_count": differences.len(),
        "output_mismatch_count": output_mismatch,
        "differences": differences,
        "metrics_delta": metrics_delta,
        "baseline_manifest_hash": baseline.get("manifest_hash").cloned().unwrap_or(Value::Null),
        "candidate_manifest_hash": candidate.get("manifest_hash").cloned().unwrap_or(Value::Null),
    })
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

#[derive(Debug, Clone)]
struct AudioQualityThresholds {
    min_correlation: f64,
    min_snr_db: f64,
    min_si_sdr_db: f64,
    max_gain_delta_db: f64,
    max_clipped_samples: usize,
}

fn parse_audio_quality_thresholds(value: Option<&Value>) -> AudioQualityThresholds {
    let obj = value.and_then(|v| v.as_object());
    let get_f64 = |snake: &str, camel: &str, default: f64| {
        obj.and_then(|map| map.get(snake).or_else(|| map.get(camel)))
            .and_then(|v| v.as_f64())
            .unwrap_or(default)
    };
    let get_usize = |snake: &str, camel: &str, default: usize| {
        obj.and_then(|map| map.get(snake).or_else(|| map.get(camel)))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    };

    AudioQualityThresholds {
        min_correlation: get_f64("min_correlation", "minCorrelation", 0.995),
        min_snr_db: get_f64("min_snr_db", "minSnrDb", 24.0),
        min_si_sdr_db: get_f64("min_si_sdr_db", "minSiSdrDb", 24.0),
        max_gain_delta_db: get_f64("max_gain_delta_db", "maxGainDeltaDb", 1.0),
        max_clipped_samples: get_usize("max_clipped_samples", "maxClippedSamples", 0),
    }
}

fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq = samples
        .iter()
        .map(|sample| (*sample as f64) * (*sample as f64))
        .sum::<f64>();
    (sum_sq / samples.len() as f64).sqrt()
}

fn peak_abs(samples: &[f32]) -> f64 {
    samples
        .iter()
        .map(|sample| (*sample as f64).abs())
        .fold(0.0_f64, f64::max)
}

fn clipped_samples(samples: &[f32], threshold: f64) -> usize {
    samples
        .iter()
        .filter(|sample| (**sample as f64).abs() > threshold)
        .count()
}

fn db(value: f64) -> f64 {
    20.0 * value.max(1e-12).log10()
}

fn correlation_stereo(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    if len < 2 {
        return 0.0;
    }

    let mut corrs: Vec<f64> = Vec::new();
    for channel in 0..2 {
        let mut a_vals: Vec<f64> = Vec::new();
        let mut b_vals: Vec<f64> = Vec::new();
        let mut idx = channel;
        while idx < len {
            a_vals.push(a[idx] as f64);
            b_vals.push(b[idx] as f64);
            idx += 2;
        }
        if a_vals.is_empty() || b_vals.is_empty() {
            continue;
        }
        let mean_a = a_vals.iter().sum::<f64>() / a_vals.len() as f64;
        let mean_b = b_vals.iter().sum::<f64>() / b_vals.len() as f64;
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for (va, vb) in a_vals.iter().zip(b_vals.iter()) {
            let ca = *va - mean_a;
            let cb = *vb - mean_b;
            dot += ca * cb;
            norm_a += ca * ca;
            norm_b += cb * cb;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom > 0.0 {
            corrs.push(dot / denom);
        }
    }

    if corrs.is_empty() {
        0.0
    } else {
        corrs.iter().sum::<f64>() / corrs.len() as f64
    }
}

fn snr_db(reference: &[f32], estimate: &[f32]) -> f64 {
    let len = reference.len().min(estimate.len());
    if len == 0 {
        return 0.0;
    }
    let mut signal = 0.0_f64;
    let mut error = 0.0_f64;
    for idx in 0..len {
        let ref_v = reference[idx] as f64;
        let err_v = estimate[idx] as f64 - ref_v;
        signal += ref_v * ref_v;
        error += err_v * err_v;
    }
    if error <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal.max(1e-20) / error.max(1e-20)).log10()
}

fn si_sdr_db(reference: &[f32], estimate: &[f32]) -> f64 {
    let len = reference.len().min(estimate.len());
    if len < 2 {
        return 0.0;
    }

    let mut values: Vec<f64> = Vec::new();
    for channel in 0..2 {
        let mut ref_vals: Vec<f64> = Vec::new();
        let mut est_vals: Vec<f64> = Vec::new();
        let mut idx = channel;
        while idx < len {
            ref_vals.push(reference[idx] as f64);
            est_vals.push(estimate[idx] as f64);
            idx += 2;
        }
        if ref_vals.is_empty() || est_vals.is_empty() {
            continue;
        }
        let s_energy = ref_vals.iter().map(|value| value * value).sum::<f64>();
        if s_energy <= 0.0 {
            continue;
        }
        let dot = ref_vals
            .iter()
            .zip(est_vals.iter())
            .map(|(ref_v, est_v)| ref_v * est_v)
            .sum::<f64>();
        let alpha = dot / s_energy;
        let mut target = 0.0_f64;
        let mut noise = 0.0_f64;
        for (ref_v, est_v) in ref_vals.iter().zip(est_vals.iter()) {
            let target_v = alpha * *ref_v;
            let noise_v = *est_v - target_v;
            target += target_v * target_v;
            noise += noise_v * noise_v;
        }
        if noise <= 0.0 {
            values.push(f64::INFINITY);
        } else {
            values.push(10.0 * (target.max(1e-20) / noise.max(1e-20)).log10());
        }
    }

    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn finite_or_null(value: f64) -> Value {
    if value.is_finite() {
        Value::from(value)
    } else {
        Value::Null
    }
}

fn manifest_output_paths(manifest: &Value) -> HashMap<String, String> {
    manifest
        .get("outputs")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| {
            let label = item.get("label").and_then(|v| v.as_str())?;
            let path = item.get("path").and_then(|v| v.as_str())?;
            let exists = item.get("exists").and_then(|v| v.as_bool()).unwrap_or(false);
            if !exists {
                return None;
            }
            Some((label.to_string(), path.to_string()))
        })
        .collect()
}

fn build_audio_quality_report(
    baseline: &Value,
    candidate: &Value,
    thresholds: &AudioQualityThresholds,
) -> Value {
    let baseline_outputs = manifest_output_paths(baseline);
    let candidate_outputs = manifest_output_paths(candidate);

    let mut labels: Vec<String> = baseline_outputs
        .keys()
        .chain(candidate_outputs.keys())
        .cloned()
        .collect();
    labels.sort();
    labels.dedup();

    let mut pairs: Vec<Value> = Vec::new();
    let mut compared_pairs = 0usize;
    let mut failed_pairs = 0usize;
    let mut skipped_pairs = 0usize;

    for label in labels {
        let Some(reference_path) = baseline_outputs.get(&label) else {
            skipped_pairs += 1;
            pairs.push(serde_json::json!({
                "label": label,
                "verdict": "skipped",
                "reason": "missing baseline output",
            }));
            continue;
        };
        let Some(candidate_path) = candidate_outputs.get(&label) else {
            failed_pairs += 1;
            pairs.push(serde_json::json!({
                "label": label,
                "reference_path": reference_path,
                "verdict": "fail",
                "reason": "missing candidate output",
            }));
            continue;
        };
        if Path::new(reference_path)
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            != "wav"
            || Path::new(candidate_path)
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                != "wav"
        {
            skipped_pairs += 1;
            pairs.push(serde_json::json!({
                "label": label,
                "reference_path": reference_path,
                "candidate_path": candidate_path,
                "verdict": "skipped",
                "reason": "audio QA currently supports WAV-only pairwise regression",
            }));
            continue;
        }

        let pair_result = (|| -> Result<Value> {
            let (reference_sr, reference_audio) = read_wav_to_f32_stereo(Path::new(reference_path))?;
            let (candidate_sr, candidate_audio) = read_wav_to_f32_stereo(Path::new(candidate_path))?;
            if reference_sr != candidate_sr {
                return Ok(serde_json::json!({
                    "label": label,
                    "reference_path": reference_path,
                    "candidate_path": candidate_path,
                    "verdict": "skipped",
                    "reason": format!("sample rate mismatch: {} != {}", reference_sr, candidate_sr),
                }));
            }
            let len = reference_audio.len().min(candidate_audio.len());
            let reference_audio = &reference_audio[..len];
            let candidate_audio = &candidate_audio[..len];
            if len == 0 {
                return Ok(serde_json::json!({
                    "label": label,
                    "reference_path": reference_path,
                    "candidate_path": candidate_path,
                    "verdict": "skipped",
                    "reason": "empty audio pair",
                }));
            }

            let correlation = correlation_stereo(reference_audio, candidate_audio);
            let snr = snr_db(reference_audio, candidate_audio);
            let si_sdr = si_sdr_db(reference_audio, candidate_audio);
            let gain_delta_db = db(rms(candidate_audio)) - db(rms(reference_audio));
            let candidate_peak = peak_abs(candidate_audio);
            let candidate_clipped = clipped_samples(candidate_audio, 1.0);
            let duration_seconds = (len as f64 / 2.0) / reference_sr as f64;
            let mut regressions: Vec<String> = Vec::new();

            if correlation < thresholds.min_correlation {
                regressions.push(format!(
                    "correlation {:.6} < {:.6}",
                    correlation, thresholds.min_correlation
                ));
            }
            if snr.is_finite() && snr < thresholds.min_snr_db {
                regressions.push(format!("snr_db {:.2} < {:.2}", snr, thresholds.min_snr_db));
            }
            if si_sdr.is_finite() && si_sdr < thresholds.min_si_sdr_db {
                regressions.push(format!(
                    "si_sdr_db {:.2} < {:.2}",
                    si_sdr, thresholds.min_si_sdr_db
                ));
            }
            if gain_delta_db.abs() > thresholds.max_gain_delta_db {
                regressions.push(format!(
                    "abs(gain_delta_db) {:.2} > {:.2}",
                    gain_delta_db.abs(),
                    thresholds.max_gain_delta_db
                ));
            }
            if candidate_clipped > thresholds.max_clipped_samples {
                regressions.push(format!(
                    "clipped_samples {} > {}",
                    candidate_clipped, thresholds.max_clipped_samples
                ));
            }

            Ok(serde_json::json!({
                "label": label,
                "reference_path": reference_path,
                "candidate_path": candidate_path,
                "sample_rate": reference_sr,
                "duration_seconds": duration_seconds,
                "correlation": finite_or_null(correlation),
                "snr_db": finite_or_null(snr),
                "si_sdr_db": finite_or_null(si_sdr),
                "gain_delta_db": finite_or_null(gain_delta_db),
                "candidate_peak_abs": finite_or_null(candidate_peak),
                "candidate_clipped_samples": candidate_clipped,
                "verdict": if regressions.is_empty() { "pass" } else { "fail" },
                "regressions": regressions,
            }))
        })();

        match pair_result {
            Ok(result) => {
                let verdict = result.get("verdict").and_then(|v| v.as_str()).unwrap_or("skipped");
                match verdict {
                    "pass" => compared_pairs += 1,
                    "fail" => {
                        compared_pairs += 1;
                        failed_pairs += 1;
                    }
                    _ => skipped_pairs += 1,
                }
                pairs.push(result);
            }
            Err(err) => {
                failed_pairs += 1;
                pairs.push(serde_json::json!({
                    "label": label,
                    "reference_path": reference_path,
                    "candidate_path": candidate_path,
                    "verdict": "fail",
                    "reason": err.to_string(),
                }));
            }
        }
    }

    serde_json::json!({
        "schema_version": "1.0",
        "mode": "pairwise_output_regression",
        "thresholds": {
            "min_correlation": thresholds.min_correlation,
            "min_snr_db": thresholds.min_snr_db,
            "min_si_sdr_db": thresholds.min_si_sdr_db,
            "max_gain_delta_db": thresholds.max_gain_delta_db,
            "max_clipped_samples": thresholds.max_clipped_samples,
        },
        "summary": {
            "compared_pairs": compared_pairs,
            "failed_pairs": failed_pairs,
            "skipped_pairs": skipped_pairs,
        },
        "verdict": if failed_pairs == 0 && compared_pairs > 0 { "pass" } else if compared_pairs == 0 { "skipped" } else { "fail" },
        "pairs": pairs,
    })
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

fn send_playback_capture_progress(
    stdout: &Arc<Mutex<io::Stdout>>,
    capture_id: &str,
    status: &str,
    detail: Option<&str>,
    progress: Option<f64>,
    elapsed_sec: Option<f64>,
    error: Option<&str>,
) -> Result<()> {
    let mut payload = serde_json::json!({
        "type": "playback_capture_progress",
        "capture_id": capture_id,
        "status": status,
    });

    if let Some(detail) = detail {
        payload["detail"] = Value::String(detail.to_string());
    }
    if let Some(progress) = progress {
        payload["progress"] = Value::from(progress);
    }
    if let Some(elapsed_sec) = elapsed_sec {
        payload["elapsed_sec"] = Value::from(elapsed_sec);
    }
    if let Some(error) = error {
        payload["error"] = Value::String(error.to_string());
    }

    write_json_line_locked(stdout, &payload)
}

fn resolve_youtube_native(url: &str, stdout: &Arc<Mutex<io::Stdout>>) -> Result<Value> {
    if url.trim().is_empty() {
        return Err(anyhow!("url is required"));
    }

    let python = locate_python_exe();
    let out_dir = std::env::temp_dir().join("stemsep_youtube");
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create youtube temp dir: {}", out_dir.display()))?;

    send_event(
        stdout,
        serde_json::json!({
            "type": "youtube_progress",
            "status": "starting"
        }),
    );

    let script = r#"import glob
import json
import os
import sys
import uuid

url = sys.argv[1]
out_dir = sys.argv[2]

try:
    import yt_dlp
except Exception as e:
    sys.stderr.write("yt-dlp import failed: " + repr(e) + "\n")
    sys.exit(2)

os.makedirs(out_dir, exist_ok=True)
uid = uuid.uuid4().hex
template = os.path.join(out_dir, f"stemsep_youtube_{uid}.%(ext)s")

opts = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "outtmpl": template,
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "0",
    }],
}

title = "YouTube Audio"
channel = None
duration = None
thumbnail = None
canonical_url = url
prepared = ""
with yt_dlp.YoutubeDL(opts) as ydl:
    info = ydl.extract_info(url, download=True)
    title = info.get("title") or title
    channel = info.get("uploader") or info.get("channel") or info.get("creator")
    duration = info.get("duration")
    thumbnail = info.get("thumbnail")
    canonical_url = info.get("webpage_url") or info.get("original_url") or url
    prepared = ydl.prepare_filename(info)

root, _ = os.path.splitext(prepared)
candidate = root + ".wav"
if not os.path.exists(candidate):
    matches = sorted(glob.glob(os.path.join(out_dir, f"stemsep_youtube_{uid}*.wav")))
    if matches:
        candidate = matches[-1]

if not os.path.exists(candidate):
    sys.stderr.write("Could not locate converted WAV output\n")
    sys.exit(3)

print(json.dumps({
    "file_path": candidate,
    "title": title,
    "source_url": url,
    "channel": channel,
    "duration_sec": duration,
    "thumbnail_url": thumbnail,
    "canonical_url": canonical_url,
}))
"#;

    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(url)
        .arg(out_dir.to_string_lossy().to_string())
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .output()
        .with_context(|| format!("spawn python for resolve_youtube (python={python})"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout_txt = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let msg = if !stderr.is_empty() {
            let from_error_line = stderr
                .lines()
                .map(|l| l.trim())
                .find(|l| l.starts_with("ERROR:"))
                .map(|l| l.to_string());
            if let Some(clean) = from_error_line {
                clean
            } else {
                stderr
                    .lines()
                    .map(|l| l.trim())
                    .find(|l| !l.is_empty())
                    .unwrap_or("resolve_youtube failed")
                    .to_string()
            }
        } else if !stdout_txt.is_empty() {
            stdout_txt
        } else {
            format!(
                "resolve_youtube failed with exit code {}",
                output.status.code().unwrap_or(-1)
            )
        };
        return Err(anyhow!(msg));
    }

    let stdout_txt = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let json_line = stdout_txt
        .lines()
        .last()
        .map(|s| s.trim())
        .unwrap_or_default();
    if json_line.is_empty() {
        return Err(anyhow!("resolve_youtube returned empty output"));
    }

    let payload: Value =
        serde_json::from_str(json_line).context("parse resolve_youtube output JSON")?;
    let file_path = payload
        .get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("resolve_youtube output missing file_path"))?;
    if !PathBuf::from(file_path).exists() {
        return Err(anyhow!("Downloaded file not found: {file_path}"));
    }

    send_event(
        stdout,
        serde_json::json!({
            "type": "youtube_progress",
            "status": "complete",
            "percent": "100%"
        }),
    );

    Ok(payload)
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

fn worker_config_dir(cfg: &BackendConfig) -> PathBuf {
    cfg.catalog_cache_dir.join("worker-configs")
}

fn write_worker_config_file(cfg: &BackendConfig, job_id: &str, config: &Value) -> Result<PathBuf> {
    let dir = worker_config_dir(cfg);
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let file_name = format!("selection-job-{}-{}.json", job_id, Uuid::new_v4());
    let path = dir.join(file_name);
    let payload = serde_json::to_vec_pretty(config).context("serialize job config")?;
    std::fs::write(&path, payload).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

fn remove_worker_config_file(path: Option<&Path>) {
    if let Some(path) = path {
        let _ = std::fs::remove_file(path);
    }
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SelectionJobSnapshot {
    job_id: String,
    status: String,
    requested_at: String,
    started_at: Option<String>,
    finished_at: Option<String>,
    progress: Option<f64>,
    message: Option<String>,
    error: Option<String>,
    model_id: Option<String>,
    selection_type: Option<String>,
    selection_id: Option<String>,
    file_path: Option<String>,
    output_dir: Option<String>,
    output_files: Option<Value>,
}

struct SeparationJob {
    process: Child,
    state: Arc<Mutex<SelectionJobSnapshot>>,
    signature: String,
    config_path: Option<PathBuf>,
}

struct PendingSelectionJob {
    job_id: String,
    config: Value,
    cfg: BackendConfig,
    stdout: Arc<Mutex<io::Stdout>>,
    state: Arc<Mutex<SelectionJobSnapshot>>,
    signature: String,
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
    pending_jobs: Arc<Mutex<VecDeque<PendingSelectionJob>>>,
    finished_jobs: Arc<Mutex<VecDeque<SelectionJobSnapshot>>>,
}

impl SeparationManager {
    fn build_snapshot(job_id: &str, config: &Value) -> SelectionJobSnapshot {
        let obj = config.as_object();
        let string_field = |key: &str, fallback: &str| -> Option<String> {
            obj.and_then(|map| {
                map.get(key)
                    .or_else(|| map.get(fallback))
                    .and_then(|v| v.as_str())
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            })
        };

        SelectionJobSnapshot {
            job_id: job_id.to_string(),
            status: "requested".to_string(),
            requested_at: Utc::now().to_rfc3339(),
            started_at: None,
            finished_at: None,
            progress: Some(0.0),
            message: Some("Selection job requested.".to_string()),
            error: None,
            model_id: string_field("model_id", "modelId"),
            selection_type: string_field("selection_type", "selectionType"),
            selection_id: string_field("selection_id", "selectionId"),
            file_path: string_field("file_path", "filePath"),
            output_dir: string_field("output_dir", "outputDir"),
            output_files: None,
        }
    }

    fn with_active_state<F>(&self, job_id: &str, mutator: F)
    where
        F: FnOnce(&mut SelectionJobSnapshot),
    {
        let state_arc = {
            let map = self.jobs.lock().expect("jobs lock");
            map.get(job_id).map(|job| job.state.clone())
        };
        if let Some(state) = state_arc {
            if let Ok(mut guard) = state.lock() {
                mutator(&mut guard);
            }
        }
    }

    fn store_finished_snapshot(&self, snapshot: SelectionJobSnapshot) {
        let mut finished = self.finished_jobs.lock().expect("finished_jobs lock");
        if let Some(existing) = finished.iter().position(|entry| entry.job_id == snapshot.job_id) {
            finished.remove(existing);
        }
        finished.push_front(snapshot);
        while finished.len() > 200 {
            finished.pop_back();
        }
    }

    fn max_concurrent_jobs(&self) -> usize {
        1
    }

    fn build_signature(config: &Value) -> String {
        let obj = config.as_object();
        let field = |key: &str, fallback: &str| -> String {
            obj.and_then(|map| {
                map.get(key)
                    .or_else(|| map.get(fallback))
                    .and_then(|v| v.as_str())
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            })
            .unwrap_or_default()
        };
        format!(
            "{}|{}|{}|{}|{}",
            field("selection_type", "selectionType"),
            field("selection_id", "selectionId"),
            field("model_id", "modelId"),
            field("file_path", "filePath"),
            field("output_dir", "outputDir")
        )
    }

    fn find_existing_by_signature(&self, signature: &str) -> Option<SelectionJobSnapshot> {
        if signature.trim().is_empty() {
            return None;
        }
        if let Some(snapshot) = {
            let jobs = self.jobs.lock().expect("jobs lock");
            jobs.values()
                .find(|job| job.signature == signature)
                .and_then(|job| job.state.lock().ok().map(|state| state.clone()))
        } {
            return Some(snapshot);
        }
        let pending = self.pending_jobs.lock().expect("pending_jobs lock");
        pending
            .iter()
            .find(|job| job.signature == signature)
            .and_then(|job| job.state.lock().ok().map(|state| state.clone()))
    }

    fn snapshot(&self, job_id: &str) -> Option<SelectionJobSnapshot> {
        if let Some(snapshot) = {
            let map = self.jobs.lock().expect("jobs lock");
            map.get(job_id)
                .and_then(|job| job.state.lock().ok().map(|state| state.clone()))
        } {
            return Some(snapshot);
        }
        if let Some(snapshot) = {
            let pending = self.pending_jobs.lock().expect("pending_jobs lock");
            pending
                .iter()
                .find(|job| job.job_id == job_id)
                .and_then(|job| job.state.lock().ok().map(|state| state.clone()))
        } {
            return Some(snapshot);
        }
        self.finished_jobs
            .lock()
            .expect("finished_jobs lock")
            .iter()
            .find(|entry| entry.job_id == job_id)
            .cloned()
    }

    fn list(&self) -> Vec<SelectionJobSnapshot> {
        let mut snapshots: Vec<SelectionJobSnapshot> = {
            let map = self.jobs.lock().expect("jobs lock");
            map.values()
                .filter_map(|job| job.state.lock().ok().map(|state| state.clone()))
                .collect()
        };
        let pending = self.pending_jobs.lock().expect("pending_jobs lock");
        snapshots.extend(
            pending
                .iter()
                .filter_map(|job| job.state.lock().ok().map(|state| state.clone())),
        );
        let finished = self.finished_jobs.lock().expect("finished_jobs lock");
        snapshots.extend(finished.iter().cloned());
        snapshots.sort_by(|a, b| b.requested_at.cmp(&a.requested_at));
        snapshots
    }

    fn discard(&self, job_id: &str) -> Result<SelectionJobSnapshot> {
        if let Some(snapshot) = {
            let map = self.jobs.lock().expect("jobs lock");
            map.get(job_id)
                .and_then(|job| job.state.lock().ok().map(|state| state.clone()))
        } {
            if matches!(snapshot.status.as_str(), "starting" | "running" | "cancelling") {
                return Err(anyhow!(
                    "Selection job {job_id} is still active and cannot be discarded"
                ));
            }
        }

        let mut finished_jobs = self.finished_jobs.lock().expect("finished_jobs lock");
        let position = finished_jobs
            .iter()
            .position(|snapshot| snapshot.job_id == job_id)
            .ok_or_else(|| anyhow!("Selection job not found: {job_id}"))?;
        let snapshot = finished_jobs
            .remove(position)
            .ok_or_else(|| anyhow!("Selection job not found: {job_id}"))?;

        if let Some(output_files) = snapshot.output_files.as_ref().and_then(|v| v.as_object()) {
            for value in output_files.values() {
                if let Some(path_str) = value.as_str() {
                    let path = PathBuf::from(path_str);
                    if path.is_file() {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }

        Ok(snapshot)
    }

    fn start_next_pending(&self) {
        loop {
            let active_jobs = self.jobs.lock().expect("jobs lock").len();
            if active_jobs >= self.max_concurrent_jobs() {
                return;
            }
            let queued = {
                let mut pending = self.pending_jobs.lock().expect("pending_jobs lock");
                pending.pop_front()
            };

            let Some(queued) = queued else {
                return;
            };

            let PendingSelectionJob {
                job_id,
                config,
                cfg,
                stdout,
                state,
                signature,
            } = queued;

            if let Ok(mut state) = state.lock() {
                state.status = "starting".to_string();
                state.message = Some("Starting queued separation worker...".to_string());
                state.error = None;
            }

            let stdout_for_error = stdout.clone();
            let state_for_error = state.clone();

            match self.spawn_job_process(job_id.clone(), config, &cfg, stdout, state, signature) {
                Ok(()) => return,
                Err(error) => {
                    let now = Utc::now().to_rfc3339();
                    if let Ok(mut state) = state_for_error.lock() {
                        state.status = "failed".to_string();
                        state.finished_at = Some(now);
                        state.message = Some("Failed to start queued worker.".to_string());
                        state.error = Some(format!("{error:#}"));
                        self.store_finished_snapshot(state.clone());
                        send_event(
                            &stdout_for_error,
                            serde_json::json!({
                                "type": "separation_error",
                                "job_id": job_id,
                                "error": format!("{error:#}"),
                                "message": "Failed to start queued worker."
                            }),
                        );
                    }
                    continue;
                }
            }
        }
    }

    fn update_from_event(&self, job_id: &str, event: &Value) {
        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
        self.with_active_state(job_id, |state| {
            match event_type {
                "separation_started" => {
                    state.status = "running".to_string();
                    if state.started_at.is_none() {
                        state.started_at = Some(Utc::now().to_rfc3339());
                    }
                    state.message = Some("Separation started.".to_string());
                    state.error = None;
                }
                "separation_progress" => {
                    state.status = "running".to_string();
                    if state.started_at.is_none() {
                        state.started_at = Some(Utc::now().to_rfc3339());
                    }
                    state.progress = event
                        .get("progress")
                        .and_then(|v| v.as_f64())
                        .map(|v| v.clamp(0.0, 100.0));
                    state.message = event
                        .get("message")
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_string());
                }
                "separation_complete" => {
                    state.status = "completed".to_string();
                    state.finished_at = Some(Utc::now().to_rfc3339());
                    state.progress = Some(100.0);
                    state.message = Some("Separation completed.".to_string());
                    state.output_files = event.get("output_files").cloned();
                    state.error = None;
                }
                "separation_error" => {
                    state.status = "failed".to_string();
                    state.finished_at = Some(Utc::now().to_rfc3339());
                    state.message = event
                        .get("message")
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_string())
                        .or_else(|| Some("Separation failed.".to_string()));
                    state.error = event
                        .get("error")
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_string());
                }
                "separation_cancelled" => {
                    state.status = "cancelled".to_string();
                    state.finished_at = Some(Utc::now().to_rfc3339());
                    state.message = Some("Separation cancelled.".to_string());
                    state.error = None;
                }
                _ => {}
            }
        });
    }

    fn submit(
        &self,
        job_id: String,
        config: Value,
        cfg: &BackendConfig,
        stdout: Arc<Mutex<io::Stdout>>,
    ) -> Result<SelectionJobSnapshot> {
        let signature = Self::build_signature(&config);
        if let Some(existing) = self.find_existing_by_signature(&signature) {
            return Ok(existing);
        }

        let state = Arc::new(Mutex::new(Self::build_snapshot(&job_id, &config)));
        let snapshot = state.lock().ok().map(|guard| guard.clone()).unwrap_or_else(|| {
            Self::build_snapshot(&job_id, &config)
        });

        let active_jobs = self.jobs.lock().expect("jobs lock").len();
        if active_jobs >= self.max_concurrent_jobs() {
            if let Ok(mut guard) = state.lock() {
                guard.status = "pending".to_string();
                guard.message = Some("Queued for separation.".to_string());
                guard.error = None;
            }
            self.pending_jobs
                .lock()
                .expect("pending_jobs lock")
                .push_back(PendingSelectionJob {
                    job_id,
                    config,
                    cfg: cfg.clone(),
                    stdout,
                    state,
                    signature,
                });
            return Ok(self
                .pending_jobs
                .lock()
                .expect("pending_jobs lock")
                .back()
                .and_then(|job| job.state.lock().ok().map(|state| state.clone()))
                .unwrap_or(snapshot));
        }

        if let Ok(mut guard) = state.lock() {
            guard.status = "starting".to_string();
            guard.message = Some("Starting separation worker...".to_string());
            guard.error = None;
        }
        self.spawn_job_process(job_id.clone(), config, cfg, stdout, state.clone(), signature)?;
        Ok(state.lock().ok().map(|state| state.clone()).unwrap_or(snapshot))
    }

    fn spawn_job_process(
        &self,
        job_id: String,
        config: Value,
        cfg: &BackendConfig,
        stdout: Arc<Mutex<io::Stdout>>,
        state: Arc<Mutex<SelectionJobSnapshot>>,
        signature: String,
    ) -> Result<()> {
        let script = locate_inference_script()
            .ok_or_else(|| anyhow!("inference.py not found (set STEMSEP_INFERENCE_SCRIPT)"))?;
        let python = locate_python_exe();
        let config_path = write_worker_config_file(cfg, &job_id, &config)?;

        let spawn_result = Command::new(python)
            .arg("-u")
            .arg(script)
            .arg("--config-file")
            .arg(&config_path)
            .env("STEMSEP_MODELS_DIR", &cfg.models_dir) // Ensure subprocess sees correct models dir
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped()) // Not really used but good practice
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match spawn_result {
            Ok(child) => child,
            Err(error) => {
                remove_worker_config_file(Some(&config_path));
                return Err(error).context("spawn inference process");
            }
        };

        let child_stdout = child
            .stdout
            .take()
            .ok_or_else(|| {
                remove_worker_config_file(Some(&config_path));
                anyhow!("inference stdout missing")
            })?;
        let child_stderr = child
            .stderr
            .take()
            .ok_or_else(|| {
                remove_worker_config_file(Some(&config_path));
                anyhow!("inference stderr missing")
            })?;

        // Store job process for cancellation and state inspection
        {
            let mut map = self.jobs.lock().expect("jobs lock");
            map.insert(
                job_id.clone(),
                SeparationJob {
                    process: child,
                    state: state.clone(),
                    signature,
                    config_path: Some(config_path),
                },
            );
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
                    mgr.update_from_event(&job_id_clone, &event);
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
        let pending_cancelled = {
            let mut pending = self.pending_jobs.lock().expect("pending_jobs lock");
            pending
                .iter()
                .position(|job| job.job_id == job_id)
                .and_then(|index| pending.remove(index))
        };
        if let Some(pending_job) = pending_cancelled {
            if let Ok(mut state) = pending_job.state.lock() {
                state.status = "cancelled".to_string();
                state.finished_at = Some(Utc::now().to_rfc3339());
                state.message = Some("Queued separation cancelled.".to_string());
                state.error = None;
                self.store_finished_snapshot(state.clone());
            }
            send_event(
                &pending_job.stdout,
                serde_json::json!({
                    "type": "separation_cancelled",
                    "job_id": job_id,
                    "message": "Queued separation cancelled."
                }),
            );
            return true;
        }

        let mut map = self.jobs.lock().expect("jobs lock");
        if let Some(job) = map.get_mut(job_id) {
            if let Ok(mut state) = job.state.lock() {
                state.status = "cancelling".to_string();
                state.message = Some("Cancellation requested.".to_string());
                state.error = None;
            }
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
        if let Some(job) = map.remove(job_id) {
            let config_path = job.config_path.clone();
            if let Ok(mut state) = job.state.lock() {
                if state.finished_at.is_none()
                    && matches!(state.status.as_str(), "running" | "starting" | "cancelling")
                {
                    if state.status == "cancelling" {
                        state.status = "cancelled".to_string();
                        state.message = Some("Separation cancelled.".to_string());
                    } else {
                        state.status = "failed".to_string();
                        state.message = Some("Worker exited before completion.".to_string());
                        if state.error.is_none() {
                            state.error = Some("Worker exited before completion.".to_string());
                        }
                    }
                    state.finished_at = Some(Utc::now().to_rfc3339());
                }
                self.store_finished_snapshot(state.clone());
            }
            remove_worker_config_file(config_path.as_deref());
        }
        drop(map);
        self.start_next_pending();
    }
}

fn emit_recipe_plan_event_if_needed(
    cfg: &BackendConfig,
    stdout: &Arc<Mutex<io::Stdout>>,
    extra: &Map<String, Value>,
    job_id: &str,
) {
    if let Some(model_id) = extra.get("model_id").and_then(|v| v.as_str()) {
        let recipes_json = read_json_file(&cfg.assets_dir.join("recipes.json")).unwrap_or(Value::Null);
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
                stdout,
                serde_json::json!({
                    "type": "recipe_plan",
                    "job_id": job_id,
                    "recipe": Value::Object(plan)
                }),
            );
        }
    }
}

fn build_selection_job_config(
    cfg: &BackendConfig,
    extra: &Map<String, Value>,
    gpu_info: &Value,
    job_id: &str,
) -> Value {
    let mut config_obj = Value::Object(extra.clone());
    if let Some(obj) = config_obj.as_object_mut() {
        obj.insert(
            "models_dir".to_string(),
            Value::String(cfg.models_dir.to_string_lossy().to_string()),
        );
        obj.insert("job_id".to_string(), Value::String(job_id.to_string()));
        let runtime_catalog = load_catalog_runtime(cfg).ok();
        let (selection_type, selection_id) = selection_descriptor_from_extra(extra);
        if let (Some(runtime_catalog), Some(selection_type), Some(selection_id)) =
            (runtime_catalog.as_ref(), selection_type, selection_id)
        {
            if let Ok(execution_plan) =
                catalog_execution_plan(cfg, runtime_catalog, &selection_type, &selection_id)
            {
                obj.insert(
                    "selection_type".to_string(),
                    Value::String(selection_type.clone()),
                );
                obj.insert(
                    "selection_id".to_string(),
                    Value::String(selection_id.clone()),
                );
                obj.insert(
                    "selection_envelope".to_string(),
                    execution_plan
                        .get("selection_envelope")
                        .cloned()
                        .unwrap_or(Value::Null),
                );
                obj.insert(
                    "resolved_bundle".to_string(),
                    execution_plan
                        .get("resolved_bundle")
                        .cloned()
                        .unwrap_or(Value::Null),
                );
                obj.insert("execution_plan".to_string(), execution_plan);
                let resolved_step = obj
                    .get("execution_plan")
                    .and_then(|v| v.get("steps"))
                    .and_then(|v| v.as_array())
                    .and_then(|steps| steps.iter().find(|step| step.is_object()))
                    .cloned()
                    .unwrap_or(Value::Null);
                obj.insert("resolved_step".to_string(), resolved_step);
            }
        }
        obj.insert(
            "worker_protocol".to_string(),
            serde_json::json!({
                "kind": "selection_step",
                "version": 1
            }),
        );

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
        let mut resolved = if incoming.is_empty() || incoming.eq_ignore_ascii_case("auto") {
            default_device.to_string()
        } else {
            incoming
        };

        if resolved.eq_ignore_ascii_case("cuda") {
            resolved = "cuda:0".to_string();
        }

        obj.insert("device".to_string(), Value::String(resolved));
    }
    config_obj
}

fn export_selection_job_outputs(
    separation_manager: &SeparationManager,
    job_id: &str,
    export_dir: &Path,
) -> Result<Value> {
    let snapshot = separation_manager
        .snapshot(job_id)
        .ok_or_else(|| anyhow!("Selection job not found: {job_id}"))?;
    if snapshot.status != "completed" {
        return Err(anyhow!(
            "Selection job {job_id} is not completed (current status: {})",
            snapshot.status
        ));
    }
    let output_files = snapshot
        .output_files
        .as_ref()
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("Selection job {job_id} has no output files to export"))?;

    std::fs::create_dir_all(export_dir)
        .with_context(|| format!("create export dir {}", export_dir.display()))?;

    let mut exported = Map::new();
    for (stem, value) in output_files {
        let source = value
            .as_str()
            .ok_or_else(|| anyhow!("Output file entry for {stem} is not a string path"))?;
        let source_path = PathBuf::from(source);
        if !source_path.is_file() {
            return Err(anyhow!(
                "Output file for {stem} does not exist: {}",
                source_path.display()
            ));
        }
        let file_name = source_path.file_name().ok_or_else(|| {
            anyhow!(
                "Invalid output filename for {stem}: {}",
                source_path.display()
            )
        })?;
        let destination = export_dir.join(file_name);
        std::fs::copy(&source_path, &destination).with_context(|| {
            format!(
                "copy exported output {} -> {}",
                source_path.display(),
                destination.display()
            )
        })?;
        exported.insert(
            stem.clone(),
            Value::String(destination.to_string_lossy().to_string()),
        );
    }

    Ok(serde_json::json!({
        "job_id": job_id,
        "export_path": export_dir.to_string_lossy().to_string(),
        "output_files": Value::Object(exported),
    }))
}

fn start_selection_job_request(
    separation_manager: &SeparationManager,
    cfg: &BackendConfig,
    stdout: &Arc<Mutex<io::Stdout>>,
    gpu_info: &Value,
    extra: &Map<String, Value>,
) -> Result<Value> {
    let job_id = Uuid::new_v4().to_string();
    emit_recipe_plan_event_if_needed(cfg, stdout, extra, &job_id);
    let config_obj = build_selection_job_config(cfg, extra, gpu_info, &job_id);
    let snapshot = separation_manager.submit(job_id.clone(), config_obj, cfg, stdout.clone())?;
    Ok(serde_json::json!({
        "job_id": snapshot.job_id,
        "status": snapshot.status,
        "job": snapshot,
    }))
}

fn main() -> Result<()> {
    let args: Vec<OsString> = std::env::args_os().collect();
    let assets_dir = locate_assets_dir(parse_arg_value(&args, "--assets-dir"))?;
    let models_dir = locate_models_dir(parse_arg_value(&args, "--models-dir"))?;
    let catalog_cache_dir = locate_catalog_cache_dir(parse_arg_value(&args, "--catalog-cache-dir"))?;
    let cfg = BackendConfig {
        catalog_bootstrap_path: locate_catalog_bootstrap_path(&assets_dir),
        catalog_remote_url: configured_catalog_remote_url(parse_arg_value(&args, "--catalog-url")),
        catalog_signature_url: configured_catalog_signature_url(parse_arg_value(&args, "--catalog-signature-url")),
        catalog_public_key_path: locate_catalog_public_key_path(
            &assets_dir,
            parse_arg_value(&args, "--catalog-public-key"),
        ),
        assets_dir,
        models_dir,
        catalog_cache_dir,
    };
    let catalog_status = refresh_or_fallback_catalog_status(&cfg)
        .context("initialize remote-first runtime catalog")?;

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
    let models_count = load_catalog_runtime(&cfg)
        .ok()
        .and_then(|v| v.get("models").and_then(|m| m.as_array()).map(|a| a.len()))
        .unwrap_or(0);
    let recipes_count = load_catalog_recipes(&cfg)
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
    caps.push(Value::from("quality"));
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
        },
        "catalog": serde_json::to_value(&catalog_status).unwrap_or(Value::Null)
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
                "remove_model"
                    | "import_custom_model"
                    | "model_preflight"
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
                    "separate_audio" | "model_preflight"
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
            "get_backend_status" => {
                let active_snapshots: Vec<SelectionJobSnapshot> = {
                    let jobs = separation_manager.jobs.lock().expect("jobs lock");
                    jobs.values()
                        .filter_map(|job| job.state.lock().ok().map(|state| state.clone()))
                        .collect()
                };
                let queued_snapshots: Vec<SelectionJobSnapshot> = {
                    let pending = separation_manager
                        .pending_jobs
                        .lock()
                        .expect("pending_jobs lock");
                    pending
                        .iter()
                        .filter_map(|job| job.state.lock().ok().map(|state| state.clone()))
                        .collect()
                };
                let active_job_id = active_snapshots
                    .iter()
                    .find(|snapshot| {
                        matches!(
                            snapshot.status.as_str(),
                            "starting" | "running" | "validating" | "cancelling"
                        )
                    })
                    .map(|snapshot| snapshot.job_id.clone());
                send_ok(
                    &stdout,
                    req.id,
                    serde_json::json!({
                        "backend_ready": true,
                        "worker_busy": active_job_id.is_some(),
                        "active_job_id": active_job_id,
                        "active_jobs": active_snapshots,
                        "queue_depth": queued_snapshots.len(),
                        "queued_jobs": queued_snapshots,
                        "rust_backend": true,
                        "has_gpu": has_gpu,
                    }),
                )?;
            }
            "get_runtime_fingerprint" => {
                send_ok(&stdout, req.id, runtime_fingerprint.clone())?;
            }
            "refresh_catalog" => {
                let status = refresh_remote_catalog(&cfg)?;
                send_ok(
                    &stdout,
                    req.id,
                    serde_json::to_value(status).unwrap_or(Value::Null),
                )?;
            }
            "get_catalog_status" => {
                let status = ensure_catalog_runtime_status(&cfg)?;
                send_ok(
                    &stdout,
                    req.id,
                    serde_json::to_value(status).unwrap_or(Value::Null),
                )?;
            }
            "get_catalog" => {
                let catalog = hydrated_catalog_runtime(&cfg)?;
                send_ok(&stdout, req.id, catalog)?;
            }
            "get_selection_installation" => {
                let runtime = load_catalog_runtime(&cfg)?;
                let (selection_type, selection_id) = selection_descriptor_from_extra(&req.extra);
                let selection_type = selection_type
                    .or_else(|| {
                        selection_id
                            .as_ref()
                            .and_then(|id| {
                                runtime
                                    .get("selection_index")
                                    .and_then(|v| v.as_array())
                                    .and_then(|entries| {
                                        entries.iter().find(|entry| {
                                            entry.get("selection_id").and_then(|v| v.as_str()) == Some(id.as_str())
                                        })
                                    })
                                    .and_then(|entry| entry.get("selection_type").and_then(|v| v.as_str()))
                                    .and_then(normalize_selection_type)
                            })
                    })
                    .or_else(|| {
                        req.extra
                            .get("model_id")
                            .and_then(|v| v.as_str())
                            .map(|_| "model".to_string())
                    })
                    .ok_or_else(|| anyhow!("selection_type is required"))?;
                let selection_id = selection_id.ok_or_else(|| anyhow!("selection_id is required"))?;
                let installation = catalog_selection_installation(&cfg, &runtime, &selection_type, &selection_id)?;
                send_ok(&stdout, req.id, installation)?;
            }
            "resolve_install_plan" => {
                let runtime = load_catalog_runtime(&cfg)?;
                let (selection_type, selection_id) = selection_descriptor_from_extra(&req.extra);
                let selection_type = selection_type
                    .or_else(|| {
                        selection_id
                            .as_ref()
                            .and_then(|id| {
                                runtime
                                    .get("selection_index")
                                    .and_then(|v| v.as_array())
                                    .and_then(|entries| {
                                        entries.iter().find(|entry| {
                                            entry.get("selection_id").and_then(|v| v.as_str()) == Some(id.as_str())
                                        })
                                    })
                                    .and_then(|entry| entry.get("selection_type").and_then(|v| v.as_str()))
                                    .and_then(normalize_selection_type)
                            })
                    })
                    .or_else(|| {
                        req.extra
                            .get("model_id")
                            .and_then(|v| v.as_str())
                            .map(|_| "model".to_string())
                    })
                    .ok_or_else(|| anyhow!("selection_type is required"))?;
                let selection_id = selection_id.ok_or_else(|| anyhow!("selection_id is required"))?;
                let plan = catalog_selection_install_plan(&cfg, &runtime, &selection_type, &selection_id)?;
                send_ok(&stdout, req.id, plan)?;
            }
            "verify_selection_artifacts" => {
                let runtime = load_catalog_runtime(&cfg)?;
                let (selection_type, selection_id) = selection_descriptor_from_extra(&req.extra);
                let selection_type = selection_type
                    .or_else(|| {
                        selection_id
                            .as_ref()
                            .and_then(|id| {
                                runtime
                                    .get("selection_index")
                                    .and_then(|v| v.as_array())
                                    .and_then(|entries| {
                                        entries.iter().find(|entry| {
                                            entry.get("selection_id").and_then(|v| v.as_str()) == Some(id.as_str())
                                        })
                                    })
                                    .and_then(|entry| entry.get("selection_type").and_then(|v| v.as_str()))
                                    .and_then(normalize_selection_type)
                            })
                    })
                    .or_else(|| {
                        req.extra
                            .get("model_id")
                            .and_then(|v| v.as_str())
                            .map(|_| "model".to_string())
                    })
                    .ok_or_else(|| anyhow!("selection_type is required"))?;
                let selection_id = selection_id.ok_or_else(|| anyhow!("selection_id is required"))?;
                let verification = catalog_selection_verification(&cfg, &runtime, &selection_type, &selection_id)?;
                send_ok(&stdout, req.id, verification)?;
            }
            "resolve_execution_plan" => {
                let runtime = load_catalog_runtime(&cfg)?;
                let (selection_type, selection_id) = selection_descriptor_from_extra(&req.extra);
                let selection_type = selection_type
                    .or_else(|| {
                        selection_id
                            .as_ref()
                            .and_then(|id| {
                                runtime
                                    .get("selection_index")
                                    .and_then(|v| v.as_array())
                                    .and_then(|entries| {
                                        entries.iter().find(|entry| {
                                            entry.get("selection_id").and_then(|v| v.as_str()) == Some(id.as_str())
                                        })
                                    })
                                    .and_then(|entry| entry.get("selection_type").and_then(|v| v.as_str()))
                                    .and_then(normalize_selection_type)
                            })
                    })
                    .or_else(|| {
                        req.extra
                            .get("model_id")
                            .and_then(|v| v.as_str())
                            .map(|_| "model".to_string())
                    })
                    .ok_or_else(|| anyhow!("selection_type is required"))?;
                let selection_id = selection_id.ok_or_else(|| anyhow!("selection_id is required"))?;
                let execution = catalog_execution_plan(&cfg, &runtime, &selection_type, &selection_id)?;
                send_ok(&stdout, req.id, execution)?;
            }
            "install_selection" | "install-selection" => {
                let result = install_selection(&cfg, &downloads, stdout.clone(), &req.extra);
                match result {
                    Ok(payload) => send_ok(&stdout, req.id, payload)?,
                    Err(error) => {
                        let _ = send_err(
                            &stdout,
                            req.id,
                            "INSTALL_SELECTION_FAILED",
                            &format!("{error:#}"),
                        );
                    }
                }
            }
            "import_selection_artifacts" | "import-selection-artifacts" => {
                let result = import_selection_artifacts(&cfg, &req.extra);
                match result {
                    Ok(payload) => send_ok(&stdout, req.id, payload)?,
                    Err(error) => {
                        let _ = send_err(
                            &stdout,
                            req.id,
                            "IMPORT_SELECTION_FAILED",
                            &format!("{error:#}"),
                        );
                    }
                }
            }
            "get_models" => {
                let catalog = hydrated_catalog_runtime(&cfg)?;
                let models = catalog
                    .get("models")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                send_ok(&stdout, req.id, Value::Array(models))?;
            }
            "get_recipes" => {
                let recipes_json = load_catalog_recipes(&cfg)?;
                let workflows_json = load_catalog_workflows(&cfg).unwrap_or(Value::Null);
                let recipes = recipes_json
                    .get("recipes")
                    .cloned()
                    .unwrap_or(Value::Array(vec![]));
                let workflows = workflows_json
                    .get("workflows")
                    .cloned()
                    .unwrap_or(Value::Array(vec![]));
                let models_json = load_models_with_guide_overrides(&cfg).unwrap_or(Value::Null);
                let mut out: Vec<Value> = Vec::new();
                let mut entries = recipes.as_array().cloned().unwrap_or_default();
                entries.extend(workflows.as_array().cloned().unwrap_or_default());

                for recipe in entries {
                    let mut obj = recipe.as_object().cloned().unwrap_or_default();
                    let required_model_ids = collect_recipe_model_ids(&recipe);
                    let simple_requested = obj
                        .get("simple_surface")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let surface_policy = obj
                        .get("surface_policy")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| {
                            if simple_requested {
                                "verified_only".to_string()
                            } else {
                                "advanced_only".to_string()
                            }
                        });
                    let requires_verified_assets = obj
                        .get("requires_verified_assets")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(simple_requested);
                    let requires_qa_pass = obj
                        .get("requires_qa_pass")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(simple_requested);
                    let golden_set_id = obj
                        .get("golden_set_id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let qa_policy_ready = golden_set_id
                        .as_ref()
                        .is_some_and(|id| !id.trim().is_empty())
                        && has_complete_audio_quality_thresholds(
                            obj.get("audio_quality_thresholds"),
                        );
                    let quality_tradeoff = obj
                        .get("quality_tradeoff")
                        .cloned()
                        .unwrap_or_else(|| {
                            obj.get("quality_goal")
                                .cloned()
                                .unwrap_or(Value::Null)
                        });

                    let mut required_models: Vec<Value> = Vec::new();
                    let mut surface_blockers: Vec<String> = Vec::new();

                    if simple_requested
                        && obj
                            .get("promotion_status")
                            .and_then(|v| v.as_str())
                            == Some("supported_advanced")
                    {
                        surface_blockers.push(
                            "Recipe is marked as supported_advanced and must stay out of Simple mode."
                                .to_string(),
                        );
                    }
                    if requires_qa_pass
                        && obj.get("qa_status").and_then(|v| v.as_str()) != Some("verified")
                    {
                        surface_blockers.push(
                            "Recipe requires qa_status=verified before it can be shown in Simple mode."
                                .to_string(),
                        );
                    }
                    if requires_qa_pass && !qa_policy_ready {
                        surface_blockers.push(
                            "Recipe requires a golden_set_id and complete audio_quality_thresholds before QA promotion."
                                .to_string(),
                        );
                    }

                    for model_id in required_model_ids {
                        let model_obj =
                            find_model_object(&models_json, &model_id).unwrap_or(Value::Null);
                        let catalog_status = model_obj
                            .get("catalog_status")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let metrics_status = model_obj
                            .get("metrics_status")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let readiness = model_obj
                            .get("status")
                            .and_then(|v| v.get("readiness"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let simple_allowed = model_obj
                            .get("status")
                            .and_then(|v| v.get("simple_allowed"))
                            .and_then(|v| v.as_bool());
                        let blocked_reason = model_obj
                            .get("status")
                            .and_then(|v| v.get("blocking_reason"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| {
                                model_obj
                                    .get("runtime")
                                    .and_then(|v| v.get("blocking_reason"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            });
                        let runtime_adapter = detect_runtime_adapter(&model_obj);
                        let install_manifest =
                            resolve_model_download_manifest(&cfg, &model_id, &model_obj);
                        let install_mode = install_manifest.install_mode.clone();
                        let download_mode = install_manifest.mode.clone();

                        if simple_requested && requires_verified_assets {
                            if catalog_status.as_deref() != Some("verified") {
                                surface_blockers.push(format!(
                                    "{model_id} is {status} and cannot back a Simple recipe yet.",
                                    status = catalog_status
                                        .clone()
                                        .unwrap_or_else(|| "unclassified".to_string())
                                ));
                            }
                            if matches!(install_mode.as_str(), "manual" | "custom_runtime") {
                                surface_blockers.push(format!(
                                    "{model_id} requires {install_mode} installation and cannot be Simple-surfaced."
                                ));
                            }
                            if matches!(download_mode.as_str(), "manual" | "unavailable") {
                                surface_blockers.push(format!(
                                    "{model_id} does not expose a verified direct-download manifest."
                                ));
                            }
                            if runtime_adapter.is_none() {
                                surface_blockers.push(format!(
                                    "{model_id} does not expose a runtime adapter."
                                ));
                            }
                            if simple_allowed == Some(false) {
                                surface_blockers.push(
                                    blocked_reason
                                        .clone()
                                        .unwrap_or_else(|| {
                                            format!(
                                                "{model_id} is explicitly blocked from Simple mode."
                                            )
                                        }),
                                );
                            }
                        }

                        required_models.push(serde_json::json!({
                            "id": model_id,
                            "catalog_status": catalog_status,
                            "metrics_status": metrics_status,
                            "readiness": readiness,
                            "simple_allowed": simple_allowed,
                            "blocked_reason": blocked_reason,
                            "runtime_adapter": runtime_adapter,
                            "install_mode": install_mode,
                            "download_mode": download_mode,
                        }));
                    }

                    surface_blockers.sort();
                    surface_blockers.dedup();

                    let effective_simple_surface = simple_requested
                        && surface_policy == "verified_only"
                        && surface_blockers.is_empty();

                    obj.insert(
                        "simple_surface".to_string(),
                        Value::Bool(effective_simple_surface),
                    );
                    obj.insert(
                        "surface_policy".to_string(),
                        Value::String(surface_policy),
                    );
                    obj.insert(
                        "requires_verified_assets".to_string(),
                        Value::Bool(requires_verified_assets),
                    );
                    obj.insert(
                        "requires_qa_pass".to_string(),
                        Value::Bool(requires_qa_pass),
                    );
                    obj.insert(
                        "qa_policy_ready".to_string(),
                        Value::Bool(qa_policy_ready),
                    );
                    if let Some(golden_set_id) = golden_set_id {
                        obj.insert(
                            "golden_set_id".to_string(),
                            Value::String(golden_set_id),
                        );
                    }
                    obj.insert(
                        "quality_tradeoff".to_string(),
                        quality_tradeoff,
                    );
                    if !obj.contains_key("guide_topics") {
                        let mut guide_topics: Vec<Value> = Vec::new();
                        if let Some(target) = obj.get("target").cloned() {
                            guide_topics.push(target);
                        }
                        if let Some(family) = obj.get("family").cloned() {
                            guide_topics.push(family);
                        }
                        obj.insert("guide_topics".to_string(), Value::Array(guide_topics));
                    }
                    obj.insert(
                        "required_model_statuses".to_string(),
                        Value::Array(required_models),
                    );
                    obj.insert(
                        "surface_blockers".to_string(),
                        Value::Array(
                            surface_blockers
                                .into_iter()
                                .map(Value::String)
                                .collect::<Vec<_>>(),
                        ),
                    );
                    out.push(Value::Object(obj));
                }

                send_ok(&stdout, req.id, Value::Array(out))?;
            }
            "quality_baseline_create" => {
                let manifest = build_quality_manifest(&cfg, &req.extra, &stdout)?;
                if let Some(path) = req
                    .extra
                    .get("manifest_path")
                    .or_else(|| req.extra.get("manifestPath"))
                    .and_then(|v| v.as_str())
                {
                    let p = PathBuf::from(path);
                    if let Some(parent) = p.parent() {
                        std::fs::create_dir_all(parent)
                            .with_context(|| format!("create dir for {}", p.display()))?;
                    }
                    std::fs::write(
                        &p,
                        serde_json::to_string_pretty(&manifest)
                            .context("serialize quality manifest")?,
                    )
                    .with_context(|| format!("write manifest {}", p.display()))?;
                }

                send_event(
                    &stdout,
                    serde_json::json!({
                        "type": "quality_complete",
                        "action": "quality_baseline_create",
                        "manifest_hash": manifest.get("manifest_hash").cloned().unwrap_or(Value::Null),
                        "ts": now_ts_seconds()
                    }),
                );
                send_ok(&stdout, req.id, manifest)?;
            }
            "quality_compare" => {
                send_quality_progress(
                    &stdout,
                    "load_manifests",
                    5.0,
                    "Loading baseline and candidate manifests",
                    None,
                );
                let baseline = parse_manifest_input(&cfg, &req.extra, &stdout, "baseline")?;
                send_quality_progress(
                    &stdout,
                    "load_manifests",
                    35.0,
                    "Baseline manifest loaded",
                    None,
                );
                let candidate = parse_manifest_input(&cfg, &req.extra, &stdout, "candidate")?;
                send_quality_progress(&stdout, "compare", 70.0, "Comparing manifests", None);
                let mut comparison = compare_quality_manifests(&baseline, &candidate);
                let recipe_id = req
                    .extra
                    .get("recipe_id")
                    .or_else(|| req.extra.get("recipeId"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| extract_recipe_id_from_manifest(&candidate))
                    .or_else(|| extract_recipe_id_from_manifest(&baseline));
                let recipes_json =
                    read_json_file(&cfg.assets_dir.join("recipes.json")).unwrap_or(Value::Null);
                let recipe_obj = recipe_id
                    .as_ref()
                    .and_then(|id| find_recipe_object(&recipes_json, id));
                let resolved_thresholds_value = req
                    .extra
                    .get("audio_quality_thresholds")
                    .or_else(|| req.extra.get("audioQualityThresholds"))
                    .cloned()
                    .or_else(|| {
                        recipe_obj
                            .as_ref()
                            .and_then(|recipe| recipe.get("audio_quality_thresholds"))
                            .cloned()
                    });
                let resolved_golden_set_id = req
                    .extra
                    .get("golden_set_id")
                    .or_else(|| req.extra.get("goldenSetId"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| extract_golden_set_id_from_manifest(&candidate))
                    .or_else(|| extract_golden_set_id_from_manifest(&baseline))
                    .or_else(|| {
                        recipe_obj
                            .as_ref()
                            .and_then(|recipe| recipe.get("golden_set_id"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    });
                let qa_policy_source = if req
                    .extra
                    .get("audio_quality_thresholds")
                    .or_else(|| req.extra.get("audioQualityThresholds"))
                    .is_some()
                {
                    "request"
                } else if recipe_obj.is_some() {
                    "recipe"
                } else {
                    "default"
                };
                let thresholds =
                    parse_audio_quality_thresholds(resolved_thresholds_value.as_ref());
                send_quality_progress(
                    &stdout,
                    "audio_regression",
                    82.0,
                    "Running pairwise audio regression checks",
                    None,
                );
                let audio_quality = build_audio_quality_report(&baseline, &candidate, &thresholds);
                if let Some(comparison_obj) = comparison.as_object_mut() {
                    let verdict = audio_quality
                        .get("verdict")
                        .and_then(|v| v.as_str())
                        .unwrap_or("skipped");
                    let failed_pairs = audio_quality
                        .get("summary")
                        .and_then(|v| v.get("failed_pairs"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    let mut difference_count = comparison_obj
                        .get("difference_count")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    if verdict == "fail" {
                        if let Some(differences) = comparison_obj
                            .get_mut("differences")
                            .and_then(|v| v.as_array_mut())
                        {
                            if let Some(pairs) = audio_quality.get("pairs").and_then(|v| v.as_array()) {
                                for pair in pairs {
                                    if pair.get("verdict").and_then(|v| v.as_str()) != Some("fail") {
                                        continue;
                                    }
                                    differences.push(serde_json::json!({
                                        "type": "audio_quality_regression",
                                        "label": pair.get("label").cloned().unwrap_or(Value::Null),
                                        "baseline": pair.get("reference_path").cloned().unwrap_or(Value::Null),
                                        "candidate": pair.get("candidate_path").cloned().unwrap_or(Value::Null),
                                        "reason": pair.get("regressions").cloned().unwrap_or(Value::Null),
                                    }));
                                }
                            }
                            difference_count = differences.len();
                        }
                        comparison_obj.insert("compatible".to_string(), Value::Bool(false));
                    }
                    let score_penalty = (difference_count as i64 * 10)
                        + (failed_pairs as i64 * 5);
                    comparison_obj.insert(
                        "quality_score".to_string(),
                        Value::from((100 - score_penalty).max(0)),
                    );
                    comparison_obj.insert(
                        "difference_count".to_string(),
                        Value::from(difference_count as u64),
                    );
                    comparison_obj.insert("audio_quality".to_string(), audio_quality.clone());
                    comparison_obj.insert(
                        "audio_quality_verdict".to_string(),
                        Value::String(verdict.to_string()),
                    );
                }
                send_event(
                    &stdout,
                    serde_json::json!({
                        "type": "quality_complete",
                        "action": "quality_compare",
                        "compatible": comparison.get("compatible").cloned().unwrap_or(Value::Bool(false)),
                        "quality_score": comparison.get("quality_score").cloned().unwrap_or(Value::Null),
                        "audio_quality_verdict": comparison.get("audio_quality_verdict").cloned().unwrap_or(Value::Null),
                        "ts": now_ts_seconds()
                    }),
                );

                let mut out = Map::new();
                out.insert("baseline".to_string(), baseline);
                out.insert("candidate".to_string(), candidate);
                out.insert("comparison".to_string(), comparison);
                out.insert(
                    "qa_policy".to_string(),
                    serde_json::json!({
                        "recipe_id": recipe_id,
                        "golden_set_id": resolved_golden_set_id,
                        "audio_quality_thresholds": {
                            "min_correlation": thresholds.min_correlation,
                            "min_snr_db": thresholds.min_snr_db,
                            "min_si_sdr_db": thresholds.min_si_sdr_db,
                            "max_gain_delta_db": thresholds.max_gain_delta_db,
                            "max_clipped_samples": thresholds.max_clipped_samples
                        },
                        "source": qa_policy_source
                    }),
                );
                send_ok(&stdout, req.id, Value::Object(out))?;
            }
            "get-gpu-devices" | "get_gpu_devices" => {
                send_ok(&stdout, req.id, gpu_info.clone())?;
            }
            "resolve_youtube" => {
                let url = match req.extra.get("url").and_then(|v| v.as_str()) {
                    Some(v) if !v.trim().is_empty() => v.trim(),
                    _ => {
                        send_err(&stdout, req.id, "INVALID", "url is required")?;
                        continue;
                    }
                };

                match resolve_youtube_native(url, &stdout) {
                    Ok(result) => send_ok(&stdout, req.id, result)?,
                    Err(e) => send_err(&stdout, req.id, "YOUTUBE_FAILED", &format!("{e:#}"))?,
                }
            }
            "import_model_files" => {
                let req_id = req.id.clone();
                let result: Result<()> = (|| {
                    let model_id = req
                        .extra
                        .get("model_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow!("model_id is required"))?
                        .to_string();
                    let files = req
                        .extra
                        .get("files")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| anyhow!("files[] is required"))?;
                    let allow_copy = req
                        .extra
                        .get("allow_copy")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);

                    let (manifest, installation) =
                        import_model_artifacts(&cfg, &model_id, files, allow_copy)?;

                    send_ok(
                        &stdout,
                        req_id,
                        serde_json::json!({
                            "model_id": model_id,
                            "download": serde_json::to_value(&manifest).unwrap_or(Value::Null),
                            "installation": serde_json::to_value(&installation).unwrap_or(Value::Null),
                            "artifacts": serde_json::to_value(&installation.artifacts).unwrap_or(Value::Null),
                        }),
                    )?;
                    Ok(())
                })();

                if let Err(e) = result {
                    let _ = send_err(&stdout, req.id, "IMPORT_FAILED", &format!("{e:#}"));
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
                let explicit_model_id = req.extra.get("model_id").and_then(|v| v.as_str());
                let selection_type_req = req
                    .extra
                    .get("selection_type")
                    .or_else(|| req.extra.get("selectionType"))
                    .and_then(|v| v.as_str())
                    .and_then(normalize_selection_type);
                let selection_id_req = req
                    .extra
                    .get("selection_id")
                    .or_else(|| req.extra.get("selectionId"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                let runtime_for_selection = if explicit_model_id.is_none() && selection_id_req.is_some()
                {
                    load_catalog_runtime(&cfg).ok()
                } else {
                    None
                };
                let resolved_selection_type = selection_type_req.clone().or_else(|| {
                    selection_id_req.as_ref().and_then(|selection_id| {
                        runtime_for_selection.as_ref().and_then(|runtime| {
                            runtime
                                .get("selection_index")
                                .and_then(|v| v.as_array())
                                .and_then(|entries| {
                                    entries.iter().find(|entry| {
                                        entry.get("selection_id").and_then(|v| v.as_str())
                                            == Some(selection_id.as_str())
                                    })
                                })
                                .and_then(|entry| entry.get("selection_type").and_then(|v| v.as_str()))
                                .and_then(normalize_selection_type)
                        })
                    })
                });
                let catalog_selection = match (
                    runtime_for_selection.as_ref(),
                    resolved_selection_type.as_deref(),
                    selection_id_req.as_deref(),
                ) {
                    (Some(runtime), Some(selection_type), Some(selection_id)) => {
                        catalog_selection_entry(runtime, selection_type, selection_id)
                    }
                    _ => None,
                };
                let model_id = explicit_model_id.or_else(|| selection_id_req.as_deref());
                let stems_val = req.extra.get("stems").cloned().unwrap_or(Value::Null);
                let device_req = req
                    .extra
                    .get("device")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                // Flexible numeric parsing for overlap and segment_size (handle strings from UI)
                let overlap_req = req.extra.get("overlap").and_then(|v| {
                    if let Some(f) = v.as_f64() {
                        Some(f)
                    } else if let Some(s) = v.as_str() {
                        s.parse::<f64>().ok()
                    } else {
                        None
                    }
                });
                let segment_size_req = req.extra.get("segment_size").and_then(|v| {
                    if let Some(i) = v.as_i64() {
                        Some(i)
                    } else if let Some(s) = v.as_str() {
                        s.parse::<i64>().ok()
                    } else {
                        None
                    }
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
                let ensemble_cfg_val = req
                    .extra
                    .get("ensemble_config")
                    .or_else(|| req.extra.get("ensembleConfig"));
                let workflow_val = req
                    .extra
                    .get("workflow")
                    .cloned()
                    .or_else(|| match resolved_selection_type.as_deref() {
                        Some("workflow") => catalog_selection.clone(),
                        _ => None,
                    })
                    .unwrap_or(Value::Null);

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
                let mut runtime_blocks: Vec<String> = Vec::new();
                let mut recommended_adjustments: Vec<String> = Vec::new();

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
                let models_json = load_models_with_guide_overrides(&cfg).unwrap_or(Value::Null);
                let recipes_json =
                    read_json_file(&cfg.assets_dir.join("recipes.json")).unwrap_or(Value::Null);

                let mut resolved_model_id: Option<String> = None;
                let mut requested_model_ids: Vec<String> = Vec::new();

                // If model_id refers to a recipe, preflight should validate the recipe's step model dependencies
                // rather than treating the recipe id itself as a downloadable model artifact.
                let mut recipe_step_model_ids: Vec<String> = Vec::new();
                let mut recipe_type: Option<String> = None;
                let mut recipe_plan: Option<Value> = None;
                let mut recipe_meta: Option<Value> = None;
                if workflow_val.is_object() {
                    recipe_type = workflow_val
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .map(|kind| kind.to_string());
                    recipe_plan = Some(workflow_val.clone());
                    recipe_meta = Some(workflow_val.clone());
                    requested_model_ids.extend(collect_workflow_model_ids(&workflow_val));
                    resolved_model_id = workflow_val
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| model_id.map(|s| s.to_string()))
                        .or_else(|| {
                            workflow_val
                                .get("kind")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                        });
                } else if let Some(id) = model_id {
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
                        for key in [
                            "family",
                            "quality_goal",
                            "difficulty",
                            "expected_vram_tier",
                            "expected_runtime_tier",
                            "guide_rank",
                            "simple_surface",
                            "simple_goal",
                            "recommended_for",
                            "contraindications",
                            "workflow_summary",
                        ] {
                            if let Some(value) = recipe.get(key) {
                                plan.insert(key.to_string(), value.clone());
                            }
                        }
                        if let Some(pp) = recipe.get("post_processing") {
                            plan.insert("post_processing".to_string(), pp.clone());
                        }
                        if !steps_out.is_empty() {
                            plan.insert("steps".to_string(), Value::Array(steps_out));
                        }
                        recipe_plan = Some(Value::Object(plan));
                        recipe_meta = Some(recipe.clone());
                    }
                }

                if workflow_val.is_object() {
                    if requested_model_ids.is_empty() {
                        warnings.push(
                            "Workflow payload did not reference any concrete models".to_string(),
                        );
                    }
                } else if let Some(Value::Object(o)) = ensemble_cfg_val {
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
                        if let Some(ref t) = recipe_type {
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

                let mut phase_fix_compatibility: Option<String> = None;
                let mut crossover_validation: Option<String> = None;

                let workflow_algorithm = workflow_val
                    .get("blend")
                    .and_then(|v| v.get("algorithm"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        ensemble_cfg_val
                            .and_then(|v| v.get("algorithm"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    });
                let split_freq = workflow_val
                    .get("blend")
                    .and_then(|v| v.get("splitFreq"))
                    .and_then(|v| v.as_i64())
                    .or_else(|| {
                        ensemble_cfg_val
                            .and_then(|v| v.get("split_freq"))
                            .and_then(|v| v.as_i64())
                    })
                    .or_else(|| req.extra.get("split_freq").and_then(|v| v.as_i64()))
                    .or_else(|| req.extra.get("splitFreq").and_then(|v| v.as_i64()));

                if workflow_algorithm.as_deref() == Some("frequency_split") {
                    if requested_model_ids.len() < 2 {
                        errors.push(
                            "Frequency-split workflows need at least two concrete models"
                                .to_string(),
                        );
                    }
                    if let Some(freq) = split_freq {
                        if !(600..=1400).contains(&freq) {
                            warnings.push(format!(
                                "Crossover frequency {freq} Hz is outside the guide-backed safe range of 600-1400 Hz"
                            ));
                            crossover_validation =
                                Some(format!("outside safe range ({freq} Hz)"));
                        } else {
                            crossover_validation = Some(format!("validated at {freq} Hz"));
                        }
                    } else {
                        warnings.push(
                            "Frequency-split workflow did not specify a crossover frequency"
                                .to_string(),
                        );
                        crossover_validation = Some("missing crossover frequency".to_string());
                    }
                }
                if phase_fix_enabled && requested_model_ids.len() < 2 {
                    warnings.push(
                        "Phase-fix is enabled but fewer than two referenced models were resolved"
                            .to_string(),
                    );
                    phase_fix_compatibility =
                        Some("needs at least two concrete models".to_string());
                } else if phase_fix_enabled {
                    phase_fix_compatibility = Some("reference pair resolved".to_string());
                }

                // For each referenced model_id, check if any expected local file exists.
                // This mirrors Python's ability to load checkpoints/configs (ckpt/pth/yaml/onnx/etc).
                let mut needs_fno_dependency = false;
                let mut required_models_out: Vec<Value> = Vec::new();
                let mut models_requiring_cuda: Vec<String> = Vec::new();
                let mut missing_runtime_assets: Vec<String> = Vec::new();
                let mut unsupported_patch_profiles: Vec<String> = Vec::new();
                let mut recommended_operating_profile: Option<String> = None;
                let mut runtime_adapter: Option<String> = None;
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

                    let readiness = model_obj
                        .get("status")
                        .and_then(|v| v.get("readiness"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let simple_allowed = model_obj
                        .get("status")
                        .and_then(|v| v.get("simple_allowed"))
                        .and_then(|v| v.as_bool());
                    let blocking_reason = model_obj
                        .get("status")
                        .and_then(|v| v.get("blocking_reason"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| {
                            model_obj
                                .get("runtime")
                                .and_then(|v| v.get("blocking_reason"))
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                        });
                    let install_mode = model_obj
                        .get("install")
                        .and_then(|v| v.get("mode"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let curated = model_obj
                        .get("status")
                        .and_then(|v| v.get("curated"))
                        .and_then(|v| v.as_bool());
                    let support_tier = model_obj
                        .get("status")
                        .and_then(|v| v.get("support_tier"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let runtime_engine = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("engine"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let runtime_model_type = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("model_type"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let runtime_adapter_for_model = detect_runtime_adapter(&model_obj);
                    let runtime_config_ref = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("config_ref"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let runtime_checkpoint_ref = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("checkpoint_ref"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let runtime_patch_profile = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("patch_profile"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let requires_manual_assets = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("requires_manual_assets"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let required_files = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("required_files"))
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let runtime_required = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("required"))
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let runtime_allowed = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("allowed"))
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let runtime_preferred = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("preferred"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    let runtime_fallbacks = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("fallbacks"))
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let runtime_hosts = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("hosts"))
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    let install_burden = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("install_burden"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let requires_patch = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("requires_patch"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let requires_custom_repo_file = model_obj
                        .get("runtime")
                        .and_then(|v| v.get("requires_custom_repo_file"))
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    if runtime_required
                        .iter()
                        .any(|value| value.as_str() == Some("cuda"))
                    {
                        models_requiring_cuda.push(mid.clone());
                    }
                    let missing_assets_for_model =
                        collect_missing_runtime_assets(&cfg.models_dir, mid, &model_obj);
                    if !missing_assets_for_model.is_empty() {
                        for asset in &missing_assets_for_model {
                            missing_runtime_assets.push(format!("{mid}: {asset}"));
                        }
                    }
                    if runtime_adapter.is_none() {
                        runtime_adapter = runtime_adapter_for_model.clone();
                    }
                    if runtime_adapter_for_model.is_none() {
                        runtime_blocks.push(format!(
                            "{mid} does not expose an internal runtime adapter profile yet"
                        ));
                    }
                    if recommended_operating_profile.is_none() {
                        recommended_operating_profile = model_obj
                            .get("operating_profiles")
                            .and_then(|v| v.as_object())
                            .and_then(|profiles| {
                                if profiles.contains_key("balanced") {
                                    Some("balanced".to_string())
                                } else {
                                    profiles.keys().next().map(|key| key.to_string())
                                }
                            });
                    }
                    if runtime_patch_profile.is_some()
                        && runtime_adapter_for_model.as_deref() != Some("custom_builtin_variant")
                    {
                        unsupported_patch_profiles.push(format!(
                            "{mid}: {}",
                            runtime_patch_profile.clone().unwrap_or_default()
                        ));
                    }

                    if matches!(readiness.as_deref(), Some("blocked") | Some("manual")) {
                        runtime_blocks.push(format!(
                            "{mid} is marked as {}",
                            readiness.clone().unwrap_or_else(|| "blocked".to_string())
                        ));
                    }
                    if matches!(install_mode.as_deref(), Some("manual")) {
                        runtime_blocks.push(format!(
                            "{mid} requires manual installation before it can run"
                        ));
                    }
                    if matches!(install_mode.as_deref(), Some("custom_runtime")) {
                        runtime_blocks.push(format!(
                            "{mid} requires a custom runtime or external dependency"
                        ));
                    }
                    if requires_patch {
                        runtime_blocks.push(format!(
                            "{mid} requires a repo/runtime patch before it can run safely"
                        ));
                    }
                    if !missing_assets_for_model.is_empty() {
                        runtime_blocks.push(format!(
                            "{mid} is missing runtime assets required by the selected adapter"
                        ));
                    }
                    if matches!(install_burden.as_deref(), Some("high")) {
                        recommended_adjustments.push(format!(
                            "{mid}: high setup burden, keep this workflow in Advanced mode unless the runtime is already verified"
                        ));
                    }
                    if !runtime_hosts.is_empty()
                        && runtime_hosts
                            .iter()
                            .all(|value| matches!(value.as_str(), Some("msst" | "custom_script")))
                    {
                        recommended_adjustments.push(format!(
                            "{mid}: best supported through {}",
                            runtime_hosts
                                .iter()
                                .filter_map(|value| value.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                    }
                    if simple_allowed == Some(false) {
                        if let Some(reason) = &blocking_reason {
                            recommended_adjustments.push(format!("{mid}: {reason}"));
                        }
                    }

                    let mut required_model = serde_json::Map::new();
                    required_model.insert("id".to_string(), Value::String(mid.clone()));
                    required_model.insert(
                        "name".to_string(),
                        model_obj.get("name").cloned().unwrap_or(Value::Null),
                    );
                    required_model.insert("installed".to_string(), Value::Bool(exists));
                    required_model.insert("curated".to_string(), curated.into());
                    required_model.insert("support_tier".to_string(), support_tier.into());
                    required_model.insert(
                        "guide_rank".to_string(),
                        model_obj.get("guide_rank").cloned().unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "catalog_status".to_string(),
                        model_obj
                            .get("catalog_status")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "metrics_status".to_string(),
                        model_obj
                            .get("metrics_status")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert("readiness".to_string(), readiness.into());
                    required_model.insert("simple_allowed".to_string(), simple_allowed.into());
                    required_model.insert("blocking_reason".to_string(), blocking_reason.into());
                    required_model.insert(
                        "blocked_reason".to_string(),
                        model_obj
                            .get("status")
                            .and_then(|v| v.get("blocking_reason"))
                            .cloned()
                            .or_else(|| {
                                model_obj
                                    .get("runtime")
                                    .and_then(|v| v.get("blocking_reason"))
                                    .cloned()
                            })
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert("install_mode".to_string(), install_mode.into());
                    required_model.insert("runtime_engine".to_string(), runtime_engine.into());
                    required_model.insert(
                        "runtime_model_type".to_string(),
                        runtime_model_type.into(),
                    );
                    required_model.insert(
                        "runtime_adapter".to_string(),
                        runtime_adapter_for_model.into(),
                    );
                    required_model.insert(
                        "runtime_allowed".to_string(),
                        Value::Array(runtime_allowed),
                    );
                    required_model.insert(
                        "runtime_preferred".to_string(),
                        runtime_preferred,
                    );
                    required_model.insert(
                        "runtime_variant".to_string(),
                        model_obj
                            .get("runtime")
                            .and_then(|v| v.get("variant"))
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "runtime_config_ref".to_string(),
                        runtime_config_ref.into(),
                    );
                    required_model.insert(
                        "runtime_checkpoint_ref".to_string(),
                        runtime_checkpoint_ref.into(),
                    );
                    required_model.insert(
                        "runtime_patch_profile".to_string(),
                        runtime_patch_profile.into(),
                    );
                    required_model.insert(
                        "runtime_required".to_string(),
                        Value::Array(runtime_required),
                    );
                    required_model.insert(
                        "runtime_fallbacks".to_string(),
                        Value::Array(runtime_fallbacks),
                    );
                    required_model.insert(
                        "runtime_hosts".to_string(),
                        Value::Array(runtime_hosts),
                    );
                    required_model.insert("install_burden".to_string(), install_burden.into());
                    required_model.insert(
                        "requires_manual_assets".to_string(),
                        Value::Bool(requires_manual_assets),
                    );
                    required_model.insert(
                        "missing_assets".to_string(),
                        Value::Array(
                            missing_assets_for_model
                                .iter()
                                .cloned()
                                .map(Value::String)
                                .collect::<Vec<_>>(),
                        ),
                    );
                    required_model.insert(
                        "required_files".to_string(),
                        Value::Array(required_files),
                    );
                    required_model.insert(
                        "requires_patch".to_string(),
                        Value::Bool(requires_patch),
                    );
                    required_model.insert(
                        "requires_custom_repo_file".to_string(),
                        Value::Array(requires_custom_repo_file),
                    );
                    required_model.insert(
                        "quality_role".to_string(),
                        model_obj
                            .get("quality_role")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "workflow_roles".to_string(),
                        model_obj
                            .get("workflow_roles")
                            .cloned()
                            .unwrap_or(Value::Array(vec![])),
                    );
                    required_model.insert(
                        "best_for".to_string(),
                        model_obj
                            .get("best_for")
                            .cloned()
                            .unwrap_or(Value::Array(vec![])),
                    );
                    required_model.insert(
                        "artifacts_risk".to_string(),
                        model_obj
                            .get("artifacts_risk")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "vram_profile".to_string(),
                        model_obj
                            .get("vram_profile")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "chunk_overlap_policy".to_string(),
                        model_obj
                            .get("chunk_overlap_policy")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "quality_axes".to_string(),
                        model_obj
                            .get("quality_axes")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "content_fit".to_string(),
                        model_obj
                            .get("content_fit")
                            .cloned()
                            .unwrap_or(Value::Array(vec![])),
                    );
                    required_model.insert(
                        "operating_profiles".to_string(),
                        model_obj
                            .get("operating_profiles")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "quality_tier".to_string(),
                        model_obj
                            .get("quality_profile")
                            .and_then(|v| v.get("quality_tier"))
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_model.insert(
                        "target_roles".to_string(),
                        model_obj
                            .get("quality_profile")
                            .and_then(|v| v.get("target_roles"))
                            .cloned()
                            .unwrap_or(Value::Array(vec![])),
                    );
                    required_model.insert(
                        "vram_required".to_string(),
                        model_obj
                            .get("vram_required")
                            .cloned()
                            .unwrap_or(Value::Null),
                    );
                    required_models_out.push(Value::Object(required_model));
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
                if !missing_runtime_assets.is_empty() {
                    errors.push(format!(
                        "Missing runtime asset(s): {}",
                        missing_runtime_assets.join(", ")
                    ));
                }
                if !unsupported_patch_profiles.is_empty() {
                    errors.push(format!(
                        "Unsupported patch profile(s): {}",
                        unsupported_patch_profiles.join(", ")
                    ));
                }

                // FNO dependency preflight: fail early with a clear message when neuralop is missing.
                if needs_fno_dependency {
                    if let Err(e) = check_neuralop_available() {
                        errors.push(e.to_string());
                        runtime_blocks.push("Missing neuralop.models.FNO1d support".to_string());
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
                    runtime_blocks.push("CUDA requested with no detected GPU".to_string());
                }
                if !resolved_device.starts_with("cuda") {
                    for model_id in &models_requiring_cuda {
                        runtime_blocks.push(format!(
                            "{model_id} requires CUDA but the resolved device is {resolved_device}"
                        ));
                    }
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
                                recommended_adjustments.push(format!(
                                    "Use segment_size={max_safe} for the selected GPU ({vram_gb:.1} GB VRAM)."
                                ));
                            }
                        }
                    }
                }

                // NOTE: We intentionally do not attempt ONNX introspection here.
                // Rust-native inference is not the current execution path; this preflight is a compatibility shim.

                let torch_available = check_torch_available();
                let can_proceed = errors.is_empty();
                let estimated_vram_gb = required_models_out
                    .iter()
                    .filter_map(|v| v.get("vram_required").and_then(|v| v.as_f64()))
                    .fold(0.0_f64, f64::max);
                let simple_surface_requested = recipe_meta
                    .as_ref()
                    .and_then(|v| v.get("simple_surface"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let surface_policy = recipe_meta
                    .as_ref()
                    .and_then(|v| v.get("surface_policy"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(if simple_surface_requested {
                        "verified_only"
                    } else {
                        "advanced_only"
                    });
                let requires_verified_assets = recipe_meta
                    .as_ref()
                    .and_then(|v| v.get("requires_verified_assets"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(simple_surface_requested);
                let requires_qa_pass = recipe_meta
                    .as_ref()
                    .and_then(|v| v.get("requires_qa_pass"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(simple_surface_requested);
                let golden_set_id = recipe_meta
                    .as_ref()
                    .and_then(|v| v.get("golden_set_id"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let qa_policy_ready = golden_set_id
                    .as_ref()
                    .is_some_and(|id| !id.trim().is_empty())
                    && has_complete_audio_quality_thresholds(
                        recipe_meta
                            .as_ref()
                            .and_then(|v| v.get("audio_quality_thresholds")),
                    );
                let mut surface_blockers: Vec<String> = Vec::new();
                if simple_surface_requested
                    && recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("promotion_status"))
                        .and_then(|v| v.as_str())
                        == Some("supported_advanced")
                {
                    surface_blockers.push(
                        "Recipe is marked supported_advanced and must stay out of Simple mode."
                            .to_string(),
                    );
                }
                if requires_qa_pass
                    && recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("qa_status"))
                        .and_then(|v| v.as_str())
                        != Some("verified")
                {
                    surface_blockers.push(
                        "Recipe requires qa_status=verified before it can be surfaced in Simple mode."
                            .to_string(),
                    );
                }
                if requires_qa_pass && !qa_policy_ready {
                    surface_blockers.push(
                        "Recipe requires a golden_set_id and complete audio_quality_thresholds before promotion."
                            .to_string(),
                    );
                }
                if requires_verified_assets {
                    for model in &required_models_out {
                        let mid = model.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                        if model.get("catalog_status").and_then(|v| v.as_str()) != Some("verified")
                        {
                            surface_blockers.push(format!(
                                "{mid} is not catalog_status=verified."
                            ));
                        }
                        if matches!(
                            model.get("install_mode").and_then(|v| v.as_str()),
                            Some("manual" | "custom_runtime")
                        ) {
                            surface_blockers.push(format!(
                                "{mid} requires manual/custom runtime installation."
                            ));
                        }
                        if model
                            .get("runtime_adapter")
                            .and_then(|v| v.as_str())
                            .is_none()
                        {
                            surface_blockers.push(format!(
                                "{mid} does not expose a runtime adapter."
                            ));
                        }
                        if model.get("simple_allowed").and_then(|v| v.as_bool()) == Some(false) {
                            let reason = model
                                .get("blocked_reason")
                                .and_then(|v| v.as_str())
                                .unwrap_or("model is blocked from Simple mode");
                            surface_blockers.push(format!("{mid}: {reason}"));
                        }
                    }
                }
                surface_blockers.sort();
                surface_blockers.dedup();
                let simple_surface = simple_surface_requested
                    && surface_policy == "verified_only"
                    && surface_blockers.is_empty();
                let difficulty = recipe_meta
                    .as_ref()
                    .and_then(|v| v.get("difficulty"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(if simple_surface { "simple" } else { "advanced" });
                let should_use_simple = simple_surface && runtime_blocks.is_empty();
                let should_use_advanced = !should_use_simple;
                let workflow_name = recipe_meta
                    .as_ref()
                    .and_then(|v| v.get("name"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| {
                        required_models_out
                            .first()
                            .and_then(|v| v.get("name"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    });
                let fallback_reason = if !runtime_blocks.is_empty() {
                    Some("Runtime blockers prevent the preferred workflow from running as-is.")
                } else if !missing_models.is_empty() {
                    Some("Install the required models before running this workflow.")
                } else {
                    None
                };

                let plan = serde_json::json!({
                    "workflow_name": workflow_name,
                    "workflow_type": recipe_type.clone().unwrap_or_else(|| {
                        if workflow_val.is_object() {
                            workflow_val
                                .get("kind")
                                .and_then(|v| v.as_str())
                                .unwrap_or("workflow")
                                .to_string()
                        } else if ensemble_cfg_val.is_some() {
                            "ensemble".to_string()
                        } else {
                            "single".to_string()
                        }
                    }),
                    "workflow": if workflow_val.is_object() { workflow_val.clone() } else { recipe_plan.clone().unwrap_or(Value::Null) },
                    "workflow_family": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("family"))
                        .cloned()
                        .unwrap_or(Value::Null),
                    "quality_goal": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("quality_goal"))
                        .cloned()
                        .unwrap_or(Value::Null),
                    "difficulty": difficulty,
                    "surface_policy": surface_policy,
                    "requires_verified_assets": requires_verified_assets,
                    "requires_qa_pass": requires_qa_pass,
                    "golden_set_id": golden_set_id,
                    "audio_quality_thresholds": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("audio_quality_thresholds"))
                        .cloned()
                        .unwrap_or(Value::Null),
                    "qa_policy_ready": qa_policy_ready,
                    "simple_surface": simple_surface,
                    "simple_goal": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("simple_goal"))
                        .cloned()
                        .unwrap_or(Value::Null),
                    "quality_tradeoff": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("quality_tradeoff"))
                        .cloned()
                        .or_else(|| recipe_meta.as_ref().and_then(|v| v.get("quality_goal")).cloned())
                        .unwrap_or(Value::Null),
                    "guide_topics": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("guide_topics"))
                        .cloned()
                        .unwrap_or(Value::Array(vec![])),
                    "surface_blockers": surface_blockers,
                    "guide_rank": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("guide_rank"))
                        .cloned()
                        .unwrap_or(Value::Null),
                    "expected_vram_tier": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("expected_vram_tier"))
                        .cloned()
                        .unwrap_or(Value::Null),
                    "expected_runtime_tier": recipe_meta
                        .as_ref()
                        .and_then(|v| v.get("expected_runtime_tier"))
                        .cloned()
                        .unwrap_or(Value::Null),
                    "effective_model_id": if model_id_resolved.is_empty() { Value::Null } else { Value::String(model_id_resolved.clone()) },
                    "effective_model_ids": requested_model_ids,
                    "required_models": required_models_out,
                    "runtime_blocks": runtime_blocks,
                    "recommended_adjustments": recommended_adjustments,
                    "estimated_vram_gb": estimated_vram_gb,
                    "resolved_device": resolved_device.clone(),
                    "resolved_overlap": resolved_overlap,
                    "resolved_segment_size": resolved_segment_size,
                    "runtime_adapter": runtime_adapter,
                    "missing_runtime_assets": missing_runtime_assets,
                    "unsupported_patch_profile": unsupported_patch_profiles,
                    "phase_fix_compatibility": phase_fix_compatibility,
                    "crossover_validation": crossover_validation,
                    "recommended_operating_profile": recommended_operating_profile,
                    "fallback_reason": fallback_reason,
                    "should_use_simple": should_use_simple,
                    "should_use_advanced": should_use_advanced
                });
                let report = serde_json::json!({
                    "can_proceed": can_proceed,
                    "errors": errors,
                    "warnings": warnings,
                    "audio": audio_info,
                    "missing_models": missing_models,
                    "torch_available": torch_available,
                    "plan": plan,
                    "resolved": {
                        "model_id": if model_id_resolved.is_empty() { Value::Null } else { Value::String(model_id_resolved) },
                        "recipe": recipe_plan.unwrap_or(Value::Null),
                        "workflow": if workflow_val.is_object() { workflow_val.clone() } else { Value::Null },
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
                        "ensemble_config": ensemble_cfg_val.cloned().unwrap_or(Value::Null),
                        "phase_fix_enabled": phase_fix_enabled,
                        "phase_fix_params": phase_fix_params,
                        "runtime_policy": req.extra.get("runtime_policy").cloned().or_else(|| req.extra.get("runtimePolicy").cloned()).unwrap_or(Value::Null),
                        "export_policy": req.extra.get("export_policy").cloned().or_else(|| req.extra.get("exportPolicy").cloned()).unwrap_or(Value::Null)
                    },
                    "memory": Value::Null
                });

                send_ok(&stdout, req.id, report)?;
            }
            "separate_audio" => {
                match start_selection_job_request(
                    &separation_manager,
                    &cfg,
                    &stdout,
                    &gpu_info,
                    &req.extra,
                ) {
                    Ok(payload) => send_ok(&stdout, req.id, payload)?,
                    Err(e) => send_err(
                        &stdout,
                        req.id,
                        "START_FAILED",
                        &format!("Failed to start separation: {e:#}"),
                    )?,
                }
            }
            "run_selection_job" => {
                match start_selection_job_request(
                    &separation_manager,
                    &cfg,
                    &stdout,
                    &gpu_info,
                    &req.extra,
                ) {
                    Ok(payload) => send_ok(&stdout, req.id, payload)?,
                    Err(e) => send_err(
                        &stdout,
                        req.id,
                        "START_FAILED",
                        &format!("Failed to start separation: {e:#}"),
                    )?,
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
            "cancel_selection_job" => {
                let job_id = req
                    .extra
                    .get("job_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| req.extra.get("selection_job_id").and_then(|v| v.as_str()));
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
                        send_err(&stdout, req.id, "NOT_FOUND", "Job not found or not running")?;
                    }
                } else {
                    send_err(&stdout, req.id, "INVALID", "job_id is required")?;
                }
            }
            "get_selection_job" => {
                let job_id = req
                    .extra
                    .get("job_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| req.extra.get("selection_job_id").and_then(|v| v.as_str()));
                if let Some(id) = job_id {
                    if let Some(snapshot) = separation_manager.snapshot(id) {
                        send_ok(&stdout, req.id, serde_json::to_value(snapshot).unwrap_or(Value::Null))?;
                    } else {
                        send_err(&stdout, req.id, "NOT_FOUND", "Selection job not found")?;
                    }
                } else {
                    send_err(&stdout, req.id, "INVALID", "job_id is required")?;
                }
            }
            "list_selection_jobs" => {
                let snapshots = separation_manager.list();
                send_ok(
                    &stdout,
                    req.id,
                    serde_json::to_value(snapshots).unwrap_or(Value::Array(vec![])),
                )?;
            }
            "export_selection_job" => {
                let job_id = req
                    .extra
                    .get("job_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| req.extra.get("selection_job_id").and_then(|v| v.as_str()));
                let export_path = req
                    .extra
                    .get("export_path")
                    .and_then(|v| v.as_str())
                    .or_else(|| req.extra.get("destination_dir").and_then(|v| v.as_str()));
                match (job_id, export_path) {
                    (Some(id), Some(path_str)) if !path_str.trim().is_empty() => {
                        match export_selection_job_outputs(
                            &separation_manager,
                            id,
                            &PathBuf::from(path_str),
                        ) {
                            Ok(payload) => send_ok(&stdout, req.id, payload)?,
                            Err(error) => send_err(
                                &stdout,
                                req.id,
                                "EXPORT_FAILED",
                                &format!("{error:#}"),
                            )?,
                        }
                    }
                    (None, _) => send_err(&stdout, req.id, "INVALID", "job_id is required")?,
                    (_, None) => send_err(
                        &stdout,
                        req.id,
                        "INVALID",
                        "export_path is required",
                    )?,
                    (_, Some(_)) => send_err(
                        &stdout,
                        req.id,
                        "INVALID",
                        "export_path is required",
                    )?,
                }
            }
            "discard_selection_job" => {
                let job_id = req
                    .extra
                    .get("job_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| req.extra.get("selection_job_id").and_then(|v| v.as_str()));
                if let Some(id) = job_id {
                    match separation_manager.discard(id) {
                        Ok(snapshot) => send_ok(
                            &stdout,
                            req.id,
                            serde_json::json!({
                                "job_id": id,
                                "status": "discarded",
                                "selection_job": snapshot,
                            }),
                        )?,
                        Err(error) => send_err(
                            &stdout,
                            req.id,
                            "DISCARD_FAILED",
                            &format!("{error:#}"),
                        )?,
                    }
                } else {
                    send_err(&stdout, req.id, "INVALID", "job_id is required")?;
                }
            }
            "detect_playback_devices" => {
                match detect_playback_devices_native() {
                    Ok(devices) => {
                        send_ok(
                            &stdout,
                            req.id,
                            serde_json::to_value(devices).unwrap_or(Value::Array(vec![])),
                        )?;
                    }
                    Err(error) => {
                        send_err(
                            &stdout,
                            req.id,
                            "CAPTURE_ENVIRONMENT_FAILED",
                            &format!("{error:#}"),
                        )?;
                    }
                }
            }
            "probe_playback_device_activity" => {
                let device_id = req.extra.get("device_id").and_then(|v| v.as_str());
                if device_id.is_none() {
                    send_err(&stdout, req.id, "INVALID", "device_id is required")?;
                    continue;
                }

                let timeout_ms = req
                    .extra
                    .get("timeout_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2_500);
                let min_active_rms = req
                    .extra
                    .get("min_active_rms")
                    .and_then(|v| v.as_f64())
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .unwrap_or(0.001);

                match probe_playback_device_activity_native(
                    device_id.unwrap(),
                    timeout_ms,
                    min_active_rms,
                ) {
                    Ok(probe) => {
                        send_ok(
                            &stdout,
                            req.id,
                            serde_json::to_value(probe).unwrap_or(Value::Null),
                        )?;
                    }
                    Err(error) => {
                        send_err(
                            &stdout,
                            req.id,
                            "CAPTURE_ENVIRONMENT_FAILED",
                            &format!("{error:#}"),
                        )?;
                    }
                }
            }
            "capture_playback_loopback" => {
                let capture_id = req.extra.get("capture_id").and_then(|v| v.as_str());
                let device_id = req.extra.get("device_id").and_then(|v| v.as_str());
                let output_path = req.extra.get("output_path").and_then(|v| v.as_str());

                if capture_id.is_none() || device_id.is_none() || output_path.is_none() {
                    send_err(
                        &stdout,
                        req.id,
                        "INVALID",
                        "capture_id, device_id and output_path are required",
                    )?;
                    continue;
                }

                let expected_duration_sec = req
                    .extra
                    .get("expected_duration_sec")
                    .and_then(|v| v.as_f64())
                    .filter(|value| value.is_finite() && *value > 0.0);
                let start_timeout_ms = req
                    .extra
                    .get("start_timeout_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(15_000);
                let trailing_silence_ms = req
                    .extra
                    .get("trailing_silence_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2_500);
                let min_active_rms = req
                    .extra
                    .get("min_active_rms")
                    .and_then(|v| v.as_f64())
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .unwrap_or(0.003);
                let capture_id = capture_id.unwrap().to_string();
                let device_id = device_id.unwrap().to_string();
                let output_path = output_path.unwrap().to_string();

                if playback_capture_jobs()
                    .lock()
                    .expect("capture job registry")
                    .contains_key(&capture_id)
                {
                    send_err(
                        &stdout,
                        req.id,
                        "CAPTURE_ALREADY_RUNNING",
                        "A playback capture with this id is already running.",
                    )?;
                    continue;
                }

                let cancel_flag = register_playback_capture_job(
                    &capture_id,
                    &device_id,
                    Path::new(&output_path),
                );
                let stdout_clone = stdout.clone();
                let response_id = req.id.clone();

                thread::spawn(move || {
                    let run_result = capture_playback_loopback_native(
                        &stdout_clone,
                        &capture_id,
                        cancel_flag.clone(),
                        &device_id,
                        Path::new(&output_path),
                        expected_duration_sec,
                        start_timeout_ms,
                        trailing_silence_ms,
                        min_active_rms,
                    );

                    match run_result {
                        Ok(result) => {
                            if cancel_flag.load(Ordering::SeqCst) {
                                let _ = std::fs::remove_file(&output_path);
                                set_playback_capture_job_status(
                                    &capture_id,
                                    "cancelled",
                                    Some("Capture cancelled.".to_string()),
                                );
                                let _ = send_playback_capture_progress(
                                    &stdout_clone,
                                    &capture_id,
                                    "cancelled",
                                    Some("Capture cancelled."),
                                    None,
                                    None,
                                    Some("Capture cancelled."),
                                );
                                let _ = send_err(
                                    &stdout_clone,
                                    response_id,
                                    "CAPTURE_CANCELLED",
                                    "Capture cancelled.",
                                );
                            } else {
                                set_playback_capture_job_status(&capture_id, "completed", None);
                                let _ = send_playback_capture_progress(
                                    &stdout_clone,
                                    &capture_id,
                                    "completed",
                                    Some("Captured WAV is ready for separation."),
                                    Some(1.0),
                                    Some(result.duration_sec),
                                    None,
                                );
                                let _ = send_ok(
                                    &stdout_clone,
                                    response_id,
                                    serde_json::to_value(result).unwrap_or(Value::Null),
                                );
                            }
                        }
                        Err(error) => {
                            let error_text = format!("{error:#}");
                            let cancelled = error_text.to_lowercase().contains("cancelled")
                                || cancel_flag.load(Ordering::SeqCst);
                            set_playback_capture_job_status(
                                &capture_id,
                                if cancelled { "cancelled" } else { "failed" },
                                Some(error_text.clone()),
                            );
                            let _ = send_playback_capture_progress(
                                &stdout_clone,
                                &capture_id,
                                if cancelled { "cancelled" } else { "error" },
                                if cancelled {
                                    Some("Capture cancelled.")
                                } else {
                                    Some("Playback capture failed.")
                                },
                                None,
                                None,
                                Some(&error_text),
                            );
                            let _ = send_err(
                                &stdout_clone,
                                response_id,
                                if cancelled {
                                    "CAPTURE_CANCELLED"
                                } else {
                                    "CAPTURE_FAILED"
                                },
                                &error_text,
                            );
                        }
                    }

                    finish_playback_capture_job(&capture_id);
                });

                continue;
            }
            "cancel_playback_capture" => {
                let capture_id = req.extra.get("capture_id").and_then(|v| v.as_str());
                let cancelled = cancel_playback_capture_request(capture_id);
                send_ok(
                    &stdout,
                    req.id,
                    serde_json::json!({
                        "success": cancelled,
                        "capture_id": capture_id,
                    }),
                )?;
            }
            "get_playback_capture_status" => {
                let capture_id = req.extra.get("capture_id").and_then(|v| v.as_str());
                send_ok(&stdout, req.id, snapshot_playback_capture_jobs(capture_id))?;
            }
            "get_workflows" => {
                let workflows_json = load_catalog_workflows(&cfg)?;
                send_ok(&stdout, req.id, workflows_json)?;
            }
            "check-preset-models" => {
                let preset_mappings = req
                    .extra
                    .get("preset_mappings")
                    .and_then(|v| v.as_object())
                    .cloned()
                    .unwrap_or_default();
                let models_json = load_models_with_guide_overrides(&cfg)?;

                let mut availability = Map::new();
                for (preset_name, model_id_val) in preset_mappings {
                    let installed = model_id_val
                        .as_str()
                        .map(|id| {
                            find_model_object(&models_json, id)
                                .map(|model_obj| any_model_file_exists(&cfg.models_dir, id, &model_obj))
                                .unwrap_or(false)
                        })
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
                let models_json = load_models_with_guide_overrides(&cfg)?;
                let models = models_json
                    .get("models")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let found = models
                    .into_iter()
                    .find(|m| m.get("id").and_then(|v| v.as_str()) == Some(model_id));
                if let Some(m) = found {
                    let manifest = resolve_model_download_manifest(&cfg, model_id, &m);
                    let installation = resolve_installation_status(&cfg, &manifest);
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
                            "card_metrics": m.get("card_metrics").cloned().unwrap_or(Value::Null),
                            "catalog_status": m.get("catalog_status").cloned().unwrap_or(Value::Null),
                            "metrics_status": m.get("metrics_status").cloned().unwrap_or(Value::Null),
                            "links": m.get("links").cloned().unwrap_or(Value::Null),
                            "runtime": m.get("runtime").cloned().unwrap_or(Value::Null),
                            "install": m.get("install").cloned().unwrap_or(Value::Null),
                            "quality_role": m.get("quality_role").cloned().unwrap_or(Value::Null),
                            "workflow_roles": m.get("workflow_roles").cloned().unwrap_or(Value::Array(vec![])),
                            "best_for": m.get("best_for").cloned().unwrap_or(Value::Array(vec![])),
                            "artifacts_risk": m.get("artifacts_risk").cloned().unwrap_or(Value::Null),
                            "vram_profile": m.get("vram_profile").cloned().unwrap_or(Value::Null),
                            "chunk_overlap_policy": m.get("chunk_overlap_policy").cloned().unwrap_or(Value::Null),
                            "workflow_groups": m.get("workflow_groups").cloned().unwrap_or(Value::Array(vec![])),
                            "quality_axes": m.get("quality_axes").cloned().unwrap_or(Value::Null),
                            "content_fit": m.get("content_fit").cloned().unwrap_or(Value::Array(vec![])),
                            "operating_profiles": m.get("operating_profiles").cloned().unwrap_or(Value::Null),
                            "download": serde_json::to_value(&manifest).unwrap_or(Value::Null),
                            "installation": serde_json::to_value(&installation).unwrap_or(Value::Null),
                            "installed": installation.installed
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_source(
        url: &str,
        provider: &str,
        resolver: &str,
        locator: Option<Value>,
    ) -> ResolvedArtifactSource {
        ResolvedArtifactSource {
            url: url.to_string(),
            host: url_host(url).unwrap_or_default(),
            provider: provider.to_string(),
            resolver: resolver.to_string(),
            locator,
            channel: "upstream".to_string(),
            priority: 0,
            auth: "public".to_string(),
            verified: false,
        }
    }

    #[test]
    fn resolves_huggingface_source_urls_without_rewriting() {
        let source = sample_source(
            "https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/model.ckpt",
            "huggingface",
            "huggingface_resolve",
            Some(serde_json::json!({
                "repo": "pcunwa/Mel-Band-Roformer-Inst",
                "revision": "main",
                "file_path": "model.ckpt"
            })),
        );
        let resolved = resolve_download_source_url(&source).expect("resolve huggingface source");
        assert_eq!(resolved, source.url);
    }

    #[test]
    fn resolves_google_drive_file_from_locator() {
        let source = sample_source(
            "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing",
            "google_drive",
            "google_drive_file",
            Some(serde_json::json!({
                "file_id": "1AbCdEfGhIjKlMnOp"
            })),
        );
        let resolved = resolve_download_source_url(&source).expect("resolve google drive file");
        assert!(
            resolved.contains("drive.usercontent.google.com/download"),
            "unexpected resolved drive url: {resolved}"
        );
        assert!(
            resolved.contains("id=1AbCdEfGhIjKlMnOp"),
            "expected drive id in resolved url: {resolved}"
        );
    }

    #[test]
    fn resolves_google_drive_folder_entry_from_locator() {
        let source = sample_source(
            "https://drive.google.com/drive/folders/abcdef123456",
            "google_drive",
            "google_drive_folder_entry",
            Some(serde_json::json!({
                "folder_id": "abcdef123456",
                "file_id": "1Zyx987",
                "expected_name": "model.ckpt"
            })),
        );
        let resolved =
            resolve_download_source_url(&source).expect("resolve google drive folder entry");
        assert!(resolved.contains("id=1Zyx987"));
    }

    #[test]
    fn rejects_proton_share_entries_without_download_locator() {
        let source = sample_source(
            "https://drive.proton.me/urls/5XM3PR1M7G#F3UhCU8RDGhX",
            "proton",
            "proton_share_entry",
            Some(serde_json::json!({
                "share_id": "5XM3PR1M7G",
                "expected_name": "model.ckpt"
            })),
        );
        let error = resolve_download_source_url(&source).expect_err("expected missing locator failure");
        assert!(
            format!("{error:#}").contains("download_url"),
            "unexpected proton resolver error: {error:#}"
        );
    }

    #[test]
    fn resolves_proton_share_entries_with_download_locator() {
        let source = sample_source(
            "https://drive.proton.me/urls/5XM3PR1M7G#F3UhCU8RDGhX",
            "proton",
            "proton_share_entry",
            Some(serde_json::json!({
                "share_id": "5XM3PR1M7G",
                "download_url": "https://proton.fake/download/model.ckpt",
                "expected_name": "model.ckpt"
            })),
        );
        let resolved = resolve_download_source_url(&source).expect("resolve proton share entry");
        assert_eq!(resolved, "https://proton.fake/download/model.ckpt");
    }
}
