# StemSep Backend API v1 (stdio JSON-lines)

This document defines the **versioned** contract between the Electron main process (bridge supervisor) and the backend process.

- Transport: **stdin/stdout**, one JSON object per line ("JSONL")
- Encoding: UTF-8
- Direction:
  - Electron → backend: **request** lines (contain `command` + optional `id`)
  - Backend → Electron: either
    - **response** lines (contain `success` + optional `id`), or
    - **event** lines (contain `type`, never `success`)

This contract is intentionally strict to support a **hard quality policy** and deterministic releases.

## 1. Message Shapes

### 1.1 Request

```json
{
  "command": "get_models",
  "id": 123,
  "...": "other command-specific fields live at top-level"
}
```

Notes:
- `id` is required for request/response correlation.
- This protocol does not require an `api_version` field today; versioning is maintained by the app/repo.
- Command parameters are sent as **top-level fields** (historical compatibility with the Python bridge).

### 1.2 Response

```json
{
  "id": 123,
  "success": true,
  "data": { }
}
```

Error:

```json
{
  "id": 123,
  "success": false,
  "error": "Human readable error"
}
```

### 1.3 Event

```json
{
  "type": "separation_progress",
  "job_id": "...",
  "progress": 42.0,
  "message": "..."
}
```

Events MUST NOT include `success`.

## 2. Startup

### 2.1 `bridge_ready` event

Emitted exactly once when backend is ready.

```json
{
  "type": "bridge_ready",
  "capabilities": ["models", "separation", "recipes", "gpu"],
  "models_count": 123,
  "recipes_count": 45
}
```

## 3. Commands

### 3.1 `ping`
- Request: `{ "command": "ping", "id": N }`
- Response `data`:

```json
{ "status": "ok", "timestamp": 1730000000.123 }
```

### 3.2 `get_models`
Returns the model registry, including installation status.

- Request: `{ "command": "get_models", "id": N }`
- Response `data`: array of objects (shape is intentionally permissive)

Required fields:
- `id` (string)
- `name` (string)
- `architecture` (string)
- `installed` (boolean)

Recommended fields:
- `stems` (string[])
- `recommended_settings` (object)
- `links` / `checkpoint_url` / `config_url`

### 3.3 `get_recipes`
Returns recipe definitions used by the UI to build “smart presets”.

- Request: `{ "command": "get_recipes", "id": N }`
- Response `data`: array of recipe objects.
  - `id`, `name`, `description`, `target`, `steps[]`, `algorithm`, `defaults`

### 3.4 `get-gpu-devices`
Returns device enumeration and recommended profile.

- Request: `{ "command": "get-gpu-devices", "id": N }`
- Response `data`:

```json
{
  "gpus": [
    { "id": "cuda:0", "index": 0, "name": "...", "memory_gb": 8.0 }
  ],
  "recommended_profile": { "profile_name": "cuda", "vram_gb": 8.0, "settings": {} }
}
```

### 3.5 `separation_preflight`
Validates: file readability, model availability, parameter defaults, and whether the job can proceed.

- Request: `{ "command": "separation_preflight", "id": N, ... }`

Response `data` must include:
- `can_proceed` (boolean)
- `errors` (string[])
- `warnings` (string[])
- `resolved` (object) with final chosen values (`device`, `overlap`, `segment_size`, etc)

### 3.6 `separate_audio`
Starts a job and emits progress events.

- Request: `{ "command": "separate_audio", "id": N, ... }`
- Response `data`: `{ "job_id": "...", "status": "started" }`

Progress events:
- `separation_progress`: `{ type, job_id, progress (0..100), message, device? }`
- `separation_complete`: `{ type, job_id, output_files: { stem: path } }`

### 3.7 `cancel_job`
Cancels an active job.

- Request: `{ "command": "cancel_job", "id": N, "job_id": "..." }`

## 4. Hard Quality Policy Hooks (v1)

v1 introduces a way to run user-provided regression without bundling audio in the repo.

Recommended future commands (not required for v1 stub):
- `quality_baseline_create`
- `quality_compare`

Both should produce a JSON manifest with:
- model ids + hashes
- config hash
- output stem hashes (optionally tolerant/feature-based)
- metrics summary
