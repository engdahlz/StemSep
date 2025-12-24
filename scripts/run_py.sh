#!/usr/bin/env sh
# -----------------------------------------------------------------------------
# run_py.sh
#
# Cross-platform helper to run a Python script in this repo WITHOUT relying on
# shell-based virtualenv activation.
#
# Behavior:
#   1) If --python is provided, use it.
#   2) Else if a preferred venv exists, use its python:
#        - StemSepApp/.venv  (default)
#        - ./.venv           (if --venv Root)
#      (Supports both Unix layout: bin/python and Windows layout: Scripts/python.exe)
#   3) Else use python/python3 from PATH.
#
# Always runs from repo root so relative paths are predictable.
#
# Usage:
#   ./scripts/run_py.sh ./validate_model_registry.py
#   ./scripts/run_py.sh ./scripts/download_configs.py --help
#   ./scripts/run_py.sh --venv Root ./validate_model_registry.py
#   ./scripts/run_py.sh --venv None ./validate_model_registry.py
#   ./scripts/run_py.sh --python /usr/bin/python3 ./validate_model_registry.py
#
# Notes:
# - This script intentionally does NOT "activate" any venv.
# - It simply selects a python executable and runs it.
# - Works in sh (POSIX), avoids bash-specific features.
# -----------------------------------------------------------------------------

set -eu

usage() {
  cat <<'USAGE'
run_py.sh - run a repo python script without relying on venv activation

Usage:
  ./scripts/run_py.sh [--venv StemSepApp|Root|None] [--python /path/to/python] <script> [args...]

Options:
  --venv, -v     Preferred venv location (default: StemSepApp)
                 StemSepApp -> StemSepApp/.venv
                 Root       -> ./.venv
                 None       -> do not use a venv even if present
  --python, -p   Explicit python executable path to use (overrides --venv)
  --help, -h     Show help

Examples:
  ./scripts/run_py.sh ./validate_model_registry.py
  ./scripts/run_py.sh -v StemSepApp ./scripts/download_configs.py --help
  ./scripts/run_py.sh -v None ./validate_model_registry.py
  ./scripts/run_py.sh -p /usr/bin/python3 ./validate_model_registry.py
USAGE
}

# Resolve repo root: this file is at <repo_root>/scripts/run_py.sh
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

VENVDIR_PREF="StemSepApp"
PY_EXPLICIT=""

# Parse flags (POSIX sh)
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -v|--venv)
      if [ $# -lt 2 ]; then
        echo "ERROR: --venv requires a value (StemSepApp|Root|None)" >&2
        exit 2
      fi
      VENVDIR_PREF="$2"
      shift 2
      ;;
    -p|--python)
      if [ $# -lt 2 ]; then
        echo "ERROR: --python requires a path" >&2
        exit 2
      fi
      PY_EXPLICIT="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [ $# -lt 1 ]; then
  usage >&2
  exit 2
fi

TARGET_SCRIPT="$1"
shift

# Convert a possibly-relative script path to an absolute path anchored at repo root
# (if already absolute, keep it)
case "$TARGET_SCRIPT" in
  /*)
    TARGET_SCRIPT_ABS="$TARGET_SCRIPT"
    ;;
  *)
    TARGET_SCRIPT_ABS="${REPO_ROOT}/${TARGET_SCRIPT}"
    ;;
esac

if [ ! -f "$TARGET_SCRIPT_ABS" ]; then
  echo "ERROR: Script not found: $TARGET_SCRIPT_ABS" >&2
  exit 2
fi

resolve_python_from_venv() {
  # Echo a python path if found, else echo nothing
  _venv_path="$1"

  # Unix venv layout
  if [ -f "${_venv_path}/bin/python" ]; then
    echo "${_venv_path}/bin/python"
    return 0
  fi

  # Windows venv layout (in case you're using Git Bash / MSYS / Cygwin)
  if [ -f "${_venv_path}/Scripts/python.exe" ]; then
    echo "${_venv_path}/Scripts/python.exe"
    return 0
  fi

  # Some environments may have python without .exe
  if [ -f "${_venv_path}/Scripts/python" ]; then
    echo "${_venv_path}/Scripts/python"
    return 0
  fi

  return 0
}

PYTHON_EXE=""

if [ -n "$PY_EXPLICIT" ]; then
  if [ ! -f "$PY_EXPLICIT" ]; then
    echo "ERROR: Explicit --python not found: $PY_EXPLICIT" >&2
    exit 2
  fi
  PYTHON_EXE="$PY_EXPLICIT"
else
  if [ "$VENVDIR_PREF" != "None" ]; then
    VENV_PATH=""
    case "$VENVDIR_PREF" in
      StemSepApp)
        VENV_PATH="${REPO_ROOT}/StemSepApp/.venv"
        ;;
      Root)
        VENV_PATH="${REPO_ROOT}/.venv"
        ;;
      *)
        echo "ERROR: Invalid --venv value: $VENVDIR_PREF (expected StemSepApp|Root|None)" >&2
        exit 2
        ;;
    esac

    if [ -d "$VENV_PATH" ]; then
      PYTHON_EXE="$(resolve_python_from_venv "$VENV_PATH" || true)"
    fi
  fi

  if [ -z "$PYTHON_EXE" ]; then
    # Fall back to PATH
    if command -v python >/dev/null 2>&1; then
      PYTHON_EXE="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
      PYTHON_EXE="$(command -v python3)"
    else
      echo "ERROR: Could not find python. Install Python or create a venv at StemSepApp/.venv or ./.venv." >&2
      exit 2
    fi
  fi
fi

echo "Repo root : $REPO_ROOT"
echo "Python    : $PYTHON_EXE"
echo "Script    : $TARGET_SCRIPT_ABS"
if [ $# -gt 0 ]; then
  echo "Args      : $*"
fi

cd "$REPO_ROOT"
exec "$PYTHON_EXE" "$TARGET_SCRIPT_ABS" "$@"
