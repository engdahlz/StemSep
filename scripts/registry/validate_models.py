#!/usr/bin/env python3
"""
Model URL Validator - Verifies all model configurations without full download.
- HEAD request for checkpoints (verifies file exists)
- Full download for configs (small, ~1KB, verifies YAML validity)
"""

import json
import asyncio
import aiohttp
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import sys

MODELS_DIR = Path(__file__).parent / "StemSepApp" / "assets" / "models"

async def check_url(session: aiohttp.ClientSession, url: str, download_content: bool = False) -> Tuple[bool, str, bytes | None]:
    """Check if URL is valid. Returns (success, message, content)."""
    try:
        if download_content:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    return True, f"OK ({len(content)} bytes)", content
                return False, f"HTTP {resp.status}", None
        else:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=15), allow_redirects=True) as resp:
                if resp.status == 200:
                    size = resp.headers.get('content-length', 'unknown')
                    return True, f"OK ({size} bytes)", None
                # Some servers don't support HEAD, try GET with range
                if resp.status == 405:
                    async with session.get(url, headers={'Range': 'bytes=0-1'}, timeout=aiohttp.ClientTimeout(total=15)) as get_resp:
                        if get_resp.status in (200, 206):
                            return True, "OK (verified via GET)", None
                        return False, f"HTTP {get_resp.status}", None
                return False, f"HTTP {resp.status}", None
    except asyncio.TimeoutError:
        return False, "TIMEOUT", None
    except Exception as e:
        return False, f"ERROR: {str(e)[:50]}", None

async def validate_model(session: aiohttp.ClientSession, model_file: Path) -> Dict:
    """Validate a single model's URLs."""
    try:
        with open(model_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        return {"file": model_file.name, "status": "INVALID_JSON", "error": str(e)}
    
    result = {
        "file": model_file.name,
        "id": config.get("id", "unknown"),
        "name": config.get("name", "unknown"),
    }
    
    links = config.get("links", {})
    
    # Check checkpoint URL
    checkpoint_url = links.get("checkpoint")
    if checkpoint_url:
        ok, msg, _ = await check_url(session, checkpoint_url, download_content=False)
        result["checkpoint"] = {"url": checkpoint_url, "ok": ok, "msg": msg}
    else:
        result["checkpoint"] = {"ok": False, "msg": "MISSING"}
    
    # Check and download config URL (small file)
    config_url = links.get("config")
    if config_url:
        ok, msg, content = await check_url(session, config_url, download_content=True)
        result["config"] = {"url": config_url, "ok": ok, "msg": msg}
        
        if ok and content:
            # Try to parse as YAML
            try:
                parsed = yaml.safe_load(content)
                result["config"]["yaml_valid"] = True
                # Extract key info
                if parsed:
                    result["config"]["keys"] = list(parsed.keys())[:5]
            except Exception as e:
                result["config"]["yaml_valid"] = False
                result["config"]["yaml_error"] = str(e)[:100]
    else:
        result["config"] = {"ok": False, "msg": "MISSING"}
    
    return result

async def main():
    print("=" * 60)
    print("Model URL Validator - StemSep V3")
    print("=" * 60)
    
    if not MODELS_DIR.exists():
        print(f"ERROR: Models directory not found: {MODELS_DIR}")
        sys.exit(1)
    
    model_files = sorted(MODELS_DIR.glob("*.json"))
    print(f"\nFound {len(model_files)} model configuration files\n")
    
    # Filter to only check new models (optional)
    new_models = [
        "mel-roformer-drumsep-5stem.json",
        "mel-roformer-decrowd.json", 
        "mel-roformer-debleed.json",
        "mel-roformer-dereverb-anvuew-v2.json",
        "gabox-voc-fv3.json",
        "gabox-karaoke.json",
        "gabox-inst-v7n.json",
        "gabox-inst-fv7-plus.json",
        "gabox-small-inst.json",
        "gabox-denoise-debleed.json",
        "sucial-debreath.json",
        "unwa-hyperace.json",
        "unwa-small-melrofo.json",
    ]
    
    # Check if --all flag to validate all models
    check_all = "--all" in sys.argv
    
    if check_all:
        files_to_check = model_files
        print("Validating ALL models...")
    else:
        files_to_check = [f for f in model_files if f.name in new_models]
        print(f"Validating {len(files_to_check)} NEW models (use --all for all)")
    
    print("-" * 60)
    
    connector = aiohttp.TCPConnector(limit=5)  # Limit concurrent connections
    async with aiohttp.ClientSession(connector=connector) as session:
        results = []
        for i, model_file in enumerate(files_to_check, 1):
            print(f"[{i}/{len(files_to_check)}] Checking {model_file.name}...", end=" ", flush=True)
            result = await validate_model(session, model_file)
            results.append(result)
            
            # Print inline status
            ckpt_ok = result.get("checkpoint", {}).get("ok", False)
            cfg_ok = result.get("config", {}).get("ok", False)
            yaml_ok = result.get("config", {}).get("yaml_valid", False)
            
            status = []
            status.append("✅ ckpt" if ckpt_ok else "❌ ckpt")
            status.append("✅ cfg" if cfg_ok else "❌ cfg")
            if cfg_ok:
                status.append("✅ yaml" if yaml_ok else "❌ yaml")
            
            print(" | ".join(status))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    failed = []
    passed = []
    
    for r in results:
        ckpt_ok = r.get("checkpoint", {}).get("ok", False)
        cfg_ok = r.get("config", {}).get("ok", False)
        yaml_ok = r.get("config", {}).get("yaml_valid", True)  # Missing config still counts
        
        if ckpt_ok and cfg_ok and yaml_ok:
            passed.append(r)
        else:
            failed.append(r)
    
    print(f"\n✅ PASSED: {len(passed)}")
    print(f"❌ FAILED: {len(failed)}")
    
    if failed:
        print("\n--- FAILED MODELS ---")
        for r in failed:
            print(f"\n{r['file']} ({r['name']})")
            ckpt = r.get("checkpoint", {})
            cfg = r.get("config", {})
            
            if not ckpt.get("ok"):
                print(f"  Checkpoint: {ckpt.get('msg', 'MISSING')}")
                if ckpt.get("url"):
                    print(f"    URL: {ckpt['url'][:80]}...")
            
            if not cfg.get("ok"):
                print(f"  Config: {cfg.get('msg', 'MISSING')}")
                if cfg.get("url"):
                    print(f"    URL: {cfg['url'][:80]}...")
            elif not cfg.get("yaml_valid", True):
                print(f"  YAML: Invalid - {cfg.get('yaml_error', 'unknown')}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
