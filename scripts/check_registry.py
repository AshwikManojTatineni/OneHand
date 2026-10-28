"""Phase 0: check each Hugging Face checkpoint in the registry for existence, gating and license.

Writes registry/checkpoints_status.yaml; non-HF sources (gs://, torch.hub, github:) are left for manual checks.
"""
import json
import pathlib
import sys
import urllib.error
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def hf_info(repo):
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"_error": e.code}
    except Exception as e:  # network failures are recorded, not fatal
        return {"_error": str(e)}


def main():
    reg = yaml.safe_load((ROOT / "registry/checkpoints.yaml").read_text())
    out = {}
    for route, entries in reg.items():
        out[route] = []
        for e in entries:
            row = dict(e)
            repo = e["repo"]
            if ":" in repo or repo.startswith("gs://"):
                row["status"] = "manual_check"
            else:
                info = hf_info(repo)
                if "_error" in info:
                    row["status"] = "missing" if info["_error"] in (401, 404) else f"error:{info['_error']}"
                else:
                    files = {s["rfilename"] for s in info.get("siblings", [])}
                    want = e.get("file")
                    row["status"] = "gated" if info.get("gated") else "available"
                    if want and want not in files:
                        row["status"] = "file_missing"
                    row["license"] = (info.get("cardData") or {}).get("license", "undeclared")
                    row["last_modified"] = info.get("lastModified")
            out[route].append(row)
            print(f"{route:18s} {e['id']:32s} {row['status']:14s} {row.get('license', '')}")
    (ROOT / "registry/checkpoints_status.yaml").write_text(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    sys.exit(main())
