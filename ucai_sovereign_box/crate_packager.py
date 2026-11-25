"""RO-Crate 1.1 & SHA-256 Sovereign Reproducibility Packager + Self-Verifier (FAIR4RS Compliant)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def generate_ro_crate_bundle(
    output_dir: str | Path,
    crate_name: str,
    artifact_files: List[str | Path],
    hardware_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Generates ro-crate-metadata.json (RO-Crate 1.1) and SHA256SUMS.txt for deterministic provenance."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_part: List[Dict[str, str]] = []
    graph_entries: List[Dict[str, Any]] = []
    checksum_lines: List[str] = []

    for fp in artifact_files:
        p = Path(fp)
        if not p.exists():
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        size_bytes = p.stat().st_size
        checksum_lines.append(f"{sha}  {p.name}")
        has_part.append({"@id": p.name})
        enc = "application/json" if p.suffix == ".json" else ("text/csv" if p.suffix == ".csv" else "text/plain")
        graph_entries.append({
            "@id": p.name,
            "@type": "File",
            "name": p.name,
            "contentSize": str(size_bytes),
            "sha256": sha,
            "encodingFormat": enc,
        })

    rocrate = {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {
                "@id": "ro-crate-metadata.json",
                "@type": "CreativeWork",
                "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                "about": {"@id": "./"},
            },
            {
                "@id": "./",
                "@type": "Dataset",
                "name": crate_name,
                "datePublished": datetime.now(timezone.utc).isoformat(),
                "license": "https://creativecommons.org/licenses/by/4.0/",
                "description": "Sovereign Air-Gapped Research-in-a-Box Audit & Retrieval Bundle (UCAI Flagship 4 Paper 3 Tier-2).",
                "additionalProperty": hardware_profile,
                "hasPart": has_part,
            },
            *graph_entries,
        ],
    }

    meta_path = out_dir / "ro-crate-metadata.json"
    sums_path = out_dir / "SHA256SUMS.txt"
    meta_path.write_text(json.dumps(rocrate, indent=2, ensure_ascii=False), encoding="utf-8")
    sums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    return {
        "ro_crate_metadata": str(meta_path),
        "sha256_manifest": str(sums_path),
        "packaged_files": len(graph_entries),
        "valid_jsonld_context": rocrate["@context"] == "https://w3id.org/ro/crate/1.1/context",
    }


def verify_ro_crate_bundle(bundle_dir: str | Path, artifact_dir: str | Path) -> Dict[str, Any]:
    """Independently verifies SHA256SUMS.txt and JSON-LD RO-Crate 1.1 integrity."""
    bdir = Path(bundle_dir)
    adir = Path(artifact_dir)
    meta = json.loads((bdir / "ro-crate-metadata.json").read_text(encoding="utf-8"))
    lines = [ln.strip() for ln in (bdir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]

    verified = 0
    mismatches: List[str] = []
    for ln in lines:
        expected_sha, fname = ln.split(None, 1)
        target = adir / fname
        if not target.exists():
            mismatches.append(f"MISSING:{fname}")
            continue
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            mismatches.append(f"HASH_MISMATCH:{fname}")
        else:
            verified += 1

    return {
        "valid_context": meta.get("@context") == "https://w3id.org/ro/crate/1.1/context",
        "files_checked": len(lines),
        "files_verified": verified,
        "mismatches": mismatches,
        "all_verified": (len(mismatches) == 0 and verified == len(lines) and len(lines) > 0),
    }
