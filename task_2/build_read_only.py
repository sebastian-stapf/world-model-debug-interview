"""Maintainer utility: refresh the static pack from explicit candidate files.

This only copies existing source/evidence and reads a short archive excerpt.
It does not import the model, train, sample or generate new model results.
"""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


TASK = Path(__file__).resolve().parent
SOURCE = TASK / "hands_on"
DESTINATION = TASK / "read_only"
EVIDENCE = [
    "diagnostics.png", "generations.png", "rollout.gif", "transitions.png",
    "metrics.json", "history.json", "provenance.json", "training_summary.json",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    provenance = json.loads((SOURCE / "assets/provenance.json").read_text())
    for name, expected in provenance["source_sha256"].items():
        if digest(SOURCE / "world_model" / name) != expected:
            raise RuntimeError(f"Source no longer matches the recorded run: {name}")
    origins = {}
    sources = sorted((SOURCE / "world_model").glob("*.py"))
    sources += [SOURCE / "scripts" / name for name in ["diagnose.py", "make_data.py"]]
    for path in sources:
        original = path.relative_to(SOURCE)
        destination = DESTINATION / "source" / original
        legacy = destination.with_suffix(".py.txt")
        if legacy.exists() and digest(legacy) != digest(path):
            raise RuntimeError(f"Legacy source has local changes: {legacy}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        legacy.unlink(missing_ok=True)
        origins[str(destination.relative_to(DESTINATION))] = str(original)
    (DESTINATION / "evidence").mkdir(exist_ok=True)
    for name in EVIDENCE:
        shutil.copyfile(SOURCE / "assets" / name, DESTINATION / "evidence" / name)
        origins["evidence/" + name] = "assets/" + name
    shutil.copyfile(SOURCE / "configs/default.json", DESTINATION / "config.json")
    origins["config.json"] = "configs/default.json"
    shutil.copyfile(SOURCE / "SPEC.md", DESTINATION / "SPEC.md")
    origins["SPEC.md"] = "SPEC.md"
    # The same evidence guide, with paths relative to the offline pack.
    evidence_guide = (SOURCE / "EVIDENCE.md").read_text().replace("assets/", "evidence/")
    evidence_guide = evidence_guide.replace(
        "See [`reports/RUN.md`](reports/RUN.md) for provenance.",
        "See [provenance.json](evidence/provenance.json) for provenance.",
    )
    (DESTINATION / "EVIDENCE.md").write_text(evidence_guide)

    archive_path = SOURCE / "data/tiny_pong.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        frames, actions = archive["frames"], archive["actions"]
        rows = []
        for t in range(11):
            centre = float(np.flatnonzero(frames[0, t, 0, :, 1] == 255).mean())
            action = int(actions[0, t])
            rows.append(f"| 0 | {t} | {centre:g} | {action} | {('STAY', 'UP', 'DOWN')[action]} |")
    excerpt = """# Raw episode excerpt

These are observations and action labels from episode 0 of the original training
archive. The paddle centre is measured from the white paddle pixels in each raw
frame. Action a[t] is issued after observing frame t and produces frame t+1.
Consecutive rows let you inspect the recorded motion without loading an archive.

| Episode | Frame t | Paddle centre row | Raw action a[t] | Name |
| --- | --- | --- | --- | --- |
""" + "\n".join(rows) + "\n\n"
    excerpt += "The contact sheet in the prompt shows dataset-loaded transitions; this table\ncontains the raw logged observations and actions. No model prediction is used.\n\n"
    excerpt += f"Archive SHA-256: `{digest(archive_path)}`\n"
    (DESTINATION / "evidence/episode_excerpt.md").write_text(excerpt)
    files = sorted(
        p for p in DESTINATION.rglob("*")
        if p.is_file() and p.name not in {"MANIFEST.json", ".DS_Store"}
        and "__pycache__" not in p.parts and p.suffix != ".pyc"
    )
    manifest = {
        "description": "Static candidate pack; copied from task_2/hands_on with no corrections.",
        "origins": origins,
        "derived_from": {"EVIDENCE.md": "EVIDENCE.md (local links adjusted)"},
        "files": {str(p.relative_to(DESTINATION)): digest(p) for p in files},
    }
    (DESTINATION / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output = TASK.parent / "outputs/task-2-read-only.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files + [DESTINATION / "MANIFEST.json"]):
            bundle.write(path, Path("read_only") / path.relative_to(DESTINATION))
    print(f"Refreshed {DESTINATION} and {output}")


if __name__ == "__main__":
    main()
