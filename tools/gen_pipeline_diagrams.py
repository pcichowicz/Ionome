"""
Draws the two pipeline diagrams in docs/assets/diagrams/ from the stages the code actually registers, so the docs
cannot drift from build_pipeline() again (tests/test_docs.py fails when they differ).

    python tools/gen_pipeline_diagrams.py            # rewrite both SVGs
    python tools/gen_pipeline_diagrams.py --check    # exit 1 if they are out of date

Colours (cohort diagram): gray = same stage class and mode as standards_only, coral = a different class for cohort mode,
teal = stage that standards_only does not have.
"""
from __future__ import annotations

import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "assets" / "diagrams"
# Layout in SVG pixels: boxes per row, left margin, box spacing, box size, top margin, row spacing
COLS, X0, DX, BOX_W, BOX_H, Y0, DY = 3, 40, 206, 186, 36, 60, 86


def _build(profile: str):
    """The real stage list for a profile, built from the shipped template with an injected (file-less) context."""
    import tempfile
    import pandas as pd
    import yaml
    from analysis.context import LCMSContext
    from analysis.pipeline_orchestrator import build_pipeline
    from core.config_schema import ExperimentConfig

    d = yaml.safe_load((ROOT / "config" / "experiments" / "template.yaml").read_text())
    d["study"] = {"id": "docs", "polarity": "negative", "instrument": "Orbitrap"}
    d["files"] = {"sample_metadata": "docs_sample_metadata.csv"}
    d["dataset_profile"] = profile
    cfg = ExperimentConfig(**d)
    tmp = Path(tempfile.mkdtemp(prefix="ionome-docs-"))
    ctx = LCMSContext(study_id="docs", experiment_id=0, config=cfg, results_dir=tmp / "r", mzml_dir=tmp / "raw",
                      sample_metadata=pd.DataFrame({"sample_id": []}), mzml_file_paths=[])
    return build_pipeline(cfg, ctx)._stages


def stage_table() -> dict[str, list[tuple[str, str]]]:
    """{profile: [(stage_name, status)]}; status is 'same' | 'modified' | 'new' | 'plain' (standards profile)."""
    std = _build("standards_only")
    coh = _build("cohort_with_qc")
    # a stage also counts as cohort-specific when the same class runs in a different mode (blank_qc)
    std_cls = {s._name: (type(s), getattr(s, "mode", None)) for s in std}
    out = {"standards_only": [(s._name, "plain") for s in std], "cohort_with_qc": []}
    for s in coh:
        if s._name not in std_cls:
            status = "new"
        elif (type(s), getattr(s, "mode", None)) == std_cls[s._name]:
            status = "same"
        else:
            status = "modified"
        out["cohort_with_qc"].append((s._name, status))
    return out


# status -> (box CSS class, label CSS class)
_CLASS = {"plain": ("box", "lbl"), "same": ("box", "lbl"), "modified": ("coral", "lbl-strong"), "new": ("teal", "lbl-strong")}


def _svg(title: str, desc: str, legend: str, stages: list[tuple[str, str]]) -> str:
    """One diagram as SVG text: boxes laid out in a snake (left to right, then right to left) with arrows between consecutive stages."""
    rows = [stages[i:i + COLS] for i in range(0, len(stages), COLS)]
    height = Y0 + (len(rows) - 1) * DY + BOX_H + 64
    parts = [
        f'<svg width="100%" viewBox="0 0 680 {height}" xmlns="http://www.w3.org/2000/svg" role="img">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(desc)}</desc>",
        '<defs>',
        '<marker id="arrow2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#5f5e5a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>',
        '</defs>',
        '<style>',
        '  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }',
        '  .lbl { font-size: 12px; fill: #2c2c2a; }',
        '  .lbl-strong { font-size: 12px; font-weight: 600; }',
        '  .box   { fill: #f1efe8; stroke: #888780; stroke-width: 0.5; }',
        '  .coral { fill: #F0997B; stroke: #993C1D; stroke-width: 0.5; }',
        '  .coral text { fill: #4A1B0C; }',
        '  .teal  { fill: #5DCAA5; stroke: #0F6E56; stroke-width: 0.5; }',
        '  .teal text { fill: #04342C; }',
        '</style>',
        '',
    ]
    pos = {}
    for r, row in enumerate(rows):
        y = Y0 + r * DY
        for c, (name, status) in enumerate(row):
            col = c if r % 2 == 0 else COLS - 1 - c          # snake: left->right, then right->left
            x = X0 + col * DX
            pos[name] = (x, y)
            box, lbl = _CLASS[status]
            parts.append(f'<g><rect class="{box}" x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="6"/>'
                         f'<text class="{lbl}" x="{x + BOX_W // 2}" y="{y + 18}" text-anchor="middle" '
                         f'dominant-baseline="central">{escape(name)}</text></g>')
        parts.append("")
    names = [n for n, _ in stages]
    for a, b in zip(names, names[1:]):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        if ya == yb:
            if xb > xa:
                x1, x2 = xa + BOX_W, xb
            else:
                x1, x2 = xa, xb + BOX_W
            parts.append(f'<line x1="{x1}" y1="{ya + 18}" x2="{x2}" y2="{yb + 18}" stroke="#5f5e5a" stroke-width="1.5" marker-end="url(#arrow2)"/>')
        else:
            cx = xa + BOX_W // 2
            parts.append(f'<line x1="{cx}" y1="{ya + BOX_H}" x2="{cx}" y2="{yb}" stroke="#5f5e5a" stroke-width="1.5" marker-end="url(#arrow2)"/>')
    parts += ["", f'<text x="340" y="{height - 30}" text-anchor="middle" font-size="12" fill="#5f5e5a">{legend}</text>', "</svg>", ""]
    return "\n".join(parts)


def render() -> dict[str, str]:
    """{file name: SVG text} for both diagrams."""
    t = stage_table()
    std, coh = t["standards_only"], t["cohort_with_qc"]
    return {
        "standards_only_pipeline.svg": _svg(
            "Standards-only pipeline",
            f"{len(std)}-stage pipeline for standards-only mode, generated from the registered stages.",
            f"{len(std)} stages, in run order", std),
        "cohort_with_qc_pipeline.svg": _svg(
            "Cohort-with-QC pipeline",
            f"{len(coh)}-stage pipeline for cohort mode, generated from the registered stages. Gray marks stages shared "
            "with the standards-only profile, coral marks stages with a cohort-specific implementation, teal marks stages "
            "the standards-only profile does not have.",
            "Gray = same as standards-only &#183; Coral = cohort-specific version &#183; Teal = new stage", coh),
    }


if __name__ == "__main__":
    stale = []
    for name, svg in render().items():
        path = OUT / name
        if "--check" in sys.argv:
            if not path.exists() or path.read_text() != svg:
                stale.append(name)
        else:
            path.write_text(svg)
            print("wrote", path)
    if stale:
        print("out of date:", ", ".join(stale), "- run: python tools/gen_pipeline_diagrams.py")
        sys.exit(1)
