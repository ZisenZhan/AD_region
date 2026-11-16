"""Confirm the scientific audit and complete, editable, nonblank exports."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.image as mpimg
import 执行实验 as run


audit = pd.read_csv(run.QA / "逐脑区数值与搜索验收.csv")
assert len(audit) == 3
for name in ["all_greedy_decisions_verified", "development_GEO_mean_verified", "all_external_AUCs_reproduced",
             "freeze_before_external", "base_predictions_match_frozen_version"]:
    assert audit[name].all()
automatic = json.loads((run.QA / "图件自动核对.json").read_text(encoding="utf-8"))
assert len(automatic) == 3 and all(not f["text_outside_canvas"] for f in automatic)
figures = list((run.OUT / "图件").glob("*.png"))
assert len(figures) == 3
checks = []
for p in figures:
    pixels = mpimg.imread(p)
    assert float((pixels[:, :, :3] < .96).any(axis=2).mean()) > .01
    assert p.with_suffix(".svg").exists() and p.with_suffix(".pdf").exists()
    assert "<text" in p.with_suffix(".svg").read_text(encoding="utf-8")
    checks.append(dict(figure=p.stem, nonblank=True, editable_svg=True, pdf_export=True, visual_review="reviewed"))
run.put(checks, run.QA / "最终图件验收.csv")
run.put([dict(path=str(p), sha256=run.digest(p)) for p in Path(__file__).parent.glob("*.py")], run.QA / "最终实现脚本SHA256.csv")
files = [p for name in ("输入与划分", "结果", "图件") for p in (run.OUT / name).iterdir() if p.is_file()]
files += [run.OUT / "结果说明.md", run.OUT / "分析方案_运行前固定.md"]
run.put([dict(path=str(p), bytes=p.stat().st_size, sha256=run.digest(p)) for p in files], run.QA / "最终输出清单.csv")
print("Passed: scientific audit, all 3 figure sets, editable SVG/PDF, nonblank PNG, final checksums.")
