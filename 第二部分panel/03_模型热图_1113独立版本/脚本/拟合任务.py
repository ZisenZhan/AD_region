"""Fit the existing model configurations to one frozen six-gene input."""
from pathlib import Path
import json
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import sys
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

EXP = Path(__file__).resolve().parents[2] / "别名审计与评分轮换_1107"
sys.path.insert(0, str(EXP))
from 模型拟合 import b, models


def aggregate_prediction(meta, scores):
    z = meta[["key", "y", "sub"]].copy()
    z["score"] = scores
    if not z.groupby("key").y.nunique().eq(1).all():
        raise AssertionError("A donor has inconsistent labels")
    return z.groupby(["key", "y", "sub"], sort=True).score.mean().groupby(["key", "y"], sort=True).mean().reset_index()


def fit_bundle(input_path, mids):
    data = pd.read_pickle(input_path)
    output = Path(input_path).with_suffix("")
    output.mkdir(exist_ok=True)
    records = []
    with threadpool_limits(limits=1):
        for mid in mids:
            path = output / (mid + ".csv.gz")
            status_path = output / (mid + "_状态.json")
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status["status"] == "ok" and path.exists():
                    records.append(status)
                    continue
                if status["status"] == "failed":
                    records.append(status)
                    continue
            start = time.time()
            status = dict(job_id=data["job_id"], model=mid)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = b.fit(models[mid], data["train"], data["meta"])
                    parts = []
                    for uid, (xx, md) in data["tests"].items():
                        pred = aggregate_prediction(md, b.mm.predict_score(fitted, xx))
                        pred["uid"] = uid
                        pred["model"] = mid
                        parts.append(pred)
                    prediction = pd.concat(parts, ignore_index=True)
                    if not np.isfinite(prediction.score).all():
                        raise ValueError("Nonfinite predictions")
                    prediction.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")
                    if data["stage"] == "external":
                        train_pred = data["meta"][["y", "weight", "gse"]].copy()
                        train_pred["row_id"] = train_pred.index.astype(str)
                        train_pred["score"] = b.mm.predict_score(fitted, data["train"])
                        train_pred.to_csv(output / (mid + "_开发回代.csv.gz"), index=False, encoding="utf-8-sig", float_format="%.17g")
                    status.update(status="ok", seconds=time.time()-start,
                                  warnings=sorted(set(str(w.message) for w in caught)))
            except Exception as error:
                status.update(status="failed", seconds=time.time()-start, error=repr(error), traceback=traceback.format_exc())
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(status)
    return records
