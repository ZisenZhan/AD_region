"""Greedy expansion uses development GEO-equal apparent AUC only."""
from pathlib import Path
from datetime import datetime, timezone
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import json
import time
import hashlib
import traceback
import warnings
import numpy as np
import pandas as pd
import joblib
from joblib.externals import cloudpickle
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parents[1]
PREV = OUT.parent / "EDQ_单外部_36模型开发内选模_20261113"
sys.path.insert(0, str(PREV / "脚本"))
import 运行分析 as ref
from 拟合任务 import aggregate_prediction
old = ref.old
CONF = old.models["M23"]


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for q in iter(lambda: f.read(1024*1024), b""):
            h.update(q)
    return h.hexdigest()


def put(z, p):
    pd.DataFrame(z).to_csv(p, index=False, encoding="utf-8-sig", float_format="%.17g")


def fit_region(input_path):
    p = pd.read_pickle(input_path)
    directory = OUT / "运行缓存" / p["region"]
    directory.mkdir(exist_ok=True)
    statepath = directory / "运行状态.json"
    if statepath.exists():
        previous = json.loads(statepath.read_text(encoding="utf-8"))
        if previous.get("status") == "ok":
            return previous
    started = time.time()
    state = dict(region=p["region"], model="M23", external=p["external"], start_unix=started)
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x, m, candidates = p["train"], p["meta"], p["candidate_order"]
            assert x.index.is_unique and x.index.equals(m.index) and len(set(candidates)) == 60
            assert set(x.columns) == set(candidates)
            groups = []
            for geo in sorted(m.gse.unique()):
                indices = np.flatnonzero(m.gse.to_numpy() == geo)
                y = m.y.to_numpy()[indices] == 1
                n, nc = int(y.sum()), int((~y).sum())
                assert min(n, nc) > 0
                groups.append((geo, indices, y, n, nc))

            def evaluate(genes):
                canonical = sorted(genes)
                fitted = old.b.fit(CONF, x[canonical], m)
                prediction = old.b.mm.predict_score(fitted, x[canonical])
                assert np.isfinite(prediction).all()
                aucs = np.array([(rankdata(prediction[ix])[yy].sum()-n*(n+1)/2)/(n*nc)
                                 for geo, ix, yy, n, nc in groups])
                return float(aucs.mean()), aucs, prediction, fitted, canonical

            order = list(candidates[:6])
            curve, trials, per_geo, training_predictions = [], [], [], []
            def commit(k, added, result):
                objective, aucs, prediction, fitted, canonical = result
                curve.append(dict(k=k, added_gene=added, dev_macro=objective, genes=";".join(order),
                                  dev_delta=np.nan if k == 6 else objective-curve[-1]["dev_macro"]))
                for (geo, ix, yy, n, nc), auc in zip(groups, aucs):
                    assert abs(auc-roc_auc_score(m.y.iloc[ix], prediction[ix])) < 1e-12
                    per_geo.append(dict(k=k, geo=geo, auc=auc, n=n+nc, AD=n, control=nc))
                training_predictions.append(m[["y", "weight", "gse"]].assign(
                    row_id=m.index.astype(str), score=prediction, k=k).reset_index(drop=True))
                joblib.dump(dict(model_bytes=cloudpickle.dumps(fitted), genes=canonical, k=k),
                            directory / (f"阶段_{k:02d}_模型.joblib"), compress=3)

            commit(6, "BASE", evaluate(order))
            for k in range(7, 61):
                best_value, best_gene, best_result = -np.inf, None, None
                step_trials = []
                for gene in candidates:
                    if gene in order:
                        continue
                    result = evaluate(order+[gene])
                    score = result[0]
                    step_trials.append(dict(k=k, added_gene=gene, dev_macro=score))
                    if score > best_value+1e-12:
                        best_value, best_gene, best_result = score, gene, result
                assert best_gene is not None
                order.append(best_gene)
                for q in step_trials:
                    q["selected"] = q["added_gene"] == best_gene
                trials.extend(step_trials)
                commit(k, best_gene, best_result)
                if k % 10 == 0:
                    print("Search", p["region"], k, round(best_value, 6), best_gene, flush=True)
            assert set(order) == set(candidates) and order[:6] == candidates[:6]
            put(curve, directory / "开发集逐步轨迹.csv")
            put(trials, directory / "全部候选试加.csv.gz")
            put(per_geo, directory / "逐步开发各GEO回代AUC.csv")
            put(pd.concat(training_predictions, ignore_index=True), directory / "逐步开发回代预测.csv.gz")
            best = max(q["dev_macro"] for q in curve)
            chosen = next(q["k"] for q in curve if q["dev_macro"] >= best-1e-12)
            lock = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), uses_external_auc=False,
                        uses_internal_cv=False, objective="equal mean of apparent AUC within each development GEO",
                        selected_k=chosen, order=order, base=candidates[:6],
                        development_curve_sha256=digest(directory / "开发集逐步轨迹.csv"),
                        trials_sha256=digest(directory / "全部候选试加.csv.gz"), input_sha256=digest(input_path))
            (directory / "开发选择冻结.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
            # External expression and labels are loaded only after this region's path and size are locked.
            external_started = time.time()
            uid = p["region"]+"__"+p["external"]
            external_path, metadata_path = old.DB / (uid+"_test.pkl"), old.DB / (uid+"_md.pkl")
            xt, mt = pd.read_pickle(external_path)[candidates], pd.read_pickle(metadata_path)
            assert xt.index.equals(mt.index) and xt.notna().any().all()
            expected = pd.read_csv(p["baseline_path"], dtype={"key": str}, float_precision="round_trip")
            expected = expected.loc[expected.region.eq(p["region"])].set_index("key").sort_index()
            preds, metrics = [], []
            for step in curve:
                k = step["k"]
                saved = joblib.load(directory / (f"阶段_{k:02d}_模型.joblib"))
                fitted = cloudpickle.loads(saved["model_bytes"])
                prediction = aggregate_prediction(mt, old.b.mm.predict_score(fitted, xt[saved["genes"]]))
                got = prediction.set_index("key").sort_index()
                assert got.index.equals(expected.index) and np.array_equal(got.y, expected.y)
                if k == 6:
                    assert np.allclose(got.score, expected.score, atol=1e-14, rtol=0)
                auc = roc_auc_score(prediction.y, prediction.score)
                metrics.append(dict(**step, external_auc=auc, selected=k == chosen))
                preds.append(prediction.assign(k=k))
            put(metrics, directory / "开发与外部完整轨迹.csv")
            put(pd.concat(preds, ignore_index=True), directory / "逐步外部预测.csv.gz")
            # Independently refit the base and development-selected model.
            for k in sorted({6, chosen}):
                fresh = evaluate(order[:k])
                got = aggregate_prediction(mt, old.b.mm.predict_score(fresh[3], xt[fresh[4]])).set_index("key").sort_index()
                original = pd.concat(preds).loc[lambda z: z.k.eq(k)].set_index("key").sort_index()
                assert np.allclose(got.score, original.score, rtol=0, atol=1e-14)
            put([dict(path=str(q), bytes=q.stat().st_size, sha256=digest(q)) for q in (external_path, metadata_path)], directory / "外部输入SHA256.csv")
            state.update(status="ok", seconds=time.time()-started, selected_k=chosen,
                         candidate_trial_fits=len(trials), path_models=len(curve),
                         external_started_unix=external_started, external_read_after_lock=True,
                         baseline_predictions_reproduced=True, selected_and_base_refit_identical=True,
                         warnings=sorted(set(str(w.message) for w in caught)))
    except Exception as error:
        state.update(status="failed", seconds=time.time()-started, error=repr(error), traceback=traceback.format_exc())
    statepath.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state
