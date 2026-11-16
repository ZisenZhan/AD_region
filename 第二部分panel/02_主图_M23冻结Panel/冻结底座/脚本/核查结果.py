"""Independent checks of greedy decisions, source inputs and apparent/external AUCs."""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import 执行实验 as run


def audit():
    candidate = pd.read_csv(run.INPUT / "三脑区60候选及机制与EDQ证据.csv", float_precision="round_trip")
    av = pd.read_pickle(run.old.DB / "可测.pkl")
    checks = []
    for region in run.REGIONS:
        p = pd.read_pickle(run.CACHE / (region+"_输入.pkl"))
        d = run.CACHE / region
        state = json.loads((d / "运行状态.json").read_text(encoding="utf-8"))
        lock = json.loads((d / "开发选择冻结.json").read_text(encoding="utf-8"))
        assert state["status"] == "ok" and state["candidate_trial_fits"] == sum(range(1, 55)) == 1485
        assert not lock["uses_external_auc"] and not lock["uses_internal_cv"]
        assert datetime.fromisoformat(lock["frozen_at_utc"]).timestamp() < state["external_started_unix"]
        for path, key in [(d / "开发集逐步轨迹.csv", "development_curve_sha256"),
                          (d / "全部候选试加.csv.gz", "trials_sha256"),
                          (run.CACHE / (region+"_输入.pkl"), "input_sha256")]:
            assert run.digest(path) == lock[key]
        c = candidate.loc[candidate.region.eq(region)].sort_values("candidate_order")
        assert len(c) == c.gene.nunique() == 60 and c.gene.tolist() == p["candidate_order"]
        assert c.assigned_domain.value_counts().eq(10).all() and c.assigned_domain.nunique() == 6
        assert c.loc[c.base, "gene"].tolist() == p["base"] == lock["base"]
        assert c.D.ge(2/3).all() and c.families.ge(2).all()
        assert av.loc[c.gene, p["units"]].all().all()
        assert not set(p["meta"].gse) & set(p["excluded_GEOs"])
        assert not set(p["meta"].gse) & run.old.base.blocked(p["external"])
        rank = pd.read_csv(run.INPUT / (region+"_完整开发EDQ排名.csv.gz"), float_precision="round_trip")
        assert np.allclose(rank.score, rank.E*rank.D*rank.Q, rtol=1e-14, atol=1e-12)
        reconstructed = list(p["base"])
        for _ in range(9):
            for axis in run.old.AXES:
                options = rank.loc[rank.domains.map(lambda s: axis in s.split(";")), "gene_symbol"]
                reconstructed.append(next(g for g in options if g not in reconstructed))
        assert reconstructed == p["candidate_order"]
        trace = pd.read_csv(d / "开发与外部完整轨迹.csv", float_precision="round_trip")
        trials = pd.read_csv(d / "全部候选试加.csv.gz", float_precision="round_trip")
        assert trace.k.tolist() == list(range(6, 61))
        current = list(p["base"])
        for step in trace.itertuples():
            if step.k > 6:
                tests = trials.loc[trials.k.eq(step.k)]
                assert len(tests) == 61-step.k
                assert tests.added_gene.tolist() == [g for g in p["candidate_order"] if g not in current]
                maximum = tests.dev_macro.max()
                first = tests.loc[tests.dev_macro.ge(maximum-1e-12)].iloc[0]
                assert first.added_gene == step.added_gene
                assert abs(first.dev_macro-step.dev_macro) < 1e-14
                assert tests.selected.sum() == 1 and tests.loc[tests.selected, "added_gene"].iloc[0] == step.added_gene
                current.append(step.added_gene)
            assert step.genes.split(";") == current and set(p["base"]) <= set(current)
        assert current == lock["order"] and len(set(current)) == 60
        best = trace.dev_macro.max()
        expected_k = int(trace.loc[trace.dev_macro.ge(best-1e-12), "k"].min())
        assert lock["selected_k"] == expected_k
        devpred = pd.read_csv(d / "逐步开发回代预测.csv.gz", float_precision="round_trip")
        geoauc = pd.read_csv(d / "逐步开发各GEO回代AUC.csv", float_precision="round_trip")
        assert not devpred.duplicated(["k", "row_id"]).any()
        assert geoauc.groupby("k").size().eq(p["meta"].gse.nunique()).all()
        for (k, geo), w in devpred.groupby(["k", "gse"]):
            actual = roc_auc_score(w.y, w.score)
            saved = geoauc.loc[geoauc.k.eq(k) & geoauc.geo.eq(geo), "auc"].iloc[0]
            assert abs(actual-saved) < 1e-12
        mean = geoauc.groupby("k").auc.mean()
        assert np.allclose(mean.loc[trace.k], trace.dev_macro, rtol=0, atol=1e-12)
        external = pd.read_csv(d / "逐步外部预测.csv.gz", dtype={"key": str}, float_precision="round_trip")
        assert not external.duplicated(["k", "key"]).any()
        base = external.loc[external.k.eq(6), ["key", "y"]].sort_values("key").reset_index(drop=True)
        for k, w in external.groupby("k"):
            assert w[["key", "y"]].sort_values("key").reset_index(drop=True).equals(base)
            auc = roc_auc_score(w.y, w.score)
            assert abs(auc-trace.loc[trace.k.eq(k), "external_auc"].iloc[0]) < 1e-12
            a, b = w.loc[w.y.eq(1), "score"].to_numpy()[:, None], w.loc[w.y.eq(0), "score"].to_numpy()
            assert abs(auc-((a>b)+.5*(a==b)).mean()) < 1e-12
        assert state["baseline_predictions_reproduced"] and state["selected_and_base_refit_identical"]
        checks.append(dict(region=region, model="M23", candidates=60, trial_fits=1485, path_models=55,
                           selected_k=expected_k, candidates_EDQ_and_quotas_verified=True,
                           all_greedy_decisions_verified=True, development_GEO_mean_verified=True,
                           all_external_AUCs_reproduced=True, freeze_before_external=True,
                           base_predictions_match_frozen_version=True))
        for row in pd.read_csv(d / "外部输入SHA256.csv").itertuples():
            assert run.digest(row.path) == row.sha256
    for name in ["准备输入SHA256.csv", "搜索输入SHA256.csv"]:
        for row in pd.read_csv(run.QA / name).itertuples():
            assert run.digest(row.path) == row.sha256
    run.put(checks, run.QA / "逐脑区数值与搜索验收.csv")
    run.put([dict(path=str(p), sha256=run.digest(p)) for p in Path(__file__).parent.glob("*.py")], run.QA / "验收时脚本SHA256.csv")
    print("Passed: all greedy decisions, 60-gene quotas, dev GEO means, 165 external AUCs, frozen bases and source hashes.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        audit()
