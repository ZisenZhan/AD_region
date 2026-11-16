"""Recheck each deletion's eligibility, selected genes, train/test and scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import 执行实验 as run


def audit():
    frozen = json.loads((run.INPUT / "三十组合_运行前冻结.json").read_text(encoding="utf-8"))
    combos = frozen["combinations"]
    assert frozen["model"] == "M23" and frozen["model_components"] == 2 and frozen["score"] == "EDQ"
    assert len(combos) == len({c["case_id"] for c in combos}) == 30
    assert {(c["deleted_GEO"], c["temporal_external"]) for c in combos} == {
        (d, h) for d in run.DELETIONS for h in run.HOLDOUTS if h not in run.old.base.blocked(d)}
    evidence = run.read(run.INPUT / "各划分六基因及EDQ证据.csv")
    splits = run.read(run.INPUT / "实际训练测试与共同排除.csv")
    jobs = {j["job_id"]: j for j in json.loads((run.INPUT / "冻结拟合任务.json").read_text(encoding="utf-8"))}
    lookup = {c["case_id"]: c for c in combos}
    checks = []
    av = pd.read_pickle(run.old.DB / "可测.pkl")
    for row in splits.itertuples():
        c = lookup[row.case_id]
        deleted_family = run.old.base.blocked(c["deleted_GEO"])
        actual_targets = [row.temporal_external] if row.scope == run.T else ["GSE261817", row.temporal_external, "GSE48350"]
        blocked = deleted_family | set.union(*(run.old.base.blocked(g) for g in actual_targets))
        units = row.training_units.split(";")
        assert set(row.excluded_GEOs.split(";")) == blocked
        assert not {u.split("__")[1] for u in units} & blocked
        assert jobs[row.job_id]["units"] == units
        p = pd.read_pickle(run.CACHE / (row.job_id+".pkl"))
        assert p["train"].index.is_unique and p["train"].index.equals(p["meta"].index)
        assert set(p["meta"].gse) == {u.split("__")[1] for u in units}
        assert p["train"].columns.tolist() == sorted(p["train"].columns)
        rankpath = run.SCORE / row.deleted_GEO / (row.case_id+"__"+row.scope+"_EDQ排名.csv.gz")
        rank = pd.read_csv(rankpath, float_precision="round_trip")
        assert rank.gene_symbol.is_unique and len(rank) == row.candidate_genes
        assert rank.D.ge(2/3).all() and rank.families.ge(2).all()
        assert np.allclose(rank.score, rank.E*rank.D*rank.Q, atol=1e-12, rtol=1e-14)
        assert np.allclose(rank.Q, -np.log10(rank.FDR.clip(lower=1e-300)), atol=1e-12, rtol=1e-14)
        assert np.allclose(rank.E, rank.median_g.abs(), atol=1e-14, rtol=0)
        assert rank.gene_symbol.tolist() == rank.sort_values(["score", "gene_symbol"], ascending=[False, True]).gene_symbol.tolist()
        assert av.loc[rank.gene_symbol, units].all().all()
        for g in actual_targets:
            assert set(rank.gene_symbol) <= run.old.static(g)
        selected = evidence.loc[evidence.case_id.eq(row.case_id) & evidence.scope.eq(row.scope) & evidence.arm.eq(row.arm)].sort_values("slot")
        assert selected.gene.nunique() == len(selected) == 6
        assert sorted(selected.gene) == jobs[row.job_id]["genes"]
        if row.arm in ("B", "D"):
            assert selected.gene.tolist() == rank.gene_symbol.head(6).tolist()
        else:
            wanted = []
            for axis in run.old.AXES:
                options = rank.loc[rank.domains.map(lambda z: axis in z.split(";")), "gene_symbol"]
                wanted.append(next(g for g in options if g not in wanted))
            assert selected.gene.tolist() == wanted
        checks.append(dict(case_id=row.case_id, scope=row.scope, arm=row.arm,
                           EDQ_and_ranking_verified=True, six_eligible_genes_verified=True,
                           deletion_and_test_families_excluded=True, candidate_background_only_actual_targets=True))
    assert len(checks) == 120
    run.put(checks, run.QA / "逐组合评分名单与排除核查.csv")
    pred = pd.read_csv(run.R / "全部组合逐供体预测.csv.gz", dtype={"key": str}, float_precision="round_trip")
    metrics = pd.read_csv(run.R / "全部组合逐策略外部AUC.csv", float_precision="round_trip")
    assert len(metrics) == 360 and metrics.groupby(["case_id", "region"]).size().eq(4).all()
    error = 0.
    for (case, region, arm), w in pred.groupby(["case_id", "region", "arm"]):
        saved = metrics.loc[metrics.case_id.eq(case) & metrics.region.eq(region) & metrics.arm.eq(arm), "auc"].iloc[0]
        error = max(error, abs(roc_auc_score(w.y, w.score)-saved))
    assert error < 1e-12
    temporal = pd.read_csv(run.R / "颞叶三十组合_四条件与删除前后变化.csv", float_precision="round_trip")
    for row in temporal.itertuples():
        passes = sum(getattr(row, a)-getattr(row, b) > 1e-12 for a, b in [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")])
        assert passes == row.conditions_passed
    for row in pd.read_csv(run.QA / "来源与实现SHA256.csv").itertuples():
        assert run.digest(row.path) == row.sha256
    run.put([dict(path=str(p), sha256=run.digest(p)) for p in Path(__file__).parent.glob("*.py")], run.QA / "最终脚本SHA256.csv")
    (run.QA / "独立复核结论.json").write_text(json.dumps(dict(
        cases=30, strategy_selection_audits=120, aucs_recomputed=360,
        maximum_auc_error=error, input_hashes_unchanged=True, fixed_M23_EDQ=True,
        strict_conditions_reproduced=True), indent=2), encoding="utf-8")
    print("Independent checks passed: 30 cases, 120 panel selections, 360 AUCs, source hashes.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        audit()
