"""Recompute metrics, audit family exclusions, final panels, and saved rankings."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import 运行分析 as run

OUT, INPUT, RESULT, CACHE, QA = run.OUT, run.INPUT, run.RESULT, run.CACHE, run.QA
old = run.old
checks = []


def check(name, condition, detail=""):
    checks.append(dict(check=name, passed=bool(condition), detail=detail))
    if not condition:
        raise AssertionError((name, detail))


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def run_audit():
    aliases = read(INPUT / "拟合任务与策略对应.csv")
    split = read(INPUT / "每折训练测试与排除单元.csv")
    panels = read(INPUT / "每折六基因与EDQ证据.csv")
    jobs = json.loads((INPUT / "冻结任务.json").read_text(encoding="utf-8"))
    mapping = {j["job_id"]: j for j in jobs}
    experiment = old.Experiment()
    check("36 model configurations", len(run.MODEL_IDS) == 36)
    check("internal and external evaluation cells", (aliases.stage.eq("internal").sum(), aliases.stage.eq("external").sum()) == (76, 12))
    evidence = []
    for row in split.itertuples():
        units, targets = row.training_units.split(";"), row.excluded_GEOs.split(";")
        geos = {u.split("__")[1] for u in units}
        test = row.test_units.split(";")
        assert not geos & set(targets)
        assert all(not geos & old.base.blocked(u.split("__")[1]) for u in test)
        assert row.training_units == row.score_training_units
        assert set(units) == set(mapping[row.job_id]["units"])
        chosen = panels.loc[panels.job_id.eq(row.job_id) & panels.arm.eq(row.arm) & panels.fold.eq(row.fold)]
        assert len(chosen) == chosen.gene.nunique() == 6
        assert np.allclose(chosen.score, chosen.E*chosen.D*chosen.Q, rtol=1e-13, atol=1e-15)
        assert (chosen.D >= 2/3).all() and (chosen.families >= 2).all()
        assert experiment.av.loc[chosen.gene, units].all().all()
        for uid in test:
            assert set(chosen.gene).issubset(old.static(uid.split("__")[1]))
        if row.arm in ("A", "C"):
            assert all(r.assigned_domain in r.domains.split(";") for r in chosen.itertuples())
        else:
            assert chosen.sort_values("slot").score_rank.tolist() == list(range(1, 7))
        payload = pd.read_pickle(CACHE / (row.job_id+".pkl"))
        assert payload["train"].columns.tolist() == sorted(chosen.gene)
        assert payload["train"].index.equals(payload["meta"].index)
        assert set(payload["meta"].gse) == geos
        if row.scope == "pooled":
            eff = experiment.pE[sorted(set(old.base.fam(g) for g in geos))]
        else:
            groups = {}
            for uid in units:
                groups.setdefault(old.base.fam(uid.split("__")[1]), []).append(uid)
            eff = pd.DataFrame({f: experiment.rE[us].mean(axis=1) for f, us in groups.items()})
        med = eff.median(axis=1)
        direction = np.sign(eff).eq(np.sign(med), axis=0).sum(axis=1)/eff.count(axis=1)
        assert np.allclose(chosen.E, med.loc[chosen.gene].abs(), atol=1e-14, rtol=1e-13)
        assert np.allclose(chosen.D, direction.loc[chosen.gene], atol=1e-14, rtol=1e-13)
        if row.stage == "internal":
            qpath = run.SCORE / (row.scope+"_"+old.digest(";".join(units))+"_训练FDR.pkl")
            saved_q = pd.read_pickle(qpath)
            assert np.allclose(chosen.FDR, saved_q.loc[chosen.gene], atol=1e-14, rtol=1e-13)
            assert np.allclose(chosen.Q, -np.log10(saved_q.loc[chosen.gene].clip(lower=1e-300)), atol=1e-13, rtol=1e-13)
        evidence.append(dict(stage=row.stage, scope=row.scope, arm=row.arm, fold=row.fold,
                             training_rows=len(payload["meta"]), AD=int(payload["meta"].y.sum()),
                             control=int((1-payload["meta"].y).sum()), training_GEOs=len(geos),
                             training_units=len(units), scoring_families=eff.shape[1],
                             test_units=row.test_units, blocked_family_overlap=0))
    check("six distinct genes, EDQ evidence, technical coverage and family exclusions", True, f"{len(split)} strategy-fold records")
    q_checks = []
    for scope, part in split.loc[split.stage.eq("internal")].groupby("scope", sort=False):
        row = part.iloc[0]
        units = row.training_units.split(";")
        if scope == "pooled":
            fams = sorted(set(old.base.fam(u.split("__")[1]) for u in units))
            blocks = {f: old.fde(f) for f in fams}
        else:
            groups = {}
            for uid in units:
                groups.setdefault(old.base.fam(uid.split("__")[1]), []).append(uid)
            blocks = {}
            for f, us in groups.items():
                if len(us) == 1:
                    blocks[f] = old.de(us[0])
                else:
                    z = pd.concat([old.de(u) for u in us]).assign(se2=lambda v: v.se**2).groupby(level=0).agg(logFC=("logFC", "mean"), se2=("se2", "mean"))
                    z["se"] = np.sqrt(z.se2)
                    z["p_value"] = 2*norm.sf(abs(z.logFC/z.se.replace(0, np.nan)))
                    blocks[f] = z.dropna()
        fresh_q = old.qval(blocks, 3 if scope == "pooled" else run.REGIONS.index(scope), check=True)
        saved_q = pd.read_pickle(run.SCORE / (scope+"_"+old.digest(";".join(units))+"_训练FDR.pkl"))
        error = float(np.max(np.abs(fresh_q.loc[saved_q.index]-saved_q)))
        q_checks.append(dict(scope=scope, fold=row.fold, genes=len(saved_q), max_absolute_error=error))
    run.put(q_checks, QA / "训练内SumRank独立复算.csv")
    check("training-only SumRank recomputed from DE blocks in four representative scopes", all(z["max_absolute_error"] < 1e-14 for z in q_checks))
    run.put(evidence, QA / "逐折输入人数及核查.csv")
    max_error = 0.
    sample_rows = []
    for stage in ("internal", "external"):
        prediction = pd.read_csv(RESULT / (stage+"_逐供体预测.csv.gz"), dtype={"key": str}, float_precision="round_trip")
        metric = read(RESULT / (stage+"_逐GEO_AUC.csv"))
        expected = int(aliases.stage.eq(stage).sum())*36
        check(stage+" complete AUC cells", len(metric) == expected, str(expected))
        assert not prediction.duplicated(["region", "geo", "arm", "model", "key"]).any()
        by = prediction.groupby(["region", "geo", "arm", "model"], sort=False)
        for row in metric.itertuples():
            z = by.get_group((row.region, row.geo, row.arm, row.model))
            assert len(z) == row.n and z.y.sum() == row.AD
            auc = roc_auc_score(z.y, z.score)
            a, b = z.loc[z.y.eq(1), "score"].to_numpy()[:, None], z.loc[z.y.eq(0), "score"].to_numpy()
            pairwise = ((a > b)+.5*(a == b)).mean()
            max_error = max(max_error, abs(auc-row.auc), abs(pairwise-row.auc))
        for (r, g), group in prediction.groupby(["region", "geo"]):
            keys = group[["key", "y"]].drop_duplicates().sort_values("key")
            for _, cell in group.groupby(["arm", "model"]):
                assert cell[["key", "y"]].sort_values("key").reset_index(drop=True).equals(keys.reset_index(drop=True))
            sample_rows.append(dict(stage=stage, region=r, geo=g, n=len(keys), AD=int(keys.y.sum()),
                                    control=int((1-keys.y).sum()), identical_samples_across_arms_and_models=True))
    check("saved predictions reproduce every AUC, including pairwise calculation", max_error < 1e-12, f"maximum absolute error {max_error:.3g}")
    run.put(sample_rows, QA / "测试供体与各策略样本一致性.csv")
    internal = read(RESULT / "internal_逐GEO_AUC.csv")
    macro = read(RESULT / "开发内部_脑区策略Macro.csv").set_index(["region", "arm", "model"])
    fm = internal.groupby(["region", "arm", "model", "family"]).auc.mean()
    recomputed = fm.groupby(["region", "arm", "model"]).mean()
    check("family-equal development Macro", np.allclose(macro.loc[recomputed.index].internal_macro, recomputed, rtol=0, atol=1e-14))
    ranking = read(RESULT / "模型总排名_开发内部四策略三脑区.csv").set_index("model")
    score = recomputed.groupby("model").mean()
    check("main selection score is equal mean of 12 internal Macro AUCs", np.allclose(ranking.loc[score.index].selection_score, score, atol=1e-14, rtol=0))
    check("main ranks use development only", np.array_equal(ranking.loc[score.index, "rank"].to_numpy(), score.rank(method="min", ascending=False).to_numpy()))
    frozen_path = RESULT / "开发内部排名_选模冻结.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    earliest = min((CACHE/j["job_id"]/(mid+"_状态.json")).stat().st_mtime
                   for j in jobs if j["stage"] == "external" for mid in run.MODEL_IDS)
    frozen_time = datetime.fromisoformat(frozen["frozen_at_utc"]).timestamp()
    check("this-run classifier ranking frozen before external fitting completed", frozen_time < earliest,
          f"ranking frozen {frozen['frozen_at_utc']}; earliest external completion {datetime.fromtimestamp(earliest, timezone.utc).isoformat()}")
    check("frozen ranking file unchanged", hashlib.sha256((RESULT / "模型总排名_开发内部四策略三脑区.csv").read_bytes()).hexdigest() == frozen["ranking_sha256"])
    check("internal predictions unchanged after model lock", hashlib.sha256((RESULT / "internal_逐供体预测.csv.gz").read_bytes()).hexdigest() == frozen["internal_predictions_sha256"])
    for filename in ["准备阶段来源SHA256.csv", "internal_表达输入SHA256.csv", "external_表达输入SHA256.csv"]:
        for row in read(QA / filename).itertuples():
            assert hashlib.sha256(Path(row.path).read_bytes()).hexdigest() == row.sha256
    check("all archived input source hashes unchanged", True)
    historical = []
    for scope, targets in [(r, (run.EXTERNAL[r],)) for r in run.REGIONS] + [("pooled", run.OUTER)]:
        code = scope+"_"+"_".join(targets)
        prior = read(run.EXP / "逐划分" / (code+"_AUC.csv"))
        prior = prior.loc[prior.formula.eq("EDQ")]
        current = read(RESULT / "external_逐GEO_AUC.csv")
        if scope != "pooled":
            current = current.loc[current.region.eq(scope) & current.arm.isin(["A", "B"])]
        else:
            current = current.loc[current.arm.isin(["C", "D"])]
        match = prior.merge(current, on=["region", "arm", "model"], suffixes=("_previous", "_current"))
        for row in match.itertuples():
            historical.append(dict(scope=scope, region=row.region, arm=row.arm, model=row.model,
                                   previous_auc=row.auc_previous, current_auc=row.auc_current,
                                   difference=row.auc_current-row.auc_previous,
                                   note="canonical alphabetical feature columns; coordinate/randomized estimators may differ"))
    history = pd.DataFrame(historical)
    run.put(history, QA / "旧版外部AUC逐项复核.csv")
    anchor = history.loc[history.model.eq("M18")]
    check("EDQ shrinkage LDA 0.5 matches archived external results", anchor.difference.abs().max() < 1e-12, str(anchor.difference.abs().max()))
    warnings = []
    for stage in ("internal", "external"):
        states = json.loads((QA / (stage+"_拟合状态.json")).read_text(encoding="utf-8"))
        check(stage+" no failed estimators", all(z["status"] == "ok" for z in states), f"{len(states)} successful fits")
        for z in states:
            warnings.extend(dict(stage=stage, job_id=z["job_id"], model=z["model"], warning=w) for w in z.get("warnings", []))
    run.put(pd.DataFrame(warnings, columns=["stage", "job_id", "model", "warning"]), QA / "拟合警告汇总.csv")
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "joblib", "matplotlib", "threadpoolctl"]
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       packages={p: importlib.metadata.version(p) for p in packages},
                       classifier_seed=20260807, SumRank_seed=20260831, SumRank_permutations=old.sr.N_RANK_PERMUTATIONS,
                       workers=4, thread_limit_per_worker=1,
                       process_shutdown_note="loky ResourceTracker destructor emitted an ignored AttributeError after successful completion; fit status and all saved outputs verified")
    (QA / "运行环境.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    run.put(checks, QA / "数值与输入验收.csv")
    print(pd.DataFrame(checks).to_string(index=False))


if __name__ == "__main__":
    run_audit()
