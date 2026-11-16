"""Full forward path selected by development leave-source-out AUC only."""
from pathlib import Path
from datetime import datetime, timezone
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import json
import hashlib
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from 模型与折 import BASE, SOURCE, Fold

OUT = Path(__file__).resolve().parents[1]
PREV = BASE / "M23_EDQ_冻结划分_三脑区贪婪扩展_20261115"
DB = BASE / "别名审计与评分轮换_1107/训练与评测缓存"
REGIONS = ["frontal_prefrontal", "temporal_neocortex", "hippocampus"]
EXPECTED = dict(zip(REGIONS, [9, 4, 5]))
EXPECTED_FOLDS = dict(zip(REGIONS, [9, 4, 4]))
INPUT, CACHE, RESULT, QA, MODELS = [OUT / n for n in ("输入与划分", "运行缓存", "结果", "核对", "模型")]
READS = set()


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(p):
    READS.add(Path(p))
    return pd.read_pickle(p)


def put(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def source(geo):
    if geo in {"GSE36980", "GSE173954"}:
        return "Hisayama"
    if geo in {"GSE33000", "GSE44768", "GSE44770", "GSE44771"}:
        return "HBTRC"
    return geo


def run_region(region, evidence):
    start = time.time()
    payload_path = PREV / "运行缓存" / (region + "_输入.pkl")
    p = load(payload_path)
    units, genes, base = p["units"], p["candidate_order"], p["base"]
    x, m = p["train"], p["meta"]
    assert len(units) == EXPECTED[region] and len(genes) == len(set(genes)) == 60
    assert genes[:6] == base and len(set(base)) == 6
    assert x.index.equals(m.index) and x.index.is_unique and m.gse.nunique() == EXPECTED[region]
    assert set(x.columns) == set(genes)
    assert p["external"] not in set(m.gse)
    tab = evidence.loc[evidence.region.eq(region)].sort_values("candidate_order")
    assert tab.gene.tolist() == genes and tab.iloc[:6].gene.tolist() == base
    assert np.allclose(tab.score, tab.E * tab.D * tab.Q) and tab.D.ge(2/3).all() and tab.families.ge(2).all()
    put(m.assign(row_id=m.index.astype(str)), INPUT / (region + "_开发记录.csv.gz"))
    group_units = {}
    for uid in units:
        group_units.setdefault(source(uid.split("__")[1]), []).append(uid)
    folds, split = [], []
    for family, queries in sorted(group_units.items()):
        heldout_geos = {u.split("__")[1] for u in queries}
        train_mask = ~m.gse.isin(heldout_geos)
        train, info = x.loc[train_mask], m.loc[train_mask]
        assert not set(info.gse) & heldout_geos
        assert not {source(g) for g in info.gse} & {source(g) for g in heldout_geos}
        tests = {}
        for uid in queries:
            # Only development GEO files may be opened here.
            assert uid in units and not uid.endswith("__" + p["external"])
            geo = uid.split("__")[1]
            xt, md = load(DB / (uid + "_test.pkl"))[genes], load(DB / (uid + "_md.pkl"))
            assert xt.index.equals(md.index)
            tests[geo] = (xt, md)
            expected = m.loc[m.gse.eq(geo)].y.to_numpy()
            assert md.key.nunique() == len(expected)
            split.append(dict(region=region, fold=family, test_GEO=geo,
                train_GEOs=";".join(sorted(info.gse.unique())), train_rows=len(info),
                train_AD=int(info.y.sum()), train_control=int((1-info.y).sum()),
                test_rows=md.key.nunique(), shared_family_held_out=True))
        folds.append(Fold(family, train, info, tests))
    assert len(folds) == EXPECTED_FOLDS[region] and len(split) == EXPECTED[region]
    put(split, INPUT / (region + "_交叉验证划分.csv"))
    directory = CACHE / region
    directory.mkdir(exist_ok=True)
    frozen = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), region=region, base=base,
        candidates=genes, units=units, folds=split, model="M23", components=2,
        objective="unweighted mean of held-out development GEO AUCs", uses_external=False,
        candidate_input_sha256=sha(payload_path), plan_sha256=sha(OUT / "分析方案_运行前固定.md"))
    (directory / "搜索前冻结.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    curve, trials, auc_rows, pred_rows, reference_checks = [], [], [], [], []

    def evaluate(panel):
        values = {}
        for fold in folds:
            one = fold.evaluate(panel)
            assert not set(values) & set(one)
            values.update(one)
        assert len(values) == EXPECTED[region]
        return float(np.mean([v["auc"] for v in values.values()])), values

    def commit(panel, added, value, values):
        k = len(panel)
        curve.append(dict(region=region, k=k, added_gene=added, cv_macro=value,
            cv_delta=np.nan if k == 6 else value-curve[-1]["cv_macro"], genes=";".join(panel)))
        for fold in folds:
            for geo, q in fold.queries.items():
                y, scores = q["y"], values[geo]["score"]
                auc = roc_auc_score(y, scores)
                assert abs(auc-values[geo]["auc"]) < 1e-12
                auc_rows.append(dict(region=region, k=k, fold=fold.fold_id, geo=geo, auc=auc,
                    n=len(y), AD=int(y.sum()), control=int((1-y).sum())))
                pred_rows.extend(dict(region=region, k=k, fold=fold.fold_id, geo=geo,
                    key=str(key), y=int(yy), score=float(s)) for key, yy, s in zip(q["keys"], y, scores))

    order = list(base)
    initial, values = evaluate(order)
    commit(order, "BASE", initial, values)
    print("BASE", region, initial, "folds", len(folds), "GEOs", EXPECTED[region], flush=True)
    for k in range(7, 61):
        best, selected, best_values, step = -np.inf, None, None, []
        for candidate in genes:
            if candidate in order:
                continue
            value, values = evaluate(order + [candidate])
            step.append(dict(region=region, k=k, candidate=candidate, cv_macro=value,
                             **{"auc_"+g: v["auc"] for g, v in values.items()}))
            if value > best+1e-12:
                best, selected, best_values = value, candidate, values
        assert selected is not None
        order.append(selected)
        for row in step:
            row["selected"] = row["candidate"] == selected
        trials.extend(step)
        commit(order, selected, best, best_values)
        if k % 5 == 0 or k == 7:
            put(curve, directory / "阶段轨迹.csv")
            put(trials, directory / "阶段全部试加.csv.gz")
            print("STEP", region, k, round(best, 6), selected, "seconds", round(time.time()-start, 1), flush=True)
    assert len(trials) == 1485 and len(order) == len(set(order)) == 60 and order[:6] == base
    peak = max(row["cv_macro"] for row in curve)
    selected_k = next(row["k"] for row in curve if row["cv_macro"] >= peak-1e-12)
    selected_genes = order[:selected_k]
    put(curve, directory / "交叉验证完整轨迹.csv")
    put(trials, directory / "全部候选试加.csv.gz")
    put(auc_rows, directory / "逐步逐GEO_AUC.csv")
    put(pred_rows, directory / "逐步OOF预测.csv.gz")
    saved_pred = pd.DataFrame(pred_rows)
    for k in sorted({6, selected_k, 60}):
        panel = order[:k]
        for fold in folds:
            fitted, predictions = fold.reference(panel)
            joblib.dump(dict(model=fitted, genes=sorted(panel), model_id="M23", k=k, fold=fold.fold_id),
                        MODELS / (region + "__" + fold.fold_id + "__" + str(k) + ".joblib"), compress=3)
            for geo, score in predictions.items():
                q = fold.queries[geo]
                expected = saved_pred.loc[saved_pred.k.eq(k) & saved_pred.geo.eq(geo)].set_index("key").loc[q["keys"]]
                error = float(np.max(np.abs(score-expected.score.to_numpy())))
                assert error < 1e-10, (region, k, geo, error)
                assert abs(roc_auc_score(q["y"], score)-roc_auc_score(expected.y, expected.score)) < 1e-12
                reference_checks.append(dict(region=region, k=k, geo=geo, prediction_max_error=error, auc_identical=True))
    summary = dict(region=region, cv_GEOs=EXPECTED[region], cv_folds=len(folds), candidates=60,
        additional_candidates=54, selected_k=selected_k, added_genes=selected_k-6,
        cv_base=initial, cv_selected=peak, cv_change=peak-initial, genes=";".join(selected_genes),
        added_order=";".join(selected_genes[6:]), trial_panels=1485,
        candidate_fits=1485*len(folds), seconds=time.time()-start)
    (directory / "完成与规模冻结.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    put(reference_checks, directory / "完整流水线独立核对.csv")
    print("FINISHED", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    for path in [INPUT, CACHE, RESULT, QA, MODELS]:
        path.mkdir(parents=True, exist_ok=True)
    path = PREV / "输入与划分/三脑区60候选及机制与EDQ证据.csv"
    READS.add(path)
    evidence = pd.read_csv(path, float_precision="round_trip")
    put(evidence.drop(columns=["external"], errors="ignore"), INPUT / "固定60候选及机制与EDQ证据.csv")
    summary = []
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for region in REGIONS:
            summary.append(run_region(region, evidence))
    for name in ["交叉验证完整轨迹.csv", "全部候选试加.csv.gz", "逐步逐GEO_AUC.csv", "逐步OOF预测.csv.gz", "完整流水线独立核对.csv"]:
        parts = [pd.read_csv(CACHE / region / name, float_precision="round_trip", dtype={"key": str}) for region in REGIONS]
        put(pd.concat(parts, ignore_index=True), RESULT / name)
    put(summary, RESULT / "三脑区交叉验证贪婪汇总.csv")
    order = []
    curve = pd.read_csv(RESULT / "交叉验证完整轨迹.csv", float_precision="round_trip")
    for row in summary:
        region = row["region"]
        path = curve.loc[curve.region.eq(region)].sort_values("k").iloc[-1].genes.split(";")
        for i, gene in enumerate(path, 1):
            ev = evidence.loc[evidence.region.eq(region) & evidence.gene.eq(gene)].iloc[0].to_dict()
            ev.pop("external", None)
            order.append(dict(path_position=i, in_selected_panel=i<=row["selected_k"], **ev))
    put(order, RESULT / "完整基因添加顺序及证据.csv")
    put([z for z in order if z["in_selected_panel"]], RESULT / "交叉验证选定Panel及证据.csv")
    READS.add(SOURCE)
    put([dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(READS)], QA / "实际读取输入SHA256.csv")
    assert not any(p.name in {"frontal_prefrontal__GSE261817_test.pkl", "frontal_prefrontal__GSE261817_md.pkl",
        "temporal_neocortex__GSE132903_test.pkl", "temporal_neocortex__GSE132903_md.pkl",
        "hippocampus__GSE48350_test.pkl", "hippocampus__GSE48350_md.pkl"} for p in READS)
    (QA / "运行状态.json").write_text(json.dumps(dict(status="ok", uses_external=False,
        scope="fixed-pool development cross-validation search; not nested selection",
        warnings=sorted({str(w.message) for w in caught})), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
