"""Freeze inputs and candidates, run regional searches, then summarize external effects."""
from pathlib import Path
from datetime import datetime, timezone
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import json
import sys
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from 贪婪任务 import fit_region, old, CONF

OUT = Path(__file__).resolve().parents[1]
PREV = OUT.parent / "EDQ_单外部_36模型开发内选模_20261113"
FIXED = OUT.parent / "M23_EDQ_颞叶删一与外部轮换_20261115"
CASE = "GSE159699__GSE132903"
INPUT, CACHE, R, QA = [OUT / n for n in ("输入与划分", "运行缓存", "结果", "核对")]
for p in (INPUT, CACHE, R, QA):
    p.mkdir(exist_ok=True)
REGIONS = {"frontal_prefrontal": "额叶前额叶", "temporal_neocortex": "颞叶新皮层", "hippocampus": "海马"}
EXTERNAL = dict(zip(REGIONS, ["GSE261817", "GSE132903", "GSE48350"]))
FILES = set()


def put(z, p):
    pd.DataFrame(z).to_csv(p, index=False, encoding="utf-8-sig", float_format="%.17g")


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for q in iter(lambda: f.read(1024*1024), b""):
            h.update(q)
    return h.hexdigest()


def read(p, **kwargs):
    FILES.add(Path(p))
    return pd.read_csv(p, float_precision="round_trip", **kwargs)


def load(p):
    FILES.add(Path(p))
    return pd.read_pickle(p)


def prepare():
    splitpath = PREV / "输入与划分/原单外部划分.json"
    FILES.add(splitpath)
    original_splits = json.loads(splitpath.read_text(encoding="utf-8"))
    new_splits = read(FIXED / "输入与划分/实际训练测试与共同排除.csv")
    new_splits = new_splits.loc[new_splits.case_id.eq(CASE) & new_splits.scope.eq("temporal_neocortex") & new_splits.arm.eq("A")].iloc[0]
    original_genes = read(PREV / "结果/最终六基因及EDQ证据.csv")
    new_genes = read(FIXED / "输入与划分/各划分六基因及EDQ证据.csv")
    fixed = read(FIXED / "结果/全部组合逐供体预测.csv.gz", dtype={"key": str})
    fixed = fixed.loc[fixed.case_id.eq(CASE) & fixed.arm.eq("A"), ["region", "geo", "key", "y", "score"]]
    baselinepath = INPUT / "冻结底座原始外部预测.csv"
    put(fixed, baselinepath)
    frozen_abcd = read(FIXED / "结果/三脑区全部四条件核查.csv")
    put(frozen_abcd.loc[frozen_abcd.case_id.eq(CASE)], INPUT / "冻结版本ABCD外部结果.csv")
    av = load(old.DB / "可测.pkl")
    candidates_out, evidence_out, split_out, rows_out, paths = [], [], [], [], []
    for region, external in EXTERNAL.items():
        if region == "temporal_neocortex":
            units = new_splits.training_units.split(";")
            excluded = new_splits.excluded_GEOs.split(";")
            poolpath = FIXED / "评分/GSE159699" / (CASE+"__temporal_neocortex_EDQ排名.csv.gz")
            base = new_genes.loc[new_genes.case_id.eq(CASE) & new_genes.scope.eq(region) & new_genes.arm.eq("A")].sort_values("slot").gene.tolist()
        else:
            units = original_splits[region]["development_units"]
            excluded = original_splits[region]["blocked_GEOs"]
            poolpath = PREV / "折内评分" / (region+"_"+external+"_完整开发评分.csv.gz")
            base = original_genes.loc[original_genes.scope.eq(region) & original_genes.arm.eq("A")].sort_values("slot").gene.tolist()
        pool = read(poolpath).set_index("gene_symbol")
        rank = pool.assign(score=pool.E*pool.D*pool.Q).rename_axis("gene_symbol").reset_index()
        rank = rank.sort_values(["score", "gene_symbol"], ascending=[False, True]).reset_index(drop=True)
        rank["EDQ_rank"] = np.arange(1, len(rank)+1)
        rank = rank.set_index("gene_symbol")
        assert len(base) == len(set(base)) == 6 and rank.D.ge(2/3).all() and rank.families.ge(2).all()
        assert av.loc[rank.index, units].all().all()
        queues = {axis: rank.index[rank.domains.map(lambda d: axis in d.split(";"))].tolist() for axis in old.AXES}
        reconstructed = []
        for axis in old.AXES:
            reconstructed.append(next(g for g in queues[axis] if g not in reconstructed))
        assert reconstructed == base, (region, reconstructed, base)
        candidates = list(base)
        assigned = list(old.AXES)
        for round_number in range(2, 11):
            for axis in old.AXES:
                gene = next((g for g in queues[axis] if g not in candidates), None)
                if gene is None:
                    raise ValueError((region, axis, "Cannot fill ten unique slots"))
                candidates.append(gene)
                assigned.append(axis)
        assert len(candidates) == len(set(candidates)) == 60
        for slot, (gene, domain) in enumerate(zip(candidates, assigned), 1):
            candidates_out.append(dict(region=region, external=external, candidate_order=slot,
                quota_round=(slot-1)//6+1, assigned_domain=domain, gene=gene, base=slot<=6, **rank.loc[gene].to_dict()))
        put(rank.reset_index(), INPUT / (region+"_完整开发EDQ排名.csv.gz"))
        x = pd.concat([load(old.DB / (u+"_train.pkl"))[candidates] for u in units])
        m = pd.concat([load(old.DB / (u+"_info.pkl")) for u in units]).loc[x.index]
        assert x.index.is_unique and x.index.equals(m.index) and set(m.y) == {0, 1}
        assert not set(m.gse) & set(excluded) and not set(m.gse) & old.base.blocked(external)
        assert len(units) == {"frontal_prefrontal": 9, "temporal_neocortex": 4, "hippocampus": 5}[region]
        for k, gene in enumerate(base, 1):
            evidence_out.append(dict(region=region, external=external, slot=k, gene=gene,
                                      assigned_domain=old.AXES[k-1], **rank.loc[gene].to_dict()))
        split_out.append(dict(region=region, external=external, development_units=";".join(units),
                              excluded_GEOs=";".join(excluded), training_rows=len(m), AD=int(m.y.sum()),
                              control=int((1-m.y).sum()), development_GEOs=m.gse.nunique(), candidates=60))
        rows_out.append(m.assign(region=region, row_id=m.index.astype(str)).reset_index(drop=True))
        payload = dict(region=region, external=external, train=x, meta=m, base=base, candidate_order=candidates,
                       candidate_domains=assigned, units=units, excluded_GEOs=excluded, baseline_path=str(baselinepath))
        path = CACHE / (region+"_输入.pkl")
        pd.to_pickle(payload, path)
        paths.append(path)
        print("Prepared", region, len(m), "training rows; base", base, flush=True)
    put(candidates_out, INPUT / "三脑区60候选及机制与EDQ证据.csv")
    put(evidence_out, INPUT / "固定六基因底座及证据.csv")
    put(split_out, INPUT / "冻结划分与开发人数.csv")
    put(pd.concat(rows_out, ignore_index=True), INPUT / "开发供体记录.csv.gz")
    (INPUT / "版本与搜索规则冻结.json").write_text(json.dumps(dict(
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), model="M23", model_label=CONF.display_label,
        model_parameters=CONF.params, score="EDQ", external=EXTERNAL, deleted_temporal_GEO="GSE159699",
        objective="development GEO-equal apparent AUC", internal_cv=False, candidate_slots_per_domain=10,
        total_candidates=60, additional_candidates=54, path_k=[6, 60], selects_k_by_external=False,
        plan_sha256=digest(OUT / "分析方案_运行前固定.md")), ensure_ascii=False, indent=2), encoding="utf-8")
    put([dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(FILES)], QA / "准备输入SHA256.csv")
    put([dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in paths], QA / "搜索输入SHA256.csv")
    return paths


def collect():
    curves, trials, per_geo, train, external, order_rows, selected_rows, summary, interval_rows = [], [], [], [], [], [], [], [], []
    candidate = read(INPUT / "三脑区60候选及机制与EDQ证据.csv")
    for ri, (region, cn) in enumerate(REGIONS.items()):
        directory = CACHE / region
        lock = json.loads((directory / "开发选择冻结.json").read_text(encoding="utf-8"))
        k = lock["selected_k"]
        q = read(directory / "开发与外部完整轨迹.csv")
        curves.append(q.assign(region=region, external=EXTERNAL[region]))
        trials.append(read(directory / "全部候选试加.csv.gz").assign(region=region))
        per_geo.append(read(directory / "逐步开发各GEO回代AUC.csv").assign(region=region))
        train.append(read(directory / "逐步开发回代预测.csv.gz").assign(region=region))
        z = read(directory / "逐步外部预测.csv.gz", dtype={"key": str})
        external.append(z.assign(region=region, external=EXTERNAL[region]))
        for position, gene in enumerate(lock["order"], 1):
            evidence = candidate.loc[candidate.region.eq(region) & candidate.gene.eq(gene)].iloc[0].to_dict()
            row = dict(greedy_position=position, inclusion_k=6 if position<=6 else position,
                       in_selected_panel=position<=k, **evidence)
            order_rows.append(row)
            if position <= k:
                selected_rows.append(row)
        wide = z.pivot(index="key", columns="k", values="score").sort_index()
        y = z.groupby("key").y.first().loc[wide.index].to_numpy()
        positive, negative = np.flatnonzero(y==1), np.flatnonzero(y==0)
        rng = np.random.default_rng(20261115+ri)
        pi = rng.integers(0, len(positive), size=(4000, len(positive)))
        ni = rng.integers(0, len(negative), size=(4000, len(negative)))
        boots = {}
        for size in sorted({6, k}):
            s = wide[size].to_numpy()
            u = (s[positive, None] > s[negative]).astype(float)+.5*(s[positive, None] == s[negative])
            bs = np.concatenate([u[pi[i:i+100, :, None], ni[i:i+100, None, :]].mean(axis=(1, 2)) for i in range(0, 4000, 100)])
            boots[size] = bs
            lo, hi = np.quantile(bs, [.025, .975])
            interval_rows.append(dict(region=region, kind="base" if size == 6 else "selected", k=size,
                                      auc=roc_auc_score(y, s), lower95=lo, upper95=hi))
        change = boots[k]-boots[6]
        lo, hi = np.quantile(change, [.025, .975])
        base = q.loc[q.k.eq(6)].iloc[0]
        chosen = q.loc[q.k.eq(k)].iloc[0]
        delta = chosen.external_auc-base.external_auc
        interval_rows.append(dict(region=region, kind="selected_minus_base", k=k, auc=delta, lower95=lo, upper95=hi))
        summary.append(dict(region=region, region_cn=cn, external=EXTERNAL[region], selected_k=k, added_genes=k-6,
            dev_base=base.dev_macro, dev_selected=chosen.dev_macro, dev_change=chosen.dev_macro-base.dev_macro,
            external_base=base.external_auc, external_selected=chosen.external_auc, external_change=delta,
            change_lower95=lo, change_upper95=hi, base_genes=";".join(lock["base"]),
            selected_genes=";".join(lock["order"][:k]), added_gene_order=";".join(lock["order"][6:k])))
    put(pd.concat(curves, ignore_index=True), R / "三脑区开发与外部完整轨迹.csv")
    put(pd.concat(trials, ignore_index=True), R / "全部候选试加记录.csv.gz")
    put(pd.concat(per_geo, ignore_index=True), R / "逐步开发各GEO回代AUC.csv")
    put(pd.concat(train, ignore_index=True), R / "逐步开发回代预测.csv.gz")
    put(pd.concat(external, ignore_index=True), R / "逐步外部供体预测.csv.gz")
    put(order_rows, R / "完整60基因添加顺序与机制.csv")
    put(selected_rows, R / "开发选定Panel名单及机制.csv")
    put(summary, R / "底座与开发选定Panel外部AUC汇总.csv")
    put(interval_rows, R / "外部AUC及配对增量_Bootstrap区间.csv")
    (QA / "运行环境.json").write_text(json.dumps(dict(python=sys.version,
        packages={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib"]},
        workers=3, threads_per_worker=1, seed_bootstrap=20261115), indent=2), encoding="utf-8")
    print(pd.DataFrame(summary)[["region", "selected_k", "dev_base", "dev_selected", "external_base", "external_selected", "external_change"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    paths = prepare()
    states = Parallel(n_jobs=3, backend="loky", batch_size=1, pre_dispatch=3)(delayed(fit_region)(str(p)) for p in paths)
    (QA / "三脑区拟合状态.json").write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(s["status"] == "ok" for s in states), [s for s in states if s["status"] != "ok"]
    collect()
