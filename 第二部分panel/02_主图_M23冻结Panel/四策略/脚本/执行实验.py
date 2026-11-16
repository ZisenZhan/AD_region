"""Complete finite deletion-by-temporal-holdout matrix with EDQ and M23 fixed."""
from pathlib import Path
from datetime import datetime, timezone
import os
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import json
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed

OUT = Path(__file__).resolve().parents[1]
PREV = OUT.parent / "EDQ_单外部_36模型开发内选模_20261113"
ROTATION = OUT.parent / "EDQ_颞叶外部队列轮换_20261113"
ONE = OUT.parent / "M23_EDQ_颞叶随机删一开发队列_20261115"
sys.path.insert(0, str(PREV / "脚本"))
import 运行分析 as ref
from 拟合任务 import fit_bundle
old = ref.old
REGIONS = ref.REGIONS
T = "temporal_neocortex"
DELETIONS = ["GSE118553", "GSE122063", "GSE159699", "GSE36980", "GSE37263"]
HOLDOUTS = sorted(DELETIONS+["GSE125583", "GSE132903"])
INPUT, SCORE, CACHE, R, QA = [OUT / n for n in ("输入与划分", "评分", "运行缓存", "结果", "核对")]
for p in (INPUT, SCORE, CACHE, R, QA):
    p.mkdir(exist_ok=True)
FILES = set()


def put(z, p):
    pd.DataFrame(z).to_csv(p, index=False, encoding="utf-8-sig", float_format="%.17g")


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    FILES.add(Path(p))
    return pd.read_pickle(p)


def read(p, **kwargs):
    FILES.add(Path(p))
    return pd.read_csv(p, float_precision="round_trip", **kwargs)


class DeletedExperiment(old.Experiment):
    def __init__(self, deleted):
        super().__init__()
        self.extra_drop = old.base.blocked(deleted)

    def partition(self, scope, targets):
        units, excluded = super().partition(scope, targets)
        return [u for u in units if u.split("__")[1] not in self.extra_drop], excluded | self.extra_drop


def main():
    combinations = []
    for deleted in DELETIONS:
        drop = old.base.blocked(deleted)
        for holdout in HOLDOUTS:
            if holdout not in drop:
                combinations.append(dict(case_id=deleted+"__"+holdout, deleted_GEO=deleted,
                                         excluded_deletion_family=sorted(drop), temporal_external=holdout))
    assert len(combinations) == 30
    freeze = INPUT / "三十组合_运行前冻结.json"
    if freeze.exists():
        assert json.loads(freeze.read_text(encoding="utf-8"))["combinations"] == combinations
    else:
        freeze.write_text(json.dumps(dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(), model="M23",
            model_components=2, score="EDQ", combinations=combinations, new_internal_cv=False), ensure_ascii=False, indent=2), encoding="utf-8")
    put(combinations, INPUT / "完整删除与留出矩阵.csv")
    jobs, aliases, evidence, splits, score_manifest = {}, [], [], [], []
    for deleted in DELETIONS:
        ex = DeletedExperiment(deleted)
        assert ex.roster[T] == HOLDOUTS
        old.STATS = SCORE / deleted
        old.STATS.mkdir(exist_ok=True)
        for case in [c for c in combinations if c["deleted_GEO"] == deleted]:
            holdout = case["temporal_external"]
            trio = ("GSE261817", holdout, "GSE48350")
            for scope, targets, queries in [
                (T, (holdout,), [T+"__"+holdout]),
                ("pooled", trio, [r+"__"+g for r, g in zip(REGIONS, trio)]),
            ]:
                pool, units = ex.fitpool(scope, targets)
                selected, ranks = ref.choose(pool, scope)
                excluded = ex.partition(scope, targets)[1]
                assert pool.D.ge(2/3).all() and pool.families.ge(2).all()
                assert not {u.split("__")[1] for u in units} & excluded
                assert ex.av.loc[pool.index, units].all().all()
                rankpath = old.STATS / (case["case_id"]+"__"+scope+"_EDQ排名.csv.gz")
                put(ranks.reset_index(), rankpath)
                score_manifest.append(dict(case_id=case["case_id"], scope=scope, path=str(rankpath), candidate_genes=len(pool)))
                for arm, genes in selected.items():
                    assert len(genes) == len(set(genes)) == 6
                    for uid in queries:
                        assert set(genes) <= old.static(uid.split("__")[1])
                    canonical = sorted(genes)
                    token = hashlib.sha256(json.dumps([scope, units, canonical]).encode()).hexdigest()[:20]
                    jobs.setdefault(token, dict(job_id=token, scope=scope, units=units, genes=canonical, queries=[]))
                    jobs[token]["queries"] = sorted(set(jobs[token]["queries"]) | set(queries))
                    for uid in queries:
                        region, geo = uid.split("__")
                        aliases.append(dict(case_id=case["case_id"], deleted_GEO=deleted, temporal_external=holdout,
                                            region=region, geo=geo, arm=arm, job_id=token))
                    for slot, gene in enumerate(genes, 1):
                        evidence.append(dict(case_id=case["case_id"], deleted_GEO=deleted, temporal_external=holdout,
                            scope=scope, arm=arm, slot=slot, gene=gene, job_id=token,
                            assigned_domain=old.AXES[slot-1] if arm in ("A", "C") else "unconstrained", **ranks.loc[gene].to_dict()))
                    splits.append(dict(case_id=case["case_id"], deleted_GEO=deleted, temporal_external=holdout,
                                       scope=scope, arm=arm, job_id=token, training_units=";".join(units),
                                       excluded_GEOs=";".join(sorted(excluded)), test_units=";".join(queries), candidate_genes=len(pool)))
            print("Prepared", case["case_id"], flush=True)
    alias = pd.DataFrame(aliases)
    assert len(alias) == 30*8 and not alias.duplicated(["case_id", "region", "arm"]).any()
    put(alias, INPUT / "实际拟合与四策略对应.csv")
    put(evidence, INPUT / "各划分六基因及EDQ证据.csv")
    put(splits, INPUT / "实际训练测试与共同排除.csv")
    put(score_manifest, INPUT / "完整EDQ表索引.csv")
    (INPUT / "冻结拟合任务.json").write_text(json.dumps(list(jobs.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    needed, tested, pooled = defaultdict(set), defaultdict(set), set()
    for j in jobs.values():
        if j["scope"] == "pooled":
            pooled.update(j["genes"])
        else:
            for uid in j["units"]:
                needed[uid].update(j["genes"])
        for uid in j["queries"]:
            tested[uid].update(j["genes"])
    training = {u: (load(old.DB / (u+"_train.pkl"))[sorted(genes)], load(old.DB / (u+"_info.pkl"))) for u, genes in needed.items()}
    tests = {u: (load(old.DB / (u+"_test.pkl"))[sorted(genes)], load(old.DB / (u+"_md.pkl"))) for u, genes in tested.items()}
    px, pm = load(old.DB / "合并训练.pkl")[sorted(pooled)], load(old.DB / "合并信息.pkl")
    counts, train_manifest, test_manifest = [], [], []
    for token, j in jobs.items():
        genes = j["genes"]
        if j["scope"] == "pooled":
            keep = (pm.region+"__"+pm.gse).isin(j["units"])
            x, m = px.loc[keep, genes].copy(), pm.loc[keep, ["y", "weight", "gse"]].copy()
        else:
            x = pd.concat([training[u][0][genes] for u in j["units"]])
            m = pd.concat([training[u][1] for u in j["units"]]).loc[x.index]
        assert x.index.is_unique and x.index.equals(m.index) and set(m.y) == {0, 1}
        assert set(m.gse) == {u.split("__")[1] for u in j["units"]}
        q = {u: (tests[u][0][genes].copy(), tests[u][1][["key", "y", "sub"]].copy()) for u in j["queries"]}
        for uid, (xx, md) in q.items():
            assert xx.index.equals(md.index) and xx.notna().any().all()
            assert not set(m.gse) & old.base.blocked(uid.split("__")[1])
            test_manifest.append(md.assign(job_id=token, uid=uid, expression_row_id=md.index.astype(str)).reset_index(drop=True))
        pd.to_pickle(dict(job_id=token, stage="external", train=x, meta=m, tests=q), CACHE / (token+".pkl"))
        counts.append(dict(job_id=token, scope=j["scope"], training_rows=len(m), AD=int(m.y.sum()),
                           control=int((1-m.y).sum()), training_GEOs=m.gse.nunique(), training_units=len(j["units"])))
        train_manifest.append(m.assign(job_id=token, training_row_id=m.index.astype(str)).reset_index(drop=True))
    put(counts, INPUT / "各实际训练集人数.csv")
    put(pd.concat(train_manifest, ignore_index=True), INPUT / "逐训练供体组织记录.csv.gz")
    put(pd.concat(test_manifest, ignore_index=True), INPUT / "逐测试样本及分组.csv.gz")
    tasks = [str(CACHE / (j["job_id"]+".pkl")) for j in jobs.values()]
    states = []
    stream = Parallel(n_jobs=3, backend="loky", return_as="generator_unordered", batch_size=1, pre_dispatch=3)(
        delayed(fit_bundle)(path, ["M23"]) for path in tasks)
    for i, state in enumerate(stream, 1):
        states.extend(state)
        if i % 12 == 0 or i == len(tasks):
            print("Fitted", i, "/", len(tasks), "failed", sum(s["status"] != "ok" for s in states), flush=True)
    (QA / "全部拟合状态.json").write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(s["status"] == "ok" for s in states)
    rows, predictions = [], []
    for token, group in alias.groupby("job_id"):
        pred = read(CACHE / token / "M23.csv.gz", dtype={"key": str})
        for row in group.itertuples():
            uid = row.region+"__"+row.geo
            z = pred.loc[pred.uid.eq(uid)].copy()
            assert z.key.is_unique and set(z.y) == {0, 1}
            auc = roc_auc_score(z.y, z.score)
            a, b = z.loc[z.y.eq(1), "score"].to_numpy()[:, None], z.loc[z.y.eq(0), "score"].to_numpy()
            assert abs(auc-((a>b)+.5*(a==b)).mean()) < 1e-12
            rows.append(dict(case_id=row.case_id, deleted_GEO=row.deleted_GEO, temporal_external=row.temporal_external,
                             region=row.region, geo=row.geo, arm=row.arm, auc=auc, n=len(z), AD=int(z.y.sum()), control=int((1-z.y).sum())))
            predictions.append(z.assign(case_id=row.case_id, deleted_GEO=row.deleted_GEO, temporal_external=row.temporal_external,
                                         region=row.region, geo=row.geo, arm=row.arm))
    baseline_pred = read(ROTATION / "结果/所有轮换逐供体预测.csv.gz", dtype={"key": str})
    baseline_pred = baseline_pred.loc[baseline_pred.model.eq("M23")].copy()
    baseline_metric = read(ROTATION / "结果/所有轮换逐策略外部AUC.csv")
    baseline_metric = baseline_metric.loc[baseline_metric.model.eq("M23")].copy()
    for case in combinations:
        holdout = case["temporal_external"]
        refpred = baseline_pred.loc[baseline_pred.temporal_external.eq(holdout) & ~baseline_pred.region.eq(T) & baseline_pred.arm.isin(["A", "B"])]
        refmetric = baseline_metric.loc[baseline_metric.temporal_external.eq(holdout) & ~baseline_metric.region.eq(T) & baseline_metric.arm.isin(["A", "B"])]
        for row in refmetric.itertuples():
            rows.append(dict(case_id=case["case_id"], deleted_GEO=case["deleted_GEO"], temporal_external=holdout,
                             region=row.region, geo=row.geo, arm=row.arm, auc=row.auc, n=row.n, AD=row.AD, control=row.control))
        predictions.append(refpred[["key", "y", "score", "region", "geo", "arm", "model"]].assign(
            case_id=case["case_id"], deleted_GEO=case["deleted_GEO"], temporal_external=holdout))
    metric = pd.DataFrame(rows)
    pred = pd.concat(predictions, ignore_index=True)
    assert len(metric) == 360 and not pred.duplicated(["case_id", "region", "arm", "key"]).any()
    put(metric, R / "全部组合逐策略外部AUC.csv")
    put(pred, R / "全部组合逐供体预测.csv.gz")
    persisted = read(R / "全部组合逐供体预测.csv.gz", dtype={"key": str})
    expected = metric.set_index(["case_id", "region", "arm"]).auc
    errors, sample_checks = [], []
    for (caseid, region), w in persisted.groupby(["case_id", "region"]):
        ids = w[["key", "y"]].drop_duplicates().sort_values("key").reset_index(drop=True)
        for arm, part in w.groupby("arm"):
            assert part[["key", "y"]].sort_values("key").reset_index(drop=True).equals(ids)
            errors.append(abs(roc_auc_score(part.y, part.score)-expected.loc[(caseid, region, arm)]))
        holdout = w.temporal_external.iloc[0]
        refids = baseline_pred.loc[baseline_pred.temporal_external.eq(holdout) & baseline_pred.region.eq(region), ["key", "y"]].drop_duplicates().sort_values("key").reset_index(drop=True)
        assert ids.equals(refids)
        sample_checks.append(dict(case_id=caseid, region=region, n=len(ids), AD=int(ids.y.sum()), control=int((1-ids.y).sum()), same_as_prior_holdout=True))
    assert max(errors) < 1e-12
    put(sample_checks, QA / "同一验证队列供体一致性.csv")
    comparisons = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")]
    wide = metric.pivot(index=["case_id", "deleted_GEO", "temporal_external", "region"], columns="arm", values="auc").reset_index()
    for left, right in comparisons:
        key = left+">"+right
        wide[key+"_difference"] = wide[left]-wide[right]
        wide[key] = wide[key+"_difference"] > 1e-12
    flags = [l+">"+r for l, r in comparisons]
    wide["conditions_passed"] = wide[flags].sum(axis=1)
    put(wide, R / "三脑区全部四条件核查.csv")
    temporal = wide.loc[wide.region.eq(T)].copy()
    before = baseline_metric.loc[baseline_metric.region.eq(T)].pivot(index="temporal_external", columns="arm", values="auc")
    for arm in "ABCD":
        temporal["before_"+arm] = temporal.temporal_external.map(before[arm])
        temporal["change_"+arm] = temporal[arm]-temporal["before_"+arm]
    put(temporal, R / "颞叶三十组合_四条件与删除前后变化.csv")
    winners = temporal.loc[temporal.conditions_passed.eq(4)]
    put(winners, R / "颞叶四条件全部满足的组合.csv")
    total = wide.groupby(["case_id", "deleted_GEO", "temporal_external"], as_index=False).agg(
        total_passed=("conditions_passed", "sum"), regions_evaluated=("region", "size"))
    total["all_three_pass"] = total.total_passed.eq(12) & total.regions_evaluated.eq(3)
    put(total, R / "三脑区十二条件汇总.csv")
    put(total.loc[total.all_three_pass], R / "三脑区全部满足的组合.csv")
    prior = read(ONE / "结果/删除前后外部AUC对照.csv")
    current = metric.loc[metric.case_id.eq("GSE159699__GSE125583")]
    reproduction = current.merge(prior, on=["region", "arm"], suffixes=("_new", "_old"))
    assert len(reproduction) == 12
    max_reproduction_error = float((reproduction.auc_new-reproduction.auc_old).abs().max())
    assert max_reproduction_error < 1e-12
    put(reproduction.assign(difference=reproduction.auc_new-reproduction.auc_old), QA / "既有随机删除十二项复现.csv")
    onepred = read(ONE / "结果/删除后_ABCD外部逐供体预测.csv.gz", dtype={"key": str})
    here = persisted.loc[persisted.case_id.eq("GSE159699__GSE125583")]
    match = here.merge(onepred, on=["region", "arm", "key", "y"], suffixes=("_new", "_old"))
    assert len(match) == len(here) and np.allclose(match.score_new, match.score_old, atol=1e-14, rtol=0)
    summary = temporal.groupby("deleted_GEO", as_index=False).agg(
        combinations=("case_id", "size"), max_conditions=("conditions_passed", "max"),
        four_of_four=("conditions_passed", lambda s: int(s.eq(4).sum())))
    put(summary, R / "按删除队列汇总.csv")
    (QA / "数值与范围验收.json").write_text(json.dumps(dict(
        model="M23", components=2, formula="EDQ", deletion_options=5, holdout_options=7,
        planned_combinations=30, completed_combinations=len(temporal), unique_fits=len(jobs),
        auc_cells=len(metric), saved_prediction_max_auc_error=max(errors),
        prior_single_deletion_max_auc_error=max_reproduction_error, prior_single_deletion_predictions_identical=True,
        same_test_donors_and_labels=True, all_training_units_exclude_test_and_deletion_families=True,
        no_new_internal_cv=True, failed_fits=0), indent=2), encoding="utf-8")
    FILES.update(old.DB / n for n in ["区域E.pkl", "合并E.pkl", "可测.pkl"])
    FILES.update([old.CACHE / "机制归属.csv.gz", Path(__file__).resolve(), OUT / "分析方案_运行前固定.md", PREV / "脚本/拟合任务.py", old.base.HERE / "检验与轮换.py"])
    put([dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(FILES)], QA / "来源与实现SHA256.csv")
    report(temporal, total, summary, winners)
    print(summary.to_string(index=False), flush=True)
    print("Temporal 4/4 combinations", len(winners), flush=True)
    print(winners[["deleted_GEO", "temporal_external", "A", "B", "C", "D"]].to_string(index=False), flush=True)
    print("All three 12/12 combinations", int(total.all_three_pass.sum()), flush=True)


def markdown(df, cols):
    lines = ["| "+" | ".join(cols)+" |", "| "+" | ".join("---" for c in cols)+" |"]
    for row in df[cols].itertuples(index=False, name=None):
        lines.append("| "+" | ".join(f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in row)+" |")
    return "\n".join(lines)


def report(temporal, total, summary, winners):
    cols = ["deleted_GEO", "temporal_external", "A", "B", "C", "D", "conditions_passed"]
    lines = ["# M23＋EDQ：颞叶删一与外部轮换结果", "", "## 范围与结果", "",
        f"完整计算5个删除对象×6个剩余外部候选，共30个组合；固定M23两成分和EDQ。颞叶满足四条件的有{len(winners)}个组合；三脑区同时满足12条件的有{int(total.all_three_pass.sum())}个。", "",
        markdown(summary, ["deleted_GEO", "combinations", "max_conditions", "four_of_four"]), "",
        "## 颞叶满足四条件的组合", "", markdown(winners, cols), "",
        "## 全部30组合", "", markdown(temporal.sort_values(["deleted_GEO", "temporal_external"]), cols), "",
        "## 口径", "",
        "删除对象为原GSE125583外部方案中的5个开发GEO。每次删除一个及其关联家族，再从完整7个颞叶GEO中选择未删除者作为外部。每个组合在实际开发集重新计算E/D和训练内SumRank Q，再选择六基因并训练，不使用外部标签评分。未增加核心BP门槛、FDR硬筛选或新的评分公式。", "",
        "额叶外部GSE261817、海马外部GSE48350固定；其区域A/B保留原值作参照。合并C/D同时排除删除家族和三个外部家族，在所有训练脑区统一排除。C/D每种策略使用同一模型和同一份名单预测三个外部脑区，不按区域挑选不同合并Panel。", "",
        "静态平台可测性仍只根据实际验证队列定义，不额外要求基因在被删队列中检出。保留原同源家族处理、百分位输入、训练内填补和标准化及M23权重处理。训练样本数在合并模型中表示供体×组织记录，不冒充独立供体数。", "",
        "每种比较均为单个外部队列的供体级AUC，不是交叉验证Macro。四关系按完整精度差值>1e-12判断，不把点估计大小称为统计显著。没有模型选择、额外内部交叉验证、二次删除或按结果新增外部队列。", "",
        "## 核对与保存", "",
        "保存全部组合和不利结果、开发及测试名单、完整EDQ表、各六基因证据、实际拟合输入、逐供体预测及逐策略AUC。30组合共360个AUC均从保存预测复算，并以病例—对照两两比较核对；同一外部供体和标签在四策略中完全相同。既有GSE159699删除＋GSE125583外部的12个AUC及预测已复现。", "",
        "这次系统考察了删除对象和外部划分，结果属于回顾性敏感性探索；不能把其中较有利的组合当作全新盲测证据，也不自动替换正式队列名单。", ""]
    (OUT / "结果说明.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
