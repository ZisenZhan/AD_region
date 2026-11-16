"""Fold-specific EDQ selection, common-classifier ranking, then external scoring."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parents[1]
EXP = OUT.parent / "别名审计与评分轮换_1107"
sys.path.insert(0, str(EXP))
import 检验与轮换 as old
from 拟合任务 import fit_bundle

SCORE, INPUT, CACHE, RESULT, QA = (OUT / n for n in ("折内评分", "输入与划分", "运行缓存", "结果", "核对"))
for directory in (SCORE, INPUT, CACHE, RESULT, QA):
    directory.mkdir(exist_ok=True)
REGIONS = list(old.regions)
EXTERNAL = dict(zip(REGIONS, ["GSE261817", "GSE125583", "GSE48350"]))
OUTER = tuple(EXTERNAL.values())
MODEL_IDS = list(old.models)
old.STATS = SCORE
FILES = set()


def put(frame, path):
    pd.DataFrame(frame).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def load(path):
    FILES.add(path.resolve())
    return pd.read_pickle(path)


def family(geo):
    return "+".join(sorted(old.base.blocked(geo)))


def all_drops(targets):
    return set.union(*(old.base.blocked(g) for g in targets))


def models_table():
    return pd.DataFrame([dict(model=c.config_id, family=c.family_key, family_label=c.family_label,
                              label=c.display_label, parameters=json.dumps(c.params, sort_keys=True),
                              row_order=i) for i, c in enumerate(old.models.values())])


def choose(pool, scope):
    rank = pool.assign(score=pool.E * pool.D * pool.Q).rename_axis("gene_symbol").reset_index()
    rank = rank.sort_values(["score", "gene_symbol"], ascending=[False, True]).reset_index(drop=True)
    rank["score_rank"] = np.arange(1, len(rank)+1)
    guided = []
    for axis in old.AXES:
        candidates = rank.loc[rank.domains.map(lambda z: axis in z.split(";")), "gene_symbol"]
        gene = next((g for g in candidates if g not in guided), None)
        if gene is None:
            raise ValueError((scope, axis, "No eligible distinct gene; thresholds not relaxed"))
        guided.append(gene)
    arms = ("C", "D") if scope == "pooled" else ("A", "B")
    selected = dict(zip(arms, [guided, rank.gene_symbol.head(6).tolist()]))
    assert all(len(g) == len(set(g)) == 6 for g in selected.values())
    return selected, rank.set_index("gene_symbol")


def prepare():
    start = time.time()
    experiment = old.Experiment()
    full_units = {}
    full_panels = {}
    full_ranks = {}
    source_splits = {}
    history = []
    for scope, targets in [(r, (EXTERNAL[r],)) for r in REGIONS] + [("pooled", OUTER)]:
        code = scope + "_" + "_".join(targets)
        source = EXP / "评分" / (code+"_基因评分.csv.gz")
        FILES.add(source)
        pool = pd.read_csv(source, float_precision="round_trip").set_index("gene_symbol")
        units, _ = experiment.partition(scope, targets)
        split_path = EXP / "逐划分" / (code+"_划分.json")
        FILES.add(split_path)
        saved_split = json.loads(split_path.read_text(encoding="utf-8"))
        assert saved_split["development_units"] == units
        source_splits[scope] = saved_split
        full_units[scope] = units
        full_panels[scope], full_ranks[scope] = choose(pool, scope)
        archive_path = EXP / "逐划分" / (code+"_基因.csv")
        FILES.add(archive_path)
        archive = pd.read_csv(archive_path)
        for arm, genes in full_panels[scope].items():
            expected = archive.loc[archive.formula.eq("EDQ") & archive.arm.eq(arm)].sort_values("slot").gene.tolist()
            assert expected == genes, (scope, arm, genes, expected)
        put(pool.reset_index(), SCORE / (code+"_完整开发评分.csv.gz"))
        prior_path = EXP / "逐划分" / (code+"_AUC.csv")
        FILES.add(prior_path)
        prior = pd.read_csv(prior_path)
        for arm in full_panels[scope]:
            n = prior.loc[prior.formula.eq("EDQ") & prior.arm.eq(arm), "model"].nunique()
            history.append(dict(scope=scope, arm=arm, historical_external_configs=n,
                                exact_new_internal_protocol_previously_found=False,
                                action="reuse checked inputs and final EDQ genes; rerun fits with canonical gene columns"))
    (INPUT / "原单外部划分.json").write_text(json.dumps(source_splits, ensure_ascii=False, indent=2), encoding="utf-8")
    put(history, QA / "历史实验复用核查.csv")
    put(models_table(), INPUT / "36模型配置.csv")
    jobs = {}
    aliases = []
    panels = []
    splitrows = []

    def register(stage, scope, units, queries, selection, ranks, fold, targets):
        assert not set(u.split("__")[1] for u in units) & all_drops(targets)
        assert all(u.split("__")[1] in all_drops(targets) for u in queries)
        for arm, genes in selection.items():
            columns = sorted(genes)
            fingerprint = json.dumps(dict(stage=stage, scope=scope, units=units, genes=columns), sort_keys=True)
            job_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:20]
            if job_id not in jobs:
                jobs[job_id] = dict(job_id=job_id, stage=stage, scope=scope, units=units, genes=columns, queries=[])
            jobs[job_id]["queries"] = sorted(set(jobs[job_id]["queries"]) | set(queries))
            for uid in queries:
                region, geo = uid.split("__")
                aliases.append(dict(stage=stage, region=region, geo=geo, family=family(geo), arm=arm, job_id=job_id, fold=fold))
            for slot, gene in enumerate(genes, 1):
                panels.append(dict(stage=stage, scope=scope, arm=arm, fold=fold, job_id=job_id, slot=slot, gene=gene,
                                   assigned_domain=old.AXES[slot-1] if arm in ("A", "C") else "unconstrained",
                                   **ranks.loc[gene].to_dict()))
            splitrows.append(dict(stage=stage, scope=scope, arm=arm, fold=fold, job_id=job_id,
                                  training_units=";".join(units), test_units=";".join(queries),
                                  excluded_GEOs=";".join(sorted(all_drops(targets))),
                                  candidate_genes=len(ranks), score_training_units=";".join(units)))

    for region in REGIONS:
        grouped = defaultdict(list)
        for uid in full_units[region]:
            grouped[family(uid.split("__")[1])].append(uid)
        for fam, queries in sorted(grouped.items()):
            targets = tuple(dict.fromkeys([EXTERNAL[region], *(u.split("__")[1] for u in queries)]))
            pool, units = experiment.fitpool(region, targets)
            assert units == [u for u in full_units[region] if u.split("__")[1] not in all_drops(targets)]
            selection, ranks = choose(pool, region)
            register("internal", region, units, queries, selection, ranks, region+"__"+fam, targets)
            print("Prepared regional fold", region, fam, "training units", len(units), flush=True)
    pooled_queries = defaultdict(list)
    for region in REGIONS:
        for uid in full_units[region]:
            pooled_queries[family(uid.split("__")[1])].append(uid)
    for fam, queries in sorted(pooled_queries.items()):
        targets = tuple(dict.fromkeys([*OUTER, *(u.split("__")[1] for u in queries)]))
        pool, units = experiment.fitpool("pooled", targets)
        assert units == [u for u in full_units["pooled"] if u.split("__")[1] not in all_drops(targets)]
        selection, ranks = choose(pool, "pooled")
        register("internal", "pooled", units, sorted(queries), selection, ranks, "pooled__"+fam, targets)
        print("Prepared pooled fold", fam, "training units", len(units), flush=True)
    for scope in [*REGIONS, "pooled"]:
        targets = OUTER if scope == "pooled" else (EXTERNAL[scope],)
        queries = [r+"__"+g for r, g in EXTERNAL.items()] if scope == "pooled" else [scope+"__"+EXTERNAL[scope]]
        register("external", scope, full_units[scope], queries, full_panels[scope], full_ranks[scope], "full_development", targets)

    aliases = pd.DataFrame(aliases).drop_duplicates()
    assert not aliases.duplicated(["stage", "region", "geo", "arm"]).any()
    expected_internal = sum(len(full_units[r]) for r in REGIONS) * 4
    assert aliases.stage.eq("internal").sum() == expected_internal
    assert aliases.stage.eq("external").sum() == 12
    put(aliases, INPUT / "拟合任务与策略对应.csv")
    put(panels, INPUT / "每折六基因与EDQ证据.csv")
    put(splitrows, INPUT / "每折训练测试与排除单元.csv")
    (INPUT / "冻结任务.json").write_text(json.dumps(list(jobs.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    roles = []
    for r in REGIONS:
        roles.extend(dict(region=r, geo=u.split("__")[1], family=family(u.split("__")[1]), role="development") for u in full_units[r])
        roles.append(dict(region=r, geo=EXTERNAL[r], family=family(EXTERNAL[r]), role="external"))
    roles = pd.DataFrame(roles)
    put(roles, INPUT / "跨脑区队列用途.csv")
    ext_families = set(roles.loc[roles.role.eq("external"), "family"])
    put(roles.loc[roles.family.isin(ext_families)], QA / "跨脑区外部家族角色审计.csv")
    FILES.update(old.base.DB / f for f in ("区域E.pkl", "合并E.pkl", "可测.pkl"))
    FILES.add(old.base.CACHE / "机制归属.csv.gz")
    FILES.update(EXP / x for x in ("检验与轮换.py", "构建评测输入.py", "模型拟合.py"))
    put([dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for p in sorted(FILES)], QA / "准备阶段来源SHA256.csv")
    print("Prepared", len(jobs), "unique fit jobs;", expected_internal, "internal evaluation cells/model.",
          "Elapsed", round(time.time()-start, 1), "seconds.", flush=True)


def job_inputs(stage):
    jobs = [j for j in json.loads((INPUT / "冻结任务.json").read_text(encoding="utf-8")) if j["stage"] == stage]
    by_uid = defaultdict(set)
    test_uid = defaultdict(set)
    pooled_genes = set()
    for j in jobs:
        if j["scope"] == "pooled":
            pooled_genes.update(j["genes"])
        else:
            for uid in j["units"]:
                by_uid[uid].update(j["genes"])
        for uid in j["queries"]:
            test_uid[uid].update(j["genes"])
    train = {}
    for uid, genes in by_uid.items():
        train[uid] = load(old.DB/(uid+"_train.pkl"))[sorted(genes)], load(old.DB/(uid+"_info.pkl"))
    px = load(old.DB/"合并训练.pkl")[sorted(pooled_genes)] if pooled_genes else None
    pm = load(old.DB/"合并信息.pkl") if pooled_genes else None
    tests = {}
    for uid, genes in test_uid.items():
        tests[uid] = load(old.DB/(uid+"_test.pkl"))[sorted(genes)], load(old.DB/(uid+"_md.pkl"))
    paths = []
    for j in jobs:
        genes = j["genes"]
        if j["scope"] == "pooled":
            mask = (pm.region+"__"+pm.gse).isin(j["units"])
            x = px.loc[mask, genes].copy()
            m = pm.loc[mask, ["y", "weight", "gse"]].copy()
        else:
            x = pd.concat([train[u][0][genes] for u in j["units"]])
            m = pd.concat([train[u][1] for u in j["units"]]).loc[x.index]
        assert x.index.is_unique and x.index.equals(m.index) and set(m.y) == {0, 1}
        query = {u: (tests[u][0][genes].copy(), tests[u][1][["key", "y", "sub"]].copy()) for u in j["queries"]}
        assert all(xx.index.equals(md.index) and xx.notna().any().all() for xx, md in query.values())
        payload = dict(job_id=j["job_id"], stage=stage, train=x, meta=m, tests=query)
        path = CACHE/(j["job_id"]+".pkl")
        pd.to_pickle(payload, path)
        paths.append(path)
    put([dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for p in sorted(FILES)], QA / (stage+"_表达输入SHA256.csv"))
    return paths


def fit_stage(stage, cpus):
    if stage == "external":
        locked = json.loads((RESULT / "开发内部排名_选模冻结.json").read_text(encoding="utf-8"))
        assert locked["selection_uses_external_auc"] is False
        assert locked["best_models"]
    paths = job_inputs(stage)
    bundles = [(path, MODEL_IDS[i:i+3]) for path in paths for i in range(0, 36, 3)]
    start = time.time()
    status = []
    stream = Parallel(n_jobs=cpus, backend="loky", return_as="generator_unordered", batch_size=1,
                      pre_dispatch=cpus*2)(delayed(fit_bundle)(str(path), mids) for path, mids in bundles)
    for done, result in enumerate(stream, 1):
        status.extend(result)
        if done % 12 == 0 or done == len(bundles):
            print(stage, "bundles", done, "/", len(bundles), "failed", sum(z["status"] == "failed" for z in status),
                  "seconds", round(time.time()-start, 1), flush=True)
    (QA / (stage+"_拟合状态.json")).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Finished", stage, "fits", len(status), flush=True)


def collect(stage):
    aliases = pd.read_csv(INPUT / "拟合任务与策略对应.csv")
    aliases = aliases.loc[aliases.stage.eq(stage)]
    predictions = []
    metrics = []
    failures = []
    for job_id, aa in aliases.groupby("job_id"):
        for mid in MODEL_IDS:
            status_file = CACHE/job_id/(mid+"_状态.json")
            if not status_file.exists():
                raise RuntimeError(("Fit not finished", stage, job_id, mid))
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if status["status"] != "ok":
                failures.extend(dict(**v, model=mid, error=status["error"]) for v in aa.to_dict("records"))
                continue
            pred = pd.read_csv(CACHE/job_id/(mid+".csv.gz"), dtype={"key": str}, float_precision="round_trip")
            for row in aa.itertuples():
                uid = row.region+"__"+row.geo
                z = pred.loc[pred.uid.eq(uid)].copy()
                assert len(z) and z.key.is_unique and set(z.y) == {0, 1}
                auc = roc_auc_score(z.y, z.score)
                rec = dict(stage=stage, region=row.region, geo=row.geo, family=row.family, arm=row.arm,
                           model=mid, job_id=job_id, fold=row.fold, n=len(z), AD=int(z.y.sum()), control=int((1-z.y).sum()), auc=auc)
                metrics.append(rec)
                z = z.assign(stage=stage, region=row.region, geo=row.geo, arm=row.arm, family=row.family, job_id=job_id, fold=row.fold)
                predictions.append(z)
    metric = pd.DataFrame(metrics)
    put(metric, RESULT/(stage+"_逐GEO_AUC.csv"))
    put(pd.concat(predictions, ignore_index=True), RESULT/(stage+"_逐供体预测.csv.gz"))
    put(failures, QA/(stage+"_失败记录.csv"))
    return metric


def select():
    metric = collect("internal")
    expected = pd.read_csv(INPUT / "拟合任务与策略对应.csv")
    expected = expected.loc[expected.stage.eq("internal")].groupby(["region", "arm"]).size()
    family_mean = metric.groupby(["region", "arm", "model", "family"], as_index=False).auc.mean()
    macro = family_mean.groupby(["region", "arm", "model"], as_index=False).agg(internal_macro=("auc", "mean"), families=("family", "size"))
    counts = metric.groupby(["region", "arm", "model"]).size().rename("n_test_GEOs").reset_index()
    macro = macro.merge(counts)
    macro["expected_test_GEOs"] = [expected.loc[(r, a)] for r, a in zip(macro.region, macro.arm)]
    macro["complete"] = macro.n_test_GEOs.eq(macro.expected_test_GEOs)
    macro.loc[~macro.complete, "internal_macro"] = np.nan
    put(family_mean, RESULT / "开发内部_逐家族AUC.csv")
    put(macro, RESULT / "开发内部_脑区策略Macro.csv")
    ranking = models_table()
    score = macro.groupby("model").agg(selection_score=("internal_macro", "mean"), valid_cells=("internal_macro", "count"))
    ranking = ranking.merge(score, left_on="model", right_index=True, how="left")
    ranking["eligible"] = ranking.valid_cells.eq(12)
    ranking.loc[~ranking.eligible, "selection_score"] = np.nan
    ranking["rank"] = ranking.selection_score.rank(method="min", ascending=False)
    ranking = ranking.sort_values(["rank", "row_order"], na_position="last")
    put(ranking, RESULT / "模型总排名_开发内部四策略三脑区.csv")
    best_score = ranking.selection_score.max()
    winners = ranking.loc[ranking.selection_score.ge(best_score-1e-12), "model"].tolist()
    assert winners
    frozen = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  primary_score="mean of 12 region-by-arm development family-macro AUCs",
                  selection_uses_external_auc=False, selection_uses_train_apparent_auc=False,
                  selection_uses_A_minus_B=False, best_models=winners, best_score=float(best_score),
                  ranking_sha256=hashlib.sha256((RESULT/"模型总排名_开发内部四策略三脑区.csv").read_bytes()).hexdigest(),
                  internal_predictions_sha256=hashlib.sha256((RESULT/"internal_逐供体预测.csv.gz").read_bytes()).hexdigest())
    (RESULT / "开发内部排名_选模冻结.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(ranking[["rank", "model", "label", "selection_score", "valid_cells"]].head(10).to_string(index=False), flush=True)


def summarize():
    external = collect("external")
    macro = pd.read_csv(RESULT / "开发内部_脑区策略Macro.csv", float_precision="round_trip")
    table = macro.merge(external[["region", "arm", "model", "geo", "auc"]].rename(columns={"geo": "external_GEO", "auc": "external_auc"}),
                        on=["region", "arm", "model"], how="outer")
    table = table.merge(models_table(), on="model", how="left")
    table["internal_rank"] = table.groupby(["region", "arm"]).internal_macro.rank(method="min", ascending=False)
    table["external_descriptive_rank"] = table.groupby(["region", "arm"]).external_auc.rank(method="min", ascending=False)
    put(table, RESULT / "36配置_三脑区四策略完整结果.csv")
    put(table.loc[table.arm.eq("A")].sort_values(["region", "internal_rank", "row_order"]), RESULT / "模型排名_各脑区六机制底座A.csv")
    regional = table.groupby(["region", "model"], as_index=False).agg(internal_four_arm_mean=("internal_macro", "mean"),
                                                                 valid_internal_arms=("internal_macro", "count"),
                                                                 external_four_arm_mean=("external_auc", "mean"))
    regional.loc[regional.valid_internal_arms.ne(4), "internal_four_arm_mean"] = np.nan
    regional["rank"] = regional.groupby("region").internal_four_arm_mean.rank(method="min", ascending=False)
    put(regional.merge(models_table(), on="model").sort_values(["region", "rank", "row_order"]), RESULT / "模型排名_各脑区四策略平均.csv")
    locked = json.loads((RESULT / "开发内部排名_选模冻结.json").read_text(encoding="utf-8"))
    put(table.loc[table.model.isin(locked["best_models"])], RESULT / "选中配置_三脑区四策略结果.csv")
    apparent = []
    aliases = pd.read_csv(INPUT / "拟合任务与策略对应.csv")
    for row in aliases.loc[aliases.stage.eq("external")].itertuples():
        for mid in MODEL_IDS:
            path = CACHE/row.job_id/(mid+"_开发回代.csv.gz")
            if path.exists():
                z = pd.read_csv(path, float_precision="round_trip")
                apparent.append(dict(region=row.region, arm=row.arm, model=mid, train_apparent_auc=roc_auc_score(z.y, z.score),
                                     train_rows=len(z), train_scope="six_region_pooled" if row.arm in ("C", "D") else row.region))
    put(apparent, RESULT / "开发全集回代AUC_仅供参考.csv")
    for region in REGIONS:
        view = table.loc[table.region.eq(region)&table.arm.eq("A")].sort_values("internal_rank")
        print(region, "A-group development top 5:\n", view[["model", "label", "internal_macro", "external_auc"]].head(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "internal", "select", "external", "summarize"])
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == "prepare":
            prepare()
        elif args.stage in ("internal", "external"):
            fit_stage(args.stage, args.cpus)
        elif args.stage == "select":
            select()
        else:
            summarize()
