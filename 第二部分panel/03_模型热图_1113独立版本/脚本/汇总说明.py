"""Create reader-facing Chinese results and ranking tables from saved outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
RESULT = OUT / "结果"
REGIONS = {"frontal_prefrontal": "额叶/前额叶", "temporal_neocortex": "颞叶新皮层", "hippocampus": "海马"}


def read(p):
    return pd.read_csv(p, float_precision="round_trip")


def md(frame, columns, labels=None, digits=6):
    labels = labels or columns
    rows = ["| "+" | ".join(labels)+" |", "| "+" | ".join("---" for c in columns)+" |"]
    for values in frame[columns].itertuples(index=False, name=None):
        cells = []
        for v in values:
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.{digits}f}" if np.isfinite(v) else "NA")
            else:
                cells.append(str(v).replace("|", ";"))
        rows.append("| "+" | ".join(cells)+" |")
    return "\n".join(rows)


def main():
    tab = read(RESULT / "36配置_三脑区四策略完整结果.csv")
    rank = read(RESULT / "模型总排名_开发内部四策略三脑区.csv")
    models = read(OUT / "输入与划分/36模型配置.csv")
    locked = json.loads((RESULT / "开发内部排名_选模冻结.json").read_text(encoding="utf-8"))
    best = locked["best_models"]
    a_mean = tab.loc[tab.arm.eq("A")].groupby("model", as_index=False).agg(A_internal_three_region_mean=("internal_macro", "mean"))
    a_mean["rank_A_internal"] = a_mean.A_internal_three_region_mean.rank(method="min", ascending=False).astype(int)
    a_mean = a_mean.merge(models, on="model").sort_values(["rank_A_internal", "row_order"])
    a_mean.to_csv(RESULT / "模型排名_三脑区六机制底座A平均.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    main_rank = rank[["rank", "model", "label", "selection_score"]].rename(columns={"rank": "总排名", "model": "模型编号", "label": "模型配置", "selection_score": "开发内部12格平均AUC"})
    main_rank["总排名"] = main_rank["总排名"].astype(int)
    main_rank.to_csv(RESULT / "模型总排名_中文.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    single = tab.loc[tab.arm.eq("A")].copy()
    single["脑区"] = single.region.map(REGIONS)
    single["开发内部排名"] = single.internal_rank.astype(int)
    single["外部描述排名"] = single.external_descriptive_rank.astype(int)
    localrank = single[["脑区", "model", "label", "internal_macro", "开发内部排名", "external_GEO", "external_auc", "外部描述排名"]].rename(
        columns={"model": "模型编号", "label": "模型配置", "internal_macro": "开发内部MacroAUC", "external_GEO": "外部队列", "external_auc": "外部AUC"})
    localrank.sort_values(["脑区", "开发内部排名", "模型编号"]).to_csv(RESULT / "各脑区六机制底座模型排名_中文.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    differences = []
    for mid in best:
        for r in REGIONS:
            z = tab.loc[tab.model.eq(mid) & tab.region.eq(r)].set_index("arm")
            values = z.external_auc.to_dict()
            differences.append(dict(model=mid, region=r, **values, **{x+"_minus_"+y: values[x]-values[y] for x, y in ("AB", "CD", "AC", "BD")}))
    diff = pd.DataFrame(differences)
    diff.to_csv(RESULT / "选中配置_外部四策略差值.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    counts = read(OUT / "核对/测试供体与各策略样本一致性.csv")
    counts["脑区"] = counts.region.map(REGIONS)
    final_genes = read(OUT / "输入与划分/每折六基因与EDQ证据.csv")
    final_genes = final_genes.loc[final_genes.stage.eq("external")].sort_values(["scope", "arm", "slot"])
    final_genes.to_csv(RESULT / "最终六基因及EDQ证据.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    winners = single.loc[single.model.isin(best)].copy()
    lines = ["# EDQ单外部：36模型比较结果", "", "日期：2026年11月13日。所有36配置已计算完成，原有论文、图件和结果未覆盖。", "",
             "## 以前是否做过", "",
             "做过相近版本，没有找到这套完整版本。相同单外部划分的EDQ区域A/B曾各算过36配置，合并C/D此前只算过9配置；旧多模型图使用过不同输入、不同外部划分或固定名单。此次补齐的是当前划分下、每折重新计算EDQ并选六基因、四策略共同参与开发内部选模的完整实验。", "",
             "## 固定设置", "",
             "评分为E×D×Q。E为训练来源家族有符号Hedges' g中位数的绝对值，D为方向一致率，Q=−log10(SumRank FDR)。各折独立重算这三项，沿用至少2个有效来源家族、D≥2/3及可测性条件。本版不另加FDR<0.05硬门槛。", "",
             "机制划分保留当前六个领域：Synapse、Immune Response、Lipid Metabolism、Mitochondrial Metabolism、Structural Stabilization、Proteostasis。沿用BP/MF优先、CC回退和最高票并列多归属；按固定机制顺序选择六个不同基因。", "",
             "| 策略 | 训练范围 | 六基因选择 |", "| --- | --- | --- |",
             "| A | 对应脑区 | 每机制选评分最高且未重复的1个 |", "| B | 对应脑区 | 相同资格池直接取评分前6，不限制机制 |",
             "| C | 六脑区合并 | 每机制选评分最高且未重复的1个 |", "| D | 六脑区合并 | 相同资格池直接取评分前6，不限制机制 |", "",
             "额叶外部为GSE261817，颞叶为GSE125583，海马为GSE48350。原开发集合不变。区域开发分别为9、5、5个GEO；内部评价按关联GEO共同留出。海马GSE173954/GSE36980共同留出，计算Macro时先对该家族内两个GEO的AUC平均，再与其他家族等权平均。", "",
             "训练与测试继续使用既有标签盲百分位输入，填补和标准化只在本折训练部分拟合，不使用ComBat。技术重复和多个亚区按原规则合并。分类器的实际特征列按基因名字母序统一；生物学挑选次序另行保留。", "",
             "本轮内部AUC同时评价选基因流程和分类器，因为每一折都会重新选基因；外部模型使用完整开发集得到的固定六基因。内部每一格并不必然对应最后六个固定基因。", "",
             "### 外部评价人数", "",
             md(counts.loc[counts.stage.eq("external")], ["脑区", "geo", "n", "AD", "control"], ["脑区", "外部队列", "供体级记录", "AD", "对照"]), "",
             "## 统一模型总排名", "",
             "主排名分数为3脑区×4策略的12个开发内部家族等权Macro AUC的算术平均。内部结果完成后先锁定排名，再进行本轮外部拟合。训练回代AUC和外部AUC不计入此分数。", "",
             md(main_rank, main_rank.columns.tolist()), "",
             "主排名第一为"+"、".join(best)+"，分数"+f"{locked['best_score']:.6f}"+"。这是四策略共享模型的排名，不等同于只对六机制A组排名。", "",
             "### 排名第一配置在六机制底座A的表现", "",
             md(winners, ["脑区", "model", "internal_macro", "开发内部排名", "external_auc", "外部描述排名"], ["脑区", "配置", "开发内部Macro AUC", "A组内部名次", "外部AUC", "A组外部描述名次"]), "",
             "PLS-logistic（2个成分）在三脑区四策略平均中第一；只看A组时，额叶内部排名第22、颞叶第11、海马第3。前五个共享配置分数相差约0.0013，排名接近。", "",
             "## 各脑区六机制底座A的前五名", ""]
    for r, label in REGIONS.items():
        part = single.loc[single.region.eq(r)].sort_values(["internal_rank", "row_order"]).head(5)
        lines.extend(["### "+label, "", md(part, ["开发内部排名", "model", "label", "internal_macro", "external_auc"], ["内部名次", "配置", "模型", "开发内部Macro AUC", "外部AUC"]), ""])
    lines.extend(["## 排名第一配置的四策略结果", ""])
    selected = tab.loc[tab.model.isin(best)].copy()
    selected["脑区"] = selected.region.map(REGIONS)
    lines.extend([md(selected, ["脑区", "arm", "internal_macro", "external_auc"], ["脑区", "策略", "开发内部Macro AUC", "外部AUC"]), "",
                  "外部AUC中，额叶和海马的四个关系A>B、C>D、A>C、B>D均满足；颞叶满足A>B与A>C，C>D和B>D未满足。颞叶A−B仅约0.0036、海马约0.0012，均为观察差值，未作差值显著性检验。主排名没有以满足这些关系作为条件。", "",
                  "## 最终六基因", "", "| 范围 | 策略 | 基因，按生物学挑选顺序 |", "| --- | --- | --- |"])
    for (scope, arm), p in final_genes.groupby(["scope", "arm"], sort=False):
        lines.append("| "+REGIONS.get(scope, "六脑区合并")+" | "+arm+" | "+"、".join(p.gene)+" |")
    lines.extend(["", "上述名单复现已有单外部EDQ名单，未因本次模型排名更改。每折名单、EDQ各项及所有机制资格见完整证据表。", "",
                  "## 图件阅读", "",
                  "三张主热力图各展示一个脑区的A组。每行为一个配置，按模型家族和原参数顺序排列，不按AUC重排。左侧GEO列为该队列或关联家族留出、训练内重新选基因后的AUC；Dev macro是家族等权内部平均；External为最右指定外部队列AUC。三个脑区使用相同0至1色标。", "",
                  "三张补充热图按A、B、C、D并列，每组各显示内部Dev与External两列。PNG用于预览，SVG/PDF保留可编辑文本。数字的完整精度另存CSV。图内不加入方法说明小字。", "",
                  "## 核查与解释", "",
                  "本次完成2232个内部拟合和288个最终开发全集拟合，得到2736个逐GEO内部AUC与432个外部AUC。所有模型拟合成功；保存的逐供体预测可重算全部AUC，并与正负样本两两比较算法核对。主排名冻结在本轮外部预测完成之前。", "",
                  "海马A组内部Macro最高约0.620，明显低于额叶和颞叶，提示当前机制约束与训练内EDQ选基因流程在海马开发队列间的迁移较弱。外部约0.81的结果应与这项内部结果一并呈现。此轮不再根据外部表现更改评分、队列或参数。", "",
                  "保留原划分也保留其跨脑区角色：GSE48350在额叶为开发、在海马为外部，颞叶GSE159699与GSE48350存在既有来源关联；合并C/D开发集合则全局排除了指定外部GEO及关联系列。因此这批结果是既有数据上的回顾性选模与外部队列比较，不能表述为三个外部家族对整个选模历史完全盲测。", "",
                  "## 文件索引", "",
                  "- 图件：3张A组主图、3张四策略补充图及热图源数据。",
                  "- 结果：中文总排名、各脑区A组排名、完整四策略结果、选中配置差值、最终六基因证据、训练回代AUC和逐供体预测。",
                  "- 输入与划分：原单外部划分、模型参数、逐折训练/测试单元和基因名单。",
                  "- 核对：数值验收、来源校验值、样本一致性、旧版结果复核、运行环境和警告。",
                  "- 脚本：本次运行、汇总、绘图和核查的可复现实现。", ""])
    (OUT / "结果说明与模型排名.md").write_text("\n".join(lines), encoding="utf-8")
    print("Saved Chinese results summary, main ranking and regional rankings.")


if __name__ == "__main__":
    main()
