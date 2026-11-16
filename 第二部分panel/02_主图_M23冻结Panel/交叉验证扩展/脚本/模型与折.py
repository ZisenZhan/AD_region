"""Pure frozen M23 interface and fold-local preprocessing cache."""
from pathlib import Path
import ast
import sys
import types
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[2]
SOURCE = BASE / "论文初稿_0926_Meta八队列_Panel冻结版/多模型比较_当前六机制底座/计算与核对/49_build_multimodel_auc_heatmap.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
names = {"PLSLogisticClassifier", "scaled_pipeline", "predict_score"}
body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
assert len(body) == len(names)
mm = types.ModuleType("frozen_m23_cv_expansion")
mm.__dict__.update(dict(np=np, pd=pd, BaseEstimator=BaseEstimator, ClassifierMixin=ClassifierMixin,
    PLSRegression=PLSRegression, LogisticRegression=LogisticRegression, SimpleImputer=SimpleImputer,
    Pipeline=Pipeline, StandardScaler=StandardScaler, SEED=20260807))
sys.modules[mm.__name__] = mm
exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec",
             flags=__import__("__future__").annotations.compiler_flag), mm.__dict__)


def full_fit(x, info):
    model = mm.scaled_pipeline(mm.PLSLogisticClassifier(n_components=2))
    weights = info.weight.to_numpy()
    args = {"scale__sample_weight": weights}
    if np.any(weights != 1):
        ix = np.repeat(np.arange(len(x)), np.rint(weights * 2).astype(int))
        x, info = x.iloc[ix], info.iloc[ix]
        args["scale__sample_weight"] = np.ones(len(ix))
    model.fit(x, info.y, **args)
    return model


def donor_spec(meta):
    assert meta.groupby("key").y.nunique().eq(1).all()
    keys = np.array(sorted(meta.key.unique()))
    ix = pd.Index(keys).get_indexer(meta.key)
    weights = 1 / (meta.groupby(["key", "sub"]).key.transform("size") * meta.groupby("key")["sub"].transform("nunique"))
    y = meta.groupby("key").y.first().loc[keys].to_numpy(dtype=int)
    assert set(y) == {0, 1}
    return keys, y, ix, weights.to_numpy()


class Fold:
    def __init__(self, fold_id, train, info, queries):
        self.fold_id = fold_id
        self.columns = sorted(train.columns)
        self.train, self.info = train[self.columns], info
        assert self.train.index.equals(info.index)
        w = info.weight.to_numpy()
        ix = np.repeat(np.arange(len(w)), np.rint(w * 2).astype(int)) if np.any(w != 1) else np.arange(len(w))
        self.y = info.y.to_numpy(dtype=int)[ix]
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        z = self.imputer.fit_transform(self.train.iloc[ix])
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(z, sample_weight=np.ones(len(ix)))
        self.queries = {}
        for geo, (xx, md) in queries.items():
            keys, y, pos, weights = donor_spec(md)
            self.queries[geo] = dict(raw=xx[self.columns], meta=md, keys=keys, y=y, ix=pos, weights=weights,
                x=self.scaler.transform(self.imputer.transform(xx[self.columns])))

    def evaluate(self, genes):
        idx = pd.Index(self.columns).get_indexer(sorted(genes))
        assert (idx >= 0).all()
        fitted = mm.PLSLogisticClassifier(n_components=2).fit(self.x[:, idx], self.y)
        result = {}
        for geo, q in self.queries.items():
            score = np.bincount(q["ix"], weights=fitted.predict_proba(q["x"][:, idx])[:, 1] * q["weights"], minlength=len(q["keys"]))
            y, n = q["y"], int(q["y"].sum())
            auc = (rankdata(score)[y == 1].sum() - n * (n + 1) / 2) / (n * (len(y) - n))
            result[geo] = dict(auc=float(auc), score=score)
        return result

    def reference(self, genes):
        model = full_fit(self.train[sorted(genes)], self.info)
        result = {}
        for geo, q in self.queries.items():
            p = mm.predict_score(model, q["raw"][sorted(genes)])
            result[geo] = np.bincount(q["ix"], weights=p * q["weights"], minlength=len(q["keys"]))
        return model, result
