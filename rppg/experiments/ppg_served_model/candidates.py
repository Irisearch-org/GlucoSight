import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
INNER_SPLITS = 4
RIDGE_ALPHAS = np.logspace(-1, 4, 6)
RIDGE_FIXED_ALPHA = 10.0          # the value used by rppg/models/glucose_estimator.py
RF_LEAVES = (2, 5, 10)
RF_TREES = 200
PCA_COMPONENTS = 10


class MeanPredictor:
    """Constant = training-fold mean glucose (B-mean baseline)."""

    def __init__(self, y):
        self.value_ = float(np.mean(y))

    def predict(self, X):
        return np.full(len(X), self.value_)


def _ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _rf(leaf):
    return RandomForestRegressor(n_estimators=RF_TREES, min_samples_leaf=leaf,
                                 random_state=RANDOM_STATE, n_jobs=1)


def _pca_rf(leaf):
    return make_pipeline(StandardScaler(),
                         PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE),
                         _rf(leaf))


def _inner_best(build, grid, X, y, groups):
    """Pick the grid value with the lowest pooled inner-GroupKFold MAE."""
    k = min(INNER_SPLITS, len(np.unique(groups)))
    best, best_mae = None, np.inf
    for p in grid:
        errs = []
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            m = build(p).fit(X[tr], y[tr])
            errs.append(np.abs(m.predict(X[te]) - y[te]))
        mae = float(np.concatenate(errs).mean())
        if mae < best_mae:
            best, best_mae = p, mae
    return best


def fit_mean(X, y, groups):
    return MeanPredictor(y)


def fit_ridge_fixed(X, y, groups):
    return _ridge(RIDGE_FIXED_ALPHA).fit(X, y)


def fit_ridge_nested(X, y, groups):
    a = _inner_best(_ridge, RIDGE_ALPHAS, X, y, groups)
    m = _ridge(a).fit(X, y)
    m.tuned_ = {"alpha": float(a)}
    return m


def fit_rf_nested(X, y, groups):
    leaf = _inner_best(_rf, RF_LEAVES, X, y, groups)
    m = _rf(leaf).fit(X, y)
    m.tuned_ = {"min_samples_leaf": int(leaf)}
    return m


def fit_pca_rf_nested(X, y, groups):
    leaf = _inner_best(_pca_rf, RF_LEAVES, X, y, groups)
    m = _pca_rf(leaf).fit(X, y)
    m.tuned_ = {"min_samples_leaf": int(leaf), "pca_components": PCA_COMPONENTS}
    return m


# name -> (feature set key in the cache, fit function, description)
CANDIDATES = {
    "ridge_handcrafted_alpha10": ("hand", fit_ridge_fixed,
                                  "Ridge(alpha=10) on 7 handcrafted features (estimator config)"),
    "ridge_handcrafted_nested": ("hand", fit_ridge_nested,
                                 "Ridge on 7 handcrafted features, alpha nested"),
    "rf_handcrafted_nested": ("hand", fit_rf_nested,
                              "RandomForest on 7 handcrafted features, min_samples_leaf nested"),
    "papagei_ridge_nested": ("papagei_native", fit_ridge_nested,
                             "PaPaGei-S 512-d frozen embedding + Ridge, alpha nested"),
    "papagei_pca10_rf_nested": ("papagei_native", fit_pca_rf_nested,
                                "PaPaGei-S embedding -> PCA(10) -> RandomForest, leaf nested"),
}


def get_X(cache, feature_key):
    return np.asarray(cache[feature_key], dtype=float)
