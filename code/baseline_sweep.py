"""Pick a text baseline by validating on a held-out slice of the exploratory
split, then report it once on the confirmatory split."""

import pathlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from baseline_text import hand_features, load_ad, score_grouped  # noqa: E402


def prepare(df: pd.DataFrame) -> None:
    log_ctr = np.log(df["ctr"].clip(lower=1e-5))
    df["y"] = log_ctr - log_ctr.groupby(df["clickability_test_id"]).transform("mean")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    exp = load_ad("exploratory")
    prepare(exp)
    conf = load_ad("confirmatory")
    prepare(conf)

    tests = sorted(exp["clickability_test_id"].unique())
    holdout_tests = set(tests[::3])
    dev = exp[exp["clickability_test_id"].isin(holdout_tests)]
    fit = exp[~exp["clickability_test_id"].isin(holdout_tests)]
    print(f"fit rows {len(fit):,} dev rows {len(dev):,} conf rows {len(conf):,}")

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True)
    w_fit = word.fit_transform(fit["headline"])
    c_fit = char.fit_transform(fit["headline"])
    h_fit = hand_features(fit["headline"]).values
    x_fit = hstack([w_fit, c_fit, h_fit]).tocsr()

    def matrix(df):
        return hstack(
            [word.transform(df["headline"]), char.transform(df["headline"]),
             hand_features(df["headline"]).values]
        ).tocsr()

    x_dev = matrix(dev)
    x_conf = matrix(conf)

    configs = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        configs.append((f"ridge_tfidf_alpha{alpha}", Ridge(alpha=alpha)))
    for leaves in (7, 15, 31):
        for mcs in (200, 1000):
            configs.append(
                (
                    f"lgbm_leaves{leaves}_mcs{mcs}",
                    lgb.LGBMRegressor(
                        n_estimators=300, learning_rate=0.03, num_leaves=leaves,
                        min_child_samples=mcs, verbose=-1,
                    ),
                )
            )

    best = None
    for name, model in configs:
        model.fit(x_fit, fit["y"], sample_weight=None)
        pred = pd.Series(model.predict(x_dev), index=dev.index)
        c, n = score_grouped(pred, dev)
        acc = c / n
        print(f"  dev {name:28s} {acc:.4f}")
        if best is None or acc > best[1]:
            best = (name, acc, model)

    print(f"selected on dev: {best[0]} ({best[1]:.4f})")
    model = best[2]
    pred = pd.Series(model.predict(x_conf), index=conf.index)
    c, n = score_grouped(pred, conf)
    print(f"confirmatory pairwise accuracy of {best[0]}: {c / n:.4f} (pairs {n:,})")


if __name__ == "__main__":
    main()
