# Examples — Time-Series Forecasting & Classical ML

Canonical, runnable-in-spirit patterns to imitate. They encode the rules from the guide: time-ordered
evaluation, causal (non-leaking) features, a baseline you must beat, and residual-based anomaly detection.
APIs (pandas/sklearn/LightGBM/statsmodels) are stable enough to copy; **verify any library version
specifics against current docs.**

---

## 1. Rolling-origin (walk-forward) backtest with a beaten baseline

The single most important habit: evaluate on the time axis across multiple origins, and report your model
*relative to seasonal-naive*. This hand-rolled version makes the mechanics explicit; in practice prefer a
library backtester (`sklearn.TimeSeriesSplit`, sktime, Darts, Nixtla `cross_validation`).

```python
import numpy as np
import pandas as pd

def mase(y_true, y_pred, y_train, season=1):
    """Mean Absolute Scaled Error: < 1 means you beat the in-sample seasonal-naive."""
    naive_err = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    return np.mean(np.abs(y_true - y_pred)) / (naive_err + 1e-12)

def rolling_origin_backtest(series, fit_predict, horizon=7, season=7,
                            n_origins=8, min_train=180, step=7):
    """Expanding-window walk-forward backtest.
    `fit_predict(train: pd.Series, horizon: int) -> np.ndarray` returns the h-step forecast.
    A GAP is implicit: we only ever forecast points strictly after the training cutoff.
    """
    n = len(series)
    # Origins are the last `n_origins` cutoffs, each `step` apart, each leaving `horizon` to score.
    origins = range(min_train, n - horizon + 1, step)
    origins = list(origins)[-n_origins:]

    rows = []
    for cutoff in origins:
        train = series.iloc[:cutoff]                      # only the past
        actual = series.iloc[cutoff:cutoff + horizon].to_numpy()

        yhat_model = fit_predict(train, horizon)          # your model
        # seasonal-naive baseline: repeat the value one season ago
        last_season = train.iloc[-season:].to_numpy()
        yhat_snaive = np.resize(last_season, horizon)

        rows.append({
            "cutoff": series.index[cutoff],
            "mase_model":  mase(actual, yhat_model,  train.to_numpy(), season),
            "mase_snaive": mase(actual, yhat_snaive, train.to_numpy(), season),
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print(f"\nmean MASE  model={res.mase_model.mean():.3f}  "
          f"seasonal-naive={res.mase_snaive.mean():.3f}")
    assert res.mase_model.mean() < res.mase_snaive.mean(), \
        "Model does not beat seasonal-naive — not shippable."
    return res
```

Notes:
- **Multiple origins**, not one lucky split. Average the metric across them.
- **Expanding window** here (`series.iloc[:cutoff]`); switch to `series.iloc[cutoff-W:cutoff]` for a
  **sliding window** when old data is stale.
- The gap is enforced structurally: we never use any point ≥ `cutoff` to fit. If features need a warm-up
  of `L` lags, ensure the forecast for `cutoff` only reads up to `cutoff-1`.
- Tune hyperparameters by running this loop on a **validation** span, then evaluate once on a later
  held-out span. Never select on the final test block.

---

## 2. LightGBM forecaster with causal lag / calendar / Fourier features

The tabular workhorse for many series + covariates (the M5-style pattern). The discipline that matters is
**no leakage**: every feature at row `t` uses only data ≤ the forecast cutoff, and trend is differenced
out because trees can't extrapolate.

```python
import numpy as np
import pandas as pd
import lightgbm as lgb

SEASON = 7  # weekly

def make_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """df: columns [date, series_id, y] sorted by (series_id, date).
    Build features available at forecast time for a `horizon`-step-ahead model (DIRECT strategy).
    Every lag/rolling is shifted by `horizon` so it ends at the cutoff, never peeking at t..t+h-1.
    """
    g = df.groupby("series_id", group_keys=False)

    # --- target: difference out trend so the tree predicts a stationary delta ---
    df["y_season_ago"] = g["y"].shift(SEASON)
    df["target"] = df["y"] - df["y_season_ago"]           # predict deviation from seasonal-naive

    # --- lag features (shifted by horizon: known at the cutoff for an h-step forecast) ---
    for lag in [horizon, horizon + 1, horizon + 6, horizon + 7, horizon + 27]:
        df[f"lag_{lag}"] = g["y"].shift(lag)

    # --- rolling stats, SHIFTED so the window ends before the cutoff (no current value) ---
    for w in [7, 28]:
        s = g["y"].shift(horizon)
        df[f"rmean_{w}"] = s.rolling(w).mean().reset_index(level=0, drop=True)
        df[f"rstd_{w}"]  = s.rolling(w).std().reset_index(level=0, drop=True)

    # --- calendar features (known-future: always safe) ---
    dt = df["date"].dt
    df["dow"], df["dom"], df["month"], df["woy"] = (
        dt.dayofweek, dt.day, dt.month, dt.isocalendar().week.astype(int))
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # --- Fourier terms for smooth yearly seasonality (no one-hot blowup) ---
    doy = dt.dayofyear
    for k in (1, 2, 3):
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return df

FEATURES = ([c for c in []]  # filled at call time; below we select by prefix/name)

def train_direct(df_feat, cutoff_date, horizon):
    feats = [c for c in df_feat.columns
             if c.startswith(("lag_", "rmean_", "rstd_", "sin_", "cos_"))
             or c in ("dow", "dom", "month", "woy", "is_weekend", "series_id")]
    train = df_feat[(df_feat.date <= cutoff_date) & df_feat.target.notna()].dropna(subset=feats)

    model = lgb.LGBMRegressor(
        objective="l1",            # MAE -> optimizes the median; use "tweedie" for intermittent demand
        n_estimators=2000, learning_rate=0.03,
        num_leaves=63, min_child_samples=100,    # regularize LightGBM's leaf-wise growth
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    model.fit(train[feats], train["target"],
              categorical_feature=["series_id"])
    return model, feats

# Predict for the h-step horizon, then add the seasonal-naive level back (we modeled the delta):
#   yhat[t] = model.predict(features_at_t) + y[t - SEASON]
# For PREDICTION INTERVALS, fit additional LightGBM models with objective="quantile", alpha=0.1 / 0.9.
```

Key points encoded above:
- **DIRECT multi-step** (features shifted by `horizon`) — avoids recursive error compounding.
- **Difference out trend** (`target = y − y_season_ago`); GBDT can't extrapolate, so reconstruct the level
  after predicting. This also makes the target roughly stationary.
- **Causal features only:** lags/rollings shifted to end at the cutoff; calendar/Fourier are known-future
  and safe. `series_id` as a categorical lets one **global** model serve all series (and new ones).
- Swap the objective for the problem: `l1`/`l2`, `tweedie`/`poisson` for counts/intermittency,
  `quantile` for probabilistic intervals.
- Always run this through the §1 backtest and confirm it beats seasonal-naive.

---

## 3. Anomaly detection via forecast residual

Most metric/operational anomaly detection is "forecast the normal, flag the residual." Robust thresholds
(median + MAD) survive outliers far better than mean ± k·std. This pattern is the upstream signal for drift
and alerting in `[[ml-observability-monitoring]]`.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_residual_anomalies(series: pd.Series, season=7, z=3.5):
    """One-step-ahead, walk-forward residuals scored with a robust z (median + MAD).
    Fit on the past only at each step -> residuals are out-of-sample (no leakage).
    """
    resid = pd.Series(index=series.index, dtype=float)
    min_train = max(3 * season, 30)

    for t in range(min_train, len(series)):
        train = series.iloc[:t]                              # strictly the past
        model = ExponentialSmoothing(
            train, trend="add", damped_trend=True,
            seasonal="add", seasonal_periods=season,
        ).fit()
        yhat = model.forecast(1).iloc[0]
        resid.iloc[t] = series.iloc[t] - yhat                # out-of-sample residual

    # robust threshold via Median Absolute Deviation (0.6745 -> consistency with std for normal data)
    r = resid.dropna()
    med = r.median()
    mad = (np.abs(r - med)).median()
    robust_z = 0.6745 * (resid - med) / (mad + 1e-12)
    anomalies = robust_z.abs() > z
    return pd.DataFrame({"residual": resid, "robust_z": robust_z, "is_anomaly": anomalies})
```

Variations:
- **STL-remainder** detector: `STL(series, period=season).fit().resid`, then the same robust-z threshold —
  cleaner when seasonality is strong and you want the decomposition explicitly.
- **Interval-based:** flag when `y` falls outside the model's prediction interval (e.g., DeepAR/quantile
  GBDT 1%/99%) — calibrate the interval first (guide §5.4).
- **Multivariate / unsupervised** (no clean "normal" series): use **Isolation Forest** (`sklearn`) on a
  window of features, or an LSTM/conv **autoencoder** and threshold reconstruction error.
- **Evaluate** with PR-AUC against labeled events; be wary of point-adjusted metrics — they can inflate
  scores dramatically.
