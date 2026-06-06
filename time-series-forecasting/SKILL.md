---
name: time-series-forecasting
description: Senior-practitioner guide to time-series and classical/tabular ML — forecasting, classification,
  anomaly detection, and imputation. Use for any temporal-data task (sales/demand/traffic/metrics/sensor
  forecasting, intermittent/sparse series, hierarchical/grouped series, exogenous regressors), for choosing
  among classical statistical methods (ARIMA/SARIMA, ETS, Theta, Prophet), gradient-boosted trees on
  engineered lag/calendar/Fourier features (XGBoost/LightGBM — the comp workhorse), deep learning
  (DeepAR, N-BEATS/N-HiTS, TFT, DLinear, PatchTST), and time-series foundation models (TimesFM, Chronos,
  Moirai, TimeGPT — zero-shot). Also the non-deep tabular ML toolkit: GBDT vs random forest vs linear,
  calibration, imbalanced data, and anomaly detection (Isolation Forest, STL, forecast-residual). Critically
  covers correct evaluation: rolling-origin/time-based backtesting (NEVER random K-fold), MAE/RMSE/MAPE/
  sMAPE/MASE, pinball loss, prediction intervals, naive/seasonal-naive baselines, and leakage prevention.
---

# Time-Series Forecasting & Classical ML

Apply the judgment of an engineer who has shipped forecasting and tabular-ML systems in production for
years and competed where simple methods win. The two non-negotiables: **always backtest on the time axis
(never random CV), and always beat a naive baseline.** Most "forecasting failures" are leakage or a missing
baseline, not a weak model.

## How to use this skill

1. **Read `time-series-forecasting-guide.md`** in this directory — the full reference (task taxonomy,
   the methods spectrum from ARIMA to foundation models, evaluation/backtesting, production, anti-patterns).
   Apply it to the task at hand.
2. For concrete patterns to imitate — a rolling-origin backtest, a LightGBM lag-feature forecaster, and
   anomaly detection via forecast residuals — read **`examples.md`**.
3. Match the surrounding codebase/stack conventions (sktime/statsmodels/Darts/Nixtla/scikit-learn);
   apply the correctness rules — time-ordered splits, no leaked future features, calibrated intervals —
   regardless of stack.

## The essentials (full rationale in `time-series-forecasting-guide.md`)

- **Baseline first, always.** Naive (last value) and seasonal-naive (value one season ago) are the bar.
  If your model can't beat seasonal-naive on a proper backtest, you don't have a model. **MASE** scales
  error by the in-sample seasonal-naive error — < 1 means you beat it.
- **Backtest on the time axis.** Rolling-origin / expanding- or sliding-window evaluation, multiple
  origins. **Never random K-fold** on a series — it trains on the future to predict the past (leakage).
- **No leaked features.** A feature at time `t` may use only data available at `t`. Lags/rollings must be
  shifted; target-derived features must respect the forecast cutoff; calendar/holiday features are safe.
- **Match method to data, not to hype.** Few series + clear seasonality → **ETS/ARIMA/Theta**. Many related
  series + rich covariates → **GBDT on lag features (LightGBM/XGBoost)** or **global deep nets (DeepAR/
  N-HiTS/TFT/PatchTST)**. Cold-start / no history → **foundation models (TimesFM/Chronos/Moirai/TimeGPT,
  verify current)** zero-shot. The **M4/M5 competitions** showed simple methods + GBDT are brutally hard
  to beat.
- **GBDT is the tabular workhorse.** For structured/forecasting features it usually beats deep nets and
  always beats linear on nonlinear interactions. Reach for deep learning when you have many long series,
  shared structure, rich covariates, or need probabilistic multivariate output.
- **Pick the metric for the decision.** MAE (median) vs RMSE (penalizes large errors) vs **sMAPE/MASE**
  for cross-series aggregation. **Never MAPE near zero** (blows up). For probabilistic forecasts use
  **pinball/quantile loss** and check **calibration** (do 90% intervals cover ~90%?).
- **Intermittent demand needs its own tools** (Croston, SBA, ADIDA, or quantile GBDT) — squared-error
  models collapse the forecast toward zero.
- **Hierarchical/grouped series should reconcile** (bottom-up / MinT) so levels sum coherently.
- **Anomaly detection** is often "forecast + flag the residual": fit a model, score `y − ŷ` against a
  robust threshold. Also know **STL decomposition**, **Isolation Forest**, and autoencoders; see
  `[[ml-observability-monitoring]]` which consumes these for drift.
- **Production:** prevent training-serving skew with point-in-time feature pipelines (`[[data-engineering-feature-stores]]`),
  set a retraining cadence to track concept drift (`[[ml-observability-monitoring]]`), and plan for cold start
  and scale (thousands–millions of series → global models).

The ecosystem moves fast (it is 2026) — **verify foundation-model names, versions, and benchmark claims
against current docs** before relying on them.

## Related skills
- `[[ml-system-design]]` — frame the problem→metric→data→model→serving→monitoring loop before modeling.
- `[[data-engineering-feature-stores]]` — point-in-time joins and feature pipelines that prevent leakage/skew.
- `[[ml-observability-monitoring]]` — concept/data drift, retrain triggers; consumes forecast-residual anomalies.
- `[[ml-evaluation-evals]]` — metrics discipline, A/B testing, eval-in-CI for the broader ML picture.
- `[[recsys-ranking]]` — when "forecasting" is really ranking/demand at the item level with strong covariates.
- `[[graph-ml-gnns]]` — relational/spatio-temporal series (traffic, sensor networks) via GNN4TS.
