# AGENTS.md — Time-Series & Classical ML Standards

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative reference lives in **`time-series-forecasting-guide.md`** next to this file —
> read it before any forecasting / anomaly / imputation / tabular-ML task and apply it. Concrete patterns
> to imitate (rolling-origin backtest, LightGBM lag-feature forecaster, anomaly-via-residual) are in
> **`examples.md`**. This file is the always-on summary.
>
> **Two non-negotiables:** (1) evaluate on the **time axis** — never random CV on a series; (2) always
> beat a **naive/seasonal-naive baseline** on a proper backtest, or you don't have a model.

## When working on temporal data or tabular ML, apply these by default:

- **Baseline first.** Compute naive + seasonal-naive (+ drift/mean). Report your model relative to them;
  **MASE < 1** means you beat seasonal-naive. No baseline = no result.
- **Time-ordered evaluation only.** Rolling-origin / walk-forward over multiple cutoffs; expanding or
  sliding window; a **gap ≥ horizon** between train and test. **Never** `KFold`/`shuffle` a series —
  it's leakage. Tune hyperparameters *inside* the time-ordered scheme, never on the test block.
- **No leaked features.** A feature at `t` uses only data available at `t`: shift lags/rollings to end at
  the cutoff; fit scalers/encoders/imputers on **train only**; use known-future or lagged covariates (not
  realized "actuals"); join external data **point-in-time** (`[[data-engineering-feature-stores]]`).
- **Match method to data, not hype.** Few series + seasonality → **ETS/ARIMA/Theta**. Many series +
  covariates → **GBDT on lag/calendar/Fourier features (LightGBM/XGBoost)** or global deep nets
  (DeepAR/N-HiTS/TFT/PatchTST). Cold start / no history → **foundation models (TimesFM/Chronos/Moirai/
  TimeGPT — verify current)** zero-shot. **M4/M5: simple methods + GBDT are very hard to beat.**
- **GBDT is the tabular workhorse**; it beats deep nets on most heterogeneous tabular data and can't
  extrapolate trend — **detrend/difference the target first**, then add trend back. Use `tweedie`/`poisson`
  for counts, `quantile`/pinball for intervals. Prefer **direct/multi-output** over recursive for long h.
- **Pick the metric for the decision.** MAE (median) vs RMSE (big-error penalty) vs **MASE/sMAPE** for
  cross-series. **Never MAPE near zero.** Probabilistic → **pinball loss + coverage/calibration check**.
- **Intermittent demand** (many zeros) → Croston/SBA/ADIDA or quantile GBDT, not squared-error regression.
- **Hierarchical/grouped series → reconcile** (bottom-up / MinT) so levels sum coherently.
- **Anomaly detection** is usually "forecast + flag residual": score `y − ŷ` against a robust threshold or
  prediction interval. Also: STL remainder, Isolation Forest (multivariate default), autoencoders. Feeds
  drift/alerting in `[[ml-observability-monitoring]]`. Evaluate with PR-AUC; be skeptical of point-adjusted scores.
- **Imputation** for forecasting must be causal (no full-series interpolation as live features).
- **Production:** identical offline/online features (no skew), retrain cadence set by how fast the model
  goes stale + drift monitoring (`[[ml-observability-monitoring]]`), global models for cold start and scale
  (millions of series), GNN4TS for relational/spatio-temporal series (`[[graph-ml-gnns]]`).
- **Classical breadth:** know linear/Ridge/Lasso, Random Forest, and GBDT (LightGBM/XGBoost/CatBoost)
  tradeoffs; **calibrate** probabilities (Platt/isotonic, check Brier/ECE); for **imbalance** fix
  metric+threshold (PR-AUC, class weights) before resampling, and never resample before splitting.

## Definition of done for a forecasting/ML change
- A naive + seasonal-naive baseline is computed and beaten on a **rolling-origin backtest** (multiple origins).
- The split is strictly time-ordered with a gap ≥ horizon; no feature/preprocessing leakage.
- The reported metric fits the decision (MASE/sMAPE/RMSE; pinball + coverage if probabilistic).
- If decisions need uncertainty, intervals are produced **and calibration-checked**.

## Version awareness
Classical core (ARIMA/ETS/Theta/STL, leakage discipline, M-competition findings) is stable. The deep-learning
and **foundation-model frontier moves monthly** — **verify model names, versions, licenses, benchmark
numbers, arXiv IDs, and library APIs against current docs** before relying on them.
