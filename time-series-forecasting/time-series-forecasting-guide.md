# Time-Series Forecasting & Classical ML — Full Reference

The single source of truth for this skill. Covers the task taxonomy, the full methods spectrum, correct
evaluation/backtesting, production concerns, and the anti-patterns that quietly destroy forecasting systems.
Two ideas run through everything: **respect the arrow of time (no leakage, time-ordered evaluation)** and
**always have a naive baseline you must beat.**

Grounding references (verify current): Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*
(3rd ed., free at <https://otexts.com/fpp3/>) is the canonical text; the **M-competitions** (M4, M5)
are the canonical empirical evidence about what actually works.

---

## 1. Mental model: what makes time series different

A time series is data indexed by time, where **observations are ordered and dependent** — the past
informs the future and the future must never inform the past. That single constraint invalidates most
default ML habits: you cannot shuffle, you cannot random-split, and you must reason about *what was known
when*. Everything below follows from it.

A series decomposes (conceptually) into **trend + seasonality + cycles + holiday/event effects + noise**.
Decomposition is additive (`y = T + S + e`) or multiplicative (`y = T · S · e`); use multiplicative (or
model `log y`) when seasonal amplitude grows with the level. **STL** (Seasonal-Trend decomposition using
Loess) is the robust workhorse for extracting these components and is the backbone of many anomaly and
deseasonalization methods.

Key properties to diagnose before modeling:
- **Stationarity** — constant mean/variance/autocovariance over time. ARIMA assumes it (after
  differencing); test with ADF / KPSS, but trust plots over p-values.
- **Seasonality** — fixed period(s): daily, weekly, yearly; often *multiple* periods at once (hourly
  data has daily + weekly + yearly seasonality). Inspect with ACF/PACF and seasonal sub-series plots.
- **Autocorrelation** — ACF (overall) and PACF (direct) guide ARIMA orders and feature lags.

## 2. The task taxonomy

**Forecasting** — predict future values. Sub-dimensions that change everything:
- **Horizon:** one-step vs **multi-step**. Multi-step strategies: *recursive* (feed predictions back in —
  compounds error, but cheap), *direct* (a separate model per horizon — no compounding, more models), and
  *multi-output/seq2seq* (deep nets predict the whole horizon jointly). Direct/multi-output are usually
  more robust at long horizons.
- **Univariate vs multivariate:** one target vs several interacting targets (VAR, multivariate deep nets).
- **Local vs global:** a separate model per series (*local*) vs **one model trained across many series**
  (*global* — GBDT and modern deep nets). Global models share statistical strength, generalize to new
  series, and are how you scale to millions of series; they won M5.
- **Exogenous regressors (covariates):** known-future (calendar, promotions, price) vs past-only
  (weather realized). Be precise about which covariates are available at forecast time.
- **Hierarchical / grouped:** series that aggregate (SKU → store → region → total). Forecasts should
  **reconcile** so levels are coherent.
- **Intermittent / sparse:** demand with many zeros (spare parts, long-tail SKUs) — needs Croston-family
  or quantile methods, not squared-error regression.

**Classification** — assign a label to a whole series or window (ECG arrhythmia, machine state, gesture).
Methods: distance-based (1-NN with **DTW**), shapelets, dictionary (BOSS), ensembles (HIVE-COTE,
**ROCKET/MiniRocket** — random convolutional kernels + linear classifier, extremely strong and fast), and
deep nets (InceptionTime). For tabular-summarized windows, GBDT on extracted features is a strong baseline.

**Anomaly detection** — flag points/subsequences/whole series that deviate. Covered in §7.

**Imputation** — fill missing values (sensor dropouts, irregular sampling). Methods from simple
(forward-fill, linear/spline interpolation, seasonal mean) to model-based (Kalman smoothing / state-space,
seasonal decomposition + interpolate, matrix factorization, and deep imputers like SAITS / BRITS — verify
current). **Rule: impute using only causally-available data when the downstream task is forecasting**, or
you leak the future. Whole-series interpolation is fine for offline analysis, not for live features.

## 3. The methods spectrum (and when each wins)

| Family | Examples | Best when | Watch out for |
|---|---|---|---|
| Naive / benchmark | naive, seasonal-naive, drift, mean | always — the bar to beat | — |
| Exponential smoothing | SES, Holt, **Holt-Winters / ETS** | clear trend+seasonality, few series | needs enough seasonal cycles |
| ARIMA family | ARIMA, **SARIMA**, SARIMAX (w/ exog) | stationary-after-differencing, autocorrelated | order selection, long-horizon decay |
| Decomposition/other classical | **Theta**, STL+ETS, TBATS | strong seasonality, M-comp winners | multiple/long seasonality (TBATS) |
| Structural / "auto" | **Prophet** | business series, holidays, analysts | not best accuracy; can over/underfit trend |
| GBDT on features | **LightGBM, XGBoost**, CatBoost | many series + covariates, tabular, comps | leakage in features; needs lag engineering |
| Global deep nets | **DeepAR, N-BEATS/N-HiTS, TFT, DLinear, PatchTST** | many long series, rich covariates, probabilistic | data-hungry, tuning, infra cost |
| Foundation models | **TimesFM, Chronos, Moirai, TimeGPT** | cold start, zero/few-shot, fast prototyping | verify version/license; benchmark vs baseline |

### 3.1 Classical statistical

- **ETS / exponential smoothing** — weighted average favoring recent observations; the *ETS taxonomy*
  (Error/Trend/Seasonality, each None/Additive/Multiplicative + damped trend) is selected automatically by
  AIC in good implementations. Damped-trend ETS is a superb default and a perennial M-competition top
  performer. Fast, robust, gives prediction intervals.
- **ARIMA(p,d,q)(P,D,Q)ₘ** — AutoRegressive Integrated Moving Average. `d`/`D` = differencing to reach
  stationarity, `(p,q)` = non-seasonal AR/MA orders, `(P,D,Q)ₘ` = seasonal with period `m`. **SARIMAX**
  adds exogenous regressors. Use `auto_arima` (pmdarima) or statsmodels; let it search orders by AICc but
  sanity-check residuals (should be white noise — Ljung-Box test).
- **Theta** — equivalent to SES with drift in its classic form; deceptively strong, won M3 and remains a
  top simple baseline. Always include it.
- **Prophet** (Meta, open source) — additive model: piecewise-linear/logistic trend + Fourier seasonality
  + holiday regressors. Easy, interpretable, handles missing data and holidays well; **not** the most
  accurate and can produce odd trend changepoints — treat as a strong baseline, not a default winner.
- **State-space / Kalman / BSTS** — flexible structural models; great for interpretable components and
  online updating. The Kalman filter also underpins robust imputation and online anomaly scoring.

### 3.2 ML on engineered features → gradient-boosted trees

This is the **workhorse for tabular forecasting and most Kaggle/M5-style problems**. You turn a series into
a supervised regression table, then fit **LightGBM / XGBoost / CatBoost**.

Feature families (build them *causally* — see §6 leakage):
- **Lags:** `y[t-1], y[t-7], y[t-28], …` — the single most predictive features. Pick lags from ACF and
  known seasonality.
- **Rolling/expanding stats:** mean/std/min/max/quantiles over trailing windows — **shifted** so window
  ends at `t-1` (or `t-h` for horizon `h`), never including `t`.
- **Calendar:** day-of-week, day-of-month, week-of-year, month, is-weekend, is-holiday, days-to-holiday.
- **Fourier terms:** `sin/cos(2π k t / period)` for smooth seasonality (multiple periods, no one-hot blowup).
- **Exogenous:** price, promo flags, weather — only as known/available at forecast time.
- **Static / group features:** store id, category, region (let a *global* model learn per-group behavior).

Why GBDT wins on tabular: it captures nonlinear interactions, handles mixed-type features and missing
values natively, is robust to monotone transforms, and trains fast with little tuning. For multi-step,
prefer **direct** (one model per horizon) or train with a horizon feature; recursive GBDT compounds error.
Objective choice matters: `tweedie`/`poisson` for count/intermittent demand, `quantile`/`pinball` for
prediction intervals. **GBDT cannot extrapolate trend** beyond the training range — detrend/difference the
target first (model `y[t] − y[t-period]` or first differences), then add the trend back.

### 3.3 Deep learning

Use when you have **many, long series**, shared structure, rich covariates, or need joint probabilistic /
multivariate output. Train *global* models. Notable architectures (cite the originating papers; verify IDs):
- **DeepAR** (Salinas et al., 2017–2020) — autoregressive RNN producing probabilistic forecasts via a
  parametric likelihood; the classic global probabilistic baseline.
- **N-BEATS** (Oreshkin et al., 2019) and **N-HiTS** (Challu et al., 2022) — pure MLP stacks with
  basis/multi-rate hierarchical interpolation; strong, fast, no recurrence.
- **TFT — Temporal Fusion Transformer** (Lim et al., 2019) — attention + variable selection + static
  covariate encoders + quantile outputs; interpretable and strong with mixed covariates.
- **DLinear / NLinear** (Zeng et al., 2022, *"Are Transformers Effective for Time Series Forecasting?"*) —
  a single linear layer on a decomposed series that **beat many transformers** on long-horizon benchmarks;
  a humbling, mandatory baseline before reaching for transformers.
- **PatchTST** (Nie et al., 2022) — patches the series into tokens (channel-independent) + a transformer;
  a strong modern long-horizon model. **iTransformer** (2023) inverts the attention to the variate
  dimension — verify current SOTA, the leaderboard churns.

Caveat (DLinear's lesson): **transformers are not automatically better** for forecasting. Always benchmark
against linear and GBDT.

### 3.4 Time-series foundation models (zero-/few-shot) — verify current

Pretrained on huge corpora of series, these forecast **out of the box without per-series training** —
ideal for cold start, prototyping, and "thousands of new series" cases. Verify names, versions, licenses,
and benchmark claims against current docs before depending on them:
- **TimesFM** (Google) — decoder-only patched foundation model; open weights.
- **Chronos** (Amazon) — tokenizes values and uses a T5-style LM; `chronos` / `chronos-bolt` variants.
- **Moirai** (Salesforce) — masked-encoder universal forecaster, multivariate, multiple frequencies.
- **TimeGPT** (Nixtla) — commercial API, zero-shot forecasting + anomaly detection.
- **Lag-Llama**, **Timer**, **TTM (TinyTimeMixer)** — other entrants; landscape is fast-moving.

Reality check: foundation models are strong zero-shot and excellent baselines, but a well-tuned GBDT or
ETS on your own data with good covariates often still wins on a specific problem. Use them to bootstrap and
to handle the long tail; benchmark against classical baselines before adopting.

## 4. Classical / tabular ML breadth (the not-deep toolkit)

A top engineer must wield the non-deep stack fluently — most production tabular problems live here.

**Model selection on tabular data:**
- **Linear / regularized (Ridge, Lasso, ElasticNet, logistic)** — fast, interpretable, great baselines and
  for genuinely linear signal or extreme high-dimensional sparse data. Lasso for feature selection.
- **Random Forest** — low-variance bagged trees; robust, little tuning, good baseline; usually loses to
  boosting on accuracy but is a great sanity check and gives easy uncertainty via tree spread.
- **Gradient-boosted trees (LightGBM/XGBoost/CatBoost)** — the default winner on heterogeneous tabular
  data. LightGBM = fast leaf-wise growth (watch overfitting via `num_leaves`/`min_data_in_leaf`); XGBoost =
  level-wise, very robust; CatBoost = best native categorical handling + ordered boosting (less target
  leakage). Tune learning rate × n_estimators jointly with early stopping; regularize with depth, subsample,
  colsample, L1/L2.

**When classical beats deep:** small/medium tabular data, heterogeneous features, strong tabular baselines,
limited compute, need for interpretability, or when feature engineering encodes the signal. Empirically,
GBDT still matches or beats deep tabular nets on most benchmarks (e.g., Grinsztajn et al., 2022, *"Why do
tree-based models still outperform deep learning on tabular data?"*).

**Feature engineering:** target/leave-one-out encoding for high-cardinality categoricals (do it *inside*
CV folds to avoid leakage; CatBoost does ordered encoding for you), interactions, binning, log/Box-Cox for
skew, and domain features. For trees, scaling and one-hot are usually unnecessary.

**Calibration:** classifier probabilities are often miscalibrated (esp. boosted trees and SVMs). Fix with
**Platt scaling** (sigmoid) or **isotonic regression** on a held-out set; measure with reliability diagrams
and **Brier score / ECE**. Critical when probabilities feed thresholds or expected-value decisions.

**Imbalanced data:** prefer fixing the *evaluation and threshold* before resampling. Use PR-AUC / F-beta
not raw accuracy; use class weights (`scale_pos_weight` in XGBoost), threshold tuning, and focal/weighted
loss. Resampling (SMOTE) helps sometimes but can hurt calibration — validate. Never oversample before
splitting (leakage).

## 5. Evaluation done right — the part people get wrong

### 5.1 Splitting / backtesting
- **NEVER random K-fold on a series.** It trains on future points to predict past ones → leakage →
  optimistic-then-disastrous. This is the single most common fatal mistake.
- **Train/validation/test must be time-ordered**, with the test block strictly after the train block, and
  a **gap** of at least the forecast horizon between them (so you can't peek across the cutoff).
- **Rolling-origin evaluation** (a.k.a. time-series cross-validation / walk-forward): pick multiple
  cutoffs; at each, train on data up to the cutoff and forecast the next `h` steps; average the metric
  across origins. *Expanding window* (train grows) vs *sliding window* (fixed train length — use when
  older data is stale or for cost). Use `sklearn.model_selection.TimeSeriesSplit` or library backtesters
  (sktime, Darts, Nixtla's `cross_validation`). See `examples.md` for a hand-rolled version.
- For grouped/panel data, split by **time across all groups at once**, not by group.
- Hyperparameter tuning must happen *inside* the time-ordered scheme (nested), never on the test block.

### 5.2 Metrics — pick for the decision
| Metric | What it does | Use / caution |
|---|---|---|
| MAE | mean abs error; optimizes the **median** | robust, interpretable in units |
| RMSE | penalizes large errors; optimizes the **mean** | when big misses are costly |
| MAPE | mean abs % error | **blows up / undefined near zero**; asymmetric (penalizes over-forecast less) |
| sMAPE | symmetric percentage | bounded, used in M-comps; still unstable near zero |
| **MASE** | error ÷ in-sample seasonal-naive MAE | **scale-free, comparable across series; < 1 beats naive** |
| WAPE / weighted | total abs error ÷ total actual | good for demand/$ aggregation |
| **Pinball / quantile loss** | asymmetric per-quantile | the right loss for **probabilistic** forecasts |
| CRPS | proper score for full distribution | comparing probabilistic forecasts |
| Coverage / interval score | do P% intervals cover ~P%? | **calibration** of intervals |

Rules: never report a single metric blindly; **MASE/sMAPE** for cross-series aggregation; **MAPE only when
values are safely away from zero and positive**; for probabilistic output report **pinball loss + coverage**.

### 5.3 Baselines
Always compute **naive** (ŷ = last value), **seasonal-naive** (ŷ = value one season ago), drift, and
historical mean. Report your model *relative* to them (e.g., MASE, or % improvement over seasonal-naive).
A model that doesn't beat seasonal-naive on a proper backtest is not shippable.

### 5.4 Prediction intervals & calibration
Point forecasts are rarely enough — decisions (inventory, capacity, alerting) need uncertainty. Get
intervals from: model likelihood (ETS/ARIMA/DeepAR), **quantile regression** (GBDT with pinball objective
at e.g. 0.1/0.5/0.9), or **conformal prediction** (distribution-free coverage guarantees; *conformalized
quantile regression* and time-series conformal variants). Then **check calibration** empirically on the
backtest — a "90% interval" that covers 70% is worse than useless. See `[[ml-evaluation-evals]]`.

## 6. Leakage — the cardinal sin (and how to avoid it)

Leakage = the model sees information at training/feature time that won't be available at prediction time.
In time series it is everywhere and it inflates offline metrics while collapsing in production.

- **Future features:** any feature using `y[t]` or later when predicting `t` (unshifted rolling stats,
  global standardization computed over the whole series, target encoding over all rows). Fix: shift all
  rolling/lag features so they end at the forecast cutoff; fit scalers/encoders on train only.
- **Look-ahead in preprocessing:** imputation, scaling, feature selection, and resampling must be fit on
  train and *applied* to validation/test — never fit on the full dataset.
- **Covariate availability:** weather/price "actuals" are not known at forecast time; use forecasted or
  lagged covariates, or known-future ones (calendar/promo schedule).
- **Point-in-time correctness:** when joining external features, use the value **as of** the timestamp,
  not the latest value — this is exactly what feature stores' point-in-time joins solve
  (`[[data-engineering-feature-stores]]`).
- **Group leakage:** the same entity in both train and test across time can leak via static features —
  usually fine if the split is purely temporal, but watch target-derived static aggregates.

## 7. Anomaly detection

Often framed as "forecast the normal, flag the residual," but several families apply:
- **Statistical / distributional:** z-score / robust z (median + MAD), seasonal hybrid ESD, control charts
  (EWMA, CUSUM), quantile thresholds. Cheap, interpretable, great for univariate metrics.
- **Decomposition-based:** **STL** → take the remainder component → threshold robustly (Twitter's
  Seasonal-Hybrid ESD is the classic). Handles seasonality cleanly.
- **Forecast-residual based:** fit any forecaster (ETS/SARIMA/GBDT/deep), compute `r = y − ŷ`, and flag
  when `|r|` exceeds a robust threshold or falls outside the prediction interval. This unifies forecasting
  and detection and is what most metric-monitoring systems do (see `examples.md`).
- **Distance / density (multivariate, unsupervised):** **Isolation Forest** (isolates anomalies with
  random splits — fast, scalable, the default), LOF, One-Class SVM, kNN distance.
- **Reconstruction-based (deep):** autoencoders / VAEs / LSTM-AE — train on normal data, flag high
  reconstruction error; useful for high-dim multivariate sensor data (verify current SOTA).
- **Change-point detection:** PELT, Bayesian online change-point detection — for regime shifts vs spikes.

Evaluation caveats: anomalies are rare and labels noisy — use PR-AUC, and **point-adjusted** or
range-based metrics carefully (point-adjustment can massively inflate scores; be skeptical). Anomaly
detection is the upstream of drift/alerting in `[[ml-observability-monitoring]]`, which consumes these
residual and reconstruction signals.

## 8. Production

- **Feature pipelines & skew:** compute features identically offline and online; prefer a feature store
  with **point-in-time joins** to guarantee training-serving parity (`[[data-engineering-feature-stores]]`).
  The #1 production forecasting bug is offline features that can't be reproduced at serving time.
- **Retraining cadence & concept drift:** series drift (new trends, regime change, promotions, post-event
  "new normal"). Decide retrain frequency by backtesting *how stale a model gets*; monitor error and input
  distribution and trigger retrains on drift (`[[ml-observability-monitoring]]`). Global models retrain
  less often than per-series local ones.
- **Cold start:** new series/SKUs/stores with no history → use a **global model** (borrows strength from
  peers via static features), hierarchical pooling, or a **foundation model** zero-shot until history
  accrues.
- **Scale (many series):** thousands–millions of series → favor **one global model** over per-series
  models (training/serving cost, maintenance). Parallelize per-series classical fits with Spark/Ray/Nixtla
  (`StatsForecast` is built for this). Cache features; precompute forecasts in batch where latency allows.
- **Relational / spatio-temporal series:** when series interact over a graph (road network traffic, sensor
  grids, supply chains), spatio-temporal **GNNs (GNN4TS** — e.g., STGCN, Graph WaveNet, MTGNN; verify
  current) model cross-series dependencies; see `[[graph-ml-gnns]]`.
- **Reconciliation in production:** for hierarchies, forecast all levels then reconcile (bottom-up or
  **MinT** — minimum-trace optimal reconciliation, Wickramasuriya et al.) so reported totals are coherent.

## 9. Anti-patterns (each has bitten real systems)

- **Random K-fold / shuffled split on a time series.** Leakage; offline metrics lie. Use rolling-origin.
- **Leaking future features** — unshifted rolling stats, full-series scaling/encoding, using realized
  covariates not known at forecast time.
- **No naive baseline.** Without seasonal-naive you can't tell if your fancy model is worthless.
- **MAPE on near-zero/zero values.** Explodes or is undefined; switch to MASE/WAPE/RMSE.
- **Point forecasts when the decision needs intervals** (inventory, capacity, alerting). Produce and
  calibrate quantiles.
- **Reaching for a transformer/LSTM where ARIMA or GBDT wins.** DLinear and M5 are the cautionary tales —
  benchmark the simple thing first.
- **GBDT extrapolating trend** — trees can't predict beyond the training target range; detrend/difference
  first.
- **Recursive multi-step without acknowledging compounding error** — prefer direct/multi-output for long h.
- **Ignoring intermittency** — squared-error models forecast spiky demand as ~zero; use Croston/quantile.
- **Tuning on the test block**, or selecting the model on a single lucky origin — use multiple origins.
- **Decomposing/imputing over the full series then forecasting** — that interpolation leaks the future.
- **Trusting a foundation model's leaderboard number on your data** — always backtest on your own series.

## 10. Version awareness

The classical/statistical core (ARIMA, ETS, Theta, STL, the M-competition findings, leakage discipline) is
stable and durable — invest in it. The **deep-learning and foundation-model frontier moves monthly**: model
names, weights, licenses, leaderboard rankings, and library APIs (Darts, Nixtla `neuralforecast`/
`statsforecast`/`mlforecast`, sktime, GluonTS, PyTorch-Forecasting, tsfresh, sktime/aeon for classification,
PyOD for anomaly detection) change frequently. **Verify any specific model, version, benchmark number, or
arXiv ID against current documentation before relying on it.** Where this guide names a paper, treat the
attribution as a pointer, not a citation to quote verbatim — confirm authors/year/ID.

## 11. Canonical references (real, verify current)

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.) — <https://otexts.com/fpp3/>.
  The textbook; ETS, ARIMA, hierarchical reconciliation, evaluation, the `fable`/`tsibble` R ecosystem.
- **M4 competition** — Makridakis, Spiliotis, Assimakopoulos (2018/2020), *Int. J. Forecasting*. Showed
  hybrid/statistical + simple combinations are extremely strong (the ES-RNN winner).
- **M5 competition** (Kaggle, 2020; results papers 2022) — Walmart hierarchical retail demand; **LightGBM
  and simple methods dominated**; intermittency and hierarchy were central. Canonical evidence for GBDT.
- Zeng et al., *Are Transformers Effective for Time Series Forecasting?* (AAAI 2023) — the DLinear paper.
- Lim et al., *Temporal Fusion Transformers* (2019); Oreshkin et al., *N-BEATS* (ICLR 2020); Challu et al.,
  *N-HiTS* (2022); Nie et al., *PatchTST* (2022); Salinas et al., *DeepAR* (2020). (Verify IDs/years.)
- Grinsztajn et al., *Why do tree-based models still outperform deep learning on tabular data?* (NeurIPS
  2022) — the tabular-GBDT evidence.
- Wickramasuriya, Athanasopoulos, Hyndman, *MinT* optimal reconciliation (JASA 2019).
- Libraries: Nixtla (`statsforecast`/`mlforecast`/`neuralforecast`), Darts, sktime/aeon, GluonTS, PyOD,
  statsmodels, pmdarima, Prophet, LightGBM/XGBoost/CatBoost. Verify current APIs.
