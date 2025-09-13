# Chess Analysis Metrics

## Overview

This document details the comprehensive set of metrics used by ChessPlayerAnalyzer to evaluate chess game quality, detect anomalies, and assess player performance. All metrics are implemented in the `app/analysis/` modules and combine traditional chess analysis with advanced statistical methods.

## Quality Metrics (`quality.py`)

### 1. Average Centipawn Loss (ACPL)

**Purpose**: Primary measure of move accuracy compared to engine evaluation.

**Mathematical Definition**:
```
ACPL = median(|eval_after - eval_before|) for each move
```

**Implementation Details**:
- Uses median by default (more robust to blunders than mean)
- Caps extreme values at 1500cp to prevent mate evaluations from distorting results
- Handles perspective correction for black moves automatically
- Fallback mechanism when `delta_eval` is not available

**Code Location**: `quality.py:44-100`

**Interpretation**:
- < 20cp: Exceptionally strong play (potentially suspicious)
- 20-50cp: Strong play (expert level)
- 50-100cp: Good play (intermediate)
- > 100cp: Weak play (beginner)

### 2. Win-Draw-Loss Probability

**Mathematical Definition**:
```
WDL_prob = 1 / (1 + 10^(-eval_cp / k))
```
Where `k = 400` (standard scaling factor)

**Code Location**: `quality.py:22-37`

### 3. Match Rate (Engine Agreement)

**Purpose**: Percentage of moves matching engine's top choice.

**Implementation**:
- Binary classification per move: `is_engine_best = True/False`
- Aggregated as percentage over all moves
- Weighted variants consider move importance

**Expected Values by Strength**:
- 2800+ Elo: 45-55%
- 2400-2800 Elo: 35-45%
- 2000-2400 Elo: 25-35%
- < 2000 Elo: 15-25%

## Timing Metrics (`timing.py`)

### 1. Time-Complexity Correlation

**Purpose**: Measures if thinking time correlates with position complexity.

**Mathematical Definition**:
```
correlation = spearman_rank(move_time, legal_moves_count)
```

**Code Location**: `timing.py:60-91`

**Interpretation**:
- Normal players: 0.3-0.7 correlation
- Suspicious (engines): < 0.1 correlation
- Random play: ~0.0 correlation

### 2. Lag Spike Detection

**Purpose**: Identifies pause-then-perfect-play patterns characteristic of engine use.

**Algorithm**:
1. Detect pauses: `move_time  [5.0, 12.0]` seconds
2. Check subsequent rapid moves: < 2.0 seconds
3. Verify high accuracy in rapid sequence: `is_engine_best = True`

**Code Location**: `timing.py:96-150`

**Parameters**:
- `pause_sec`: (5.0, 12.0) - suspicious pause range
- `rapid_thresh`: 2.0 - rapid move threshold
- `rapid_window`: 2 - moves to check after pause

### 3. Time Distribution Analysis

**Statistical Tests**:
- **Kolmogorov-Smirnov Test**: Tests if times follow log-normal distribution
- **Anderson-Darling Test**: More sensitive to tail deviations
- **Coefficient of Variation**: `std/mean` ratio

**Code Location**: `timing.py:36-55`

## Anomaly Detection (`anomaly.py`)

### 1. STL Decomposition

**Purpose**: Isolates trend, seasonal, and residual components of time series.

**Mathematical Framework**:
```
X(t) = Trend(t) + Seasonal(t) + Residual(t)
Z-score = (Residual - ¼) / Ã
```

**Code Location**: `anomaly.py:39-67`

**Parameters**:
- `period`: 10 (seasonal cycle length)
- `robust`: True (outlier-resistant fitting)

### 2. Isolation Forest

**Purpose**: Detects anomalous move patterns using ensemble isolation.

**Algorithm**:
- Creates random forest of isolation trees
- Measures path length to isolate each observation
- Shorter paths indicate anomalies

**Code Location**: `anomaly.py:70-99`

**Parameters**:
- `n_estimators`: 100 trees
- `contamination`: 0.1 (expected anomaly rate)
- `random_state`: 42 (reproducibility)

## Bayesian Analysis (`bayesian.py`)

### 1. Suspicion Prior Calculation

**Mathematical Framework**:
```
P(suspicious | rating, experience) = ± / (± + ² + extra_games)
```

**Beta-Binomial Model**:
- Historical suspicious game counts grouped by rating/experience buckets
- Rating buckets: 200-point intervals (1200-1399, 1400-1599, etc.)
- Experience buckets: 50-game intervals

**Code Location**: `bayesian.py:48-69`

### 2. Likelihood Ratios

**Evidence Integration**:
```
posterior_odds = prior_odds × (likelihood_ratios)
```

**Likelihood Rules**:
- `acpl < 20`: 5.0x more likely if suspicious
- `match_rate > 0.7`: 4.0x more likely if suspicious
- `time_complexity_corr < 0.1`: 3.0x more likely if suspicious
- `lag_spike_count > 2`: 2.0x more likely if suspicious

**Code Location**: `bayesian.py:74-85`

## Performance Modeling (`performance_model.py`)

### 1. GARCH(1,1) Model

**Purpose**: Models volatility clustering in performance metrics.

**Mathematical Definition**:
```
Ã²(t) = É + ± × µ²(t-1) + ² × Ã²(t-1)
```

**Parameter Estimation**:
- `± = 0.1` (ARCH coefficient)
- `² = 0.8` (GARCH coefficient)
- `É = variance × (1 - ± - ²)` (long-term variance)

**Code Location**: `performance_model.py:6-38`

### 2. Auto Model Selection

**Selection Criteria**:
```
volatility = std(series.diff())
if volatility > 1.0: use GARCH
elif volatility > 0.1: use ARIMA
else: use Kalman Filter
```

**Code Location**: `performance_model.py:61-68`

### 3. Kalman Filter

**State Space Model**:
```
x(t) = x(t-1) + w(t)    [state evolution]
z(t) = x(t) + v(t)      [observation]
```

**Update Equations**:
```
K(t) = P(t|t-1) / (P(t|t-1) + R)          [Kalman gain]
x(t|t) = x(t|t-1) + K(t) × (z(t) - x(t|t-1))  [state update]
P(t|t) = (1 - K(t)) × P(t|t-1)            [covariance update]
```

**Code Location**: `performance_model.py:80-90`

## Longitudinal Analysis (`longitudinal.py`)

### 1. ROI (Rating of Interest) Calculation

**Mathematical Definition**:
```
ROI = 800 × match_rate - 0.5 × acpl + 2000
```

**Code Location**: `longitudinal.py:45-50`

**Reference Standard Deviations by ELO**:
```python
ACPL_SD_BY_ELO = {1200: 95, 1600: 80, 2000: 65, 2400: 55, 2800: 45}
MATCH_SD_BY_ELO = {1200: 7.5, 1600: 7.0, 2000: 6.5, 2400: 5.5, 2800: 5.0}
```

### 2. Change Point Detection

**CUSUM Algorithm**:
```
Sz(t) = max(0, Sz(t-1) + (x(t) - ¼ - ´))
S{(t) = min(0, S{(t-1) + (x(t) - ¼ + ´))
```

**Bayesian Online Change Point Detection**:
- Adams & MacKay (2007) algorithm
- Constant hazard function with » = 250
- Student-t likelihood model

**Code Location**: `change_point.py:7-80`

## Clustering Analysis (`clustering.py`)

### 1. Feature Engineering

**Player Profile Vector**:
```
features = [mean_move_time, avg_acpl, mean_entropy]
```

### 2. Unsupervised Learning Models

**K-Means Clustering**:
- `k = 3` clusters (aggressive, balanced, conservative)
- Euclidean distance metric
- Random state = 42

**Gaussian Mixture Model**:
- 3 components with full covariance
- Expectation-Maximization fitting
- Model persistence via joblib

**Code Location**: `clustering.py:41-50`

## Statistical Process Control (`spc.py`)

### 1. Control Charts

**X-bar Chart** (process mean):
```
UCL = ¼ + 3Ã/n
LCL = ¼ - 3Ã/n
```

**R Chart** (process range):
```
UCL = D„ × R
LCL = Dƒ × R
```

**Individual-Moving Range Chart**:
- For single observations
- Moving range of consecutive values
- 3-sigma control limits

### 2. Process Capability

**Capability Indices**:
```
Cp = (USL - LSL) / (6Ã)
Cpk = min((USL - ¼)/3Ã, (¼ - LSL)/3Ã)
```

## Machine Learning Features (`ml_classifier.py`)

### 1. Feature Extraction Pipeline

**Engineered Features**:
- Rolling statistics (mean, std, min, max) over windows [3, 5, 10]
- Lag features (1, 2, 3 moves back)
- Interaction terms between timing and accuracy
- Fourier transform coefficients for periodicity detection

### 2. Classification Models

**Supported Algorithms**:
- Random Forest (default)
- XGBoost
- Neural Networks (PyTorch)
- Support Vector Machines

**Cross-Validation**:
- 5-fold stratified CV
- Time series split for temporal data
- Hyperparameter optimization via grid search

---

## Implementation Notes

### Numerical Stability
- All metrics include NaN/Inf handling
- Robust statistics (median, IQR) preferred over mean/std
- Clipping of extreme values to prevent outlier influence

### Performance Optimizations
- Vectorized operations using NumPy/Pandas
- Caching of expensive computations
- Progressive computation for real-time analysis

### Statistical Validation
- Bootstrap confidence intervals for all metrics
- Multiple testing correction (Bonferroni)
- Cross-validation for machine learning models

---

## References

1. **STL Decomposition**: Cleveland et al. (1990). "STL: A seasonal-trend decomposition procedure based on loess"
2. **Isolation Forest**: Liu et al. (2008). "Isolation Forest"
3. **Bayesian Change Points**: Adams & MacKay (2007). "Bayesian Online Changepoint Detection"
4. **GARCH Models**: Bollerslev (1986). "Generalized autoregressive conditional heteroskedasticity"
5. **Chess Engine Analysis**: Guid & Bratko (2006). "Computer analysis of world chess champions"

---

**See Also**:
- [Statistical Models](./statistical-models.md) - Detailed mathematical derivations
- [Performance Analysis](./performance.md) - Benchmarking and optimization
- [Analysis Modules](../modules/analysis/README.md) - Implementation details