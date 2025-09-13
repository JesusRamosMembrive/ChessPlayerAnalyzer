# Statistical Models and Mathematical Foundations

## Overview

This document provides detailed mathematical foundations for the statistical models implemented in ChessPlayerAnalyzer. These models range from classical time series analysis to modern machine learning techniques, all specifically adapted for chess game analysis.

## Time Series Models

### 1. GARCH(1,1) - Generalized Autoregressive Conditional Heteroskedasticity

**Purpose**: Models time-varying volatility in chess performance metrics.

**Mathematical Specification**:
```
r(t) = ¼ + µ(t)
µ(t) = Ã(t) × z(t),  z(t) ~ N(0,1)
Ã²(t) = É + ± × µ²(t-1) + ² × Ã²(t-1)
```

**Parameter Constraints**:
- `É > 0` (non-negative long-term variance)
- `± e 0` (ARCH coefficient)
- `² e 0` (GARCH coefficient)
- `± + ² < 1` (stationarity condition)

**Implementation** (`performance_model.py:6-38`):
```python
def fit_garch_model(series: pd.Series) -> Dict[str, float]:
    # Moment-based parameter estimation
    alpha = 0.1  # ARCH coefficient
    beta = 0.8   # GARCH coefficient
    omega = max(var * (1 - alpha - beta), 1e-6)

    # Recursive variance estimation
    sigma2 = var
    for r in resid:
        sigma2 = omega + alpha * r**2 + beta * sigma2
```

**Applications in Chess Analysis**:
- Modeling volatility clustering in ACPL over time
- Detecting periods of inconsistent play
- Risk assessment for player performance

### 2. Kalman Filter

**Purpose**: State estimation for tracking true player skill over time.

**State Space Representation**:
```
State Equation:    x(t) = x(t-1) + w(t)
Observation Eq:    z(t) = x(t) + v(t)

where:
w(t) ~ N(0, Q)  [process noise]
v(t) ~ N(0, R)  [observation noise]
```

**Kalman Filter Equations**:

**Prediction Step**:
```
x(t|t-1) = x(t-1|t-1)
P(t|t-1) = P(t-1|t-1) + Q
```

**Update Step**:
```
K(t) = P(t|t-1) / (P(t|t-1) + R)
x(t|t) = x(t|t-1) + K(t) × (z(t) - x(t|t-1))
P(t|t) = (1 - K(t)) × P(t|t-1)
```

**Implementation** (`performance_model.py:80-90`):
```python
def fit_kalman_filter(series):
    x = float(series.iloc[0])  # Initial state
    p = 1.0                    # Initial covariance
    q = float(np.var(series.diff().dropna()))  # Process noise
    r = float(np.var(series) * 0.1)            # Observation noise

    for observation in series:
        # Prediction
        p = p + q
        # Update
        k = p / (p + r)
        x = x + k * (observation - x)
        p = (1 - k) * p
```

**Chess Applications**:
- Tracking true rating through noisy game results
- Smoothing performance metrics over time
- Real-time skill estimation

### 3. ARIMA Models

**Mathematical Form**:
```
(1 - ÆL - ÆL² - ... - ÆLV)(1 - L)HX(t) = (1 + ¸L + ¸L² + ... + ¸L`)µ(t)
```

**Simple AR(1) Implementation**:
```
X(t) = c + ÆX(t-1) + µ(t)
```

**Parameter Estimation** (Least Squares):
```python
# X(t) = c + ÆX(t-1) + µ(t)
y = series.iloc[1:].values
X = np.vstack([np.ones(len(y)), series.iloc[:-1].values]).T
c, phi = np.linalg.lstsq(X, y, rcond=None)[0]
```

## Change Point Detection

### 1. CUSUM (Cumulative Sum) Algorithm

**Mathematical Foundation**:
```
Sz(t) = max(0, Sz(t-1) + (x(t) - ¼ - k))
S{(t) = min(0, S{(t-1) + (x(t) - ¼ + k))

Change detected when: |Sz(t)| > h or |S{(t)| > h
```

**Parameters**:
- `k`: allowable drift (usually 0.5Ã)
- `h`: decision threshold (usually 4-5Ã)

**Implementation** (`change_point.py:7-40`):
```python
def cusum_change_points(series, threshold=None, drift=0.0):
    if threshold is None:
        threshold = 5 * np.std(series)

    pos_sum = neg_sum = 0.0
    change_points = []

    for i in range(1, len(series)):
        diff = series[i] - series[i-1] - drift
        pos_sum = max(0.0, pos_sum + diff)
        neg_sum = min(0.0, neg_sum + diff)

        if pos_sum > threshold or neg_sum < -threshold:
            change_points.append(i)
            pos_sum = neg_sum = 0.0
```

### 2. Bayesian Online Change Point Detection

**Mathematical Framework** (Adams & MacKay, 2007):

**Prior Distribution**:
```
P(r(t) = Ä) = H(Ä)bW{W(1 - H(i))

where H(Ä) is the hazard function
```

**Posterior Update**:
```
P(r(t)|x:t)  P(x(t)|r(t), x:t-1) × P(r(t)|x:t-1)
```

**Student-t Predictive Distribution**:
```
p(x(t)|x_{Ä+1:t-1}) = Student-t(2±, ¼, ²(º+1)/(±º))

where:
¼ = (º¼ + £x) / (º + t - Ä)
± = ± + (t - Ä) / 2
² = ² + 0.5£(x - ¼)² + º(t-Ä)(x - ¼)² / (2(º + t - Ä))
```

**Implementation** (`change_point.py:43-120`):
```python
def bayesian_online_change_points(series, hazard_lambda=250):
    hazard = 1.0 / hazard_lambda
    R = np.zeros((n+1, n+1))  # Run length probabilities

    for t, x in enumerate(data):
        # Predictive probabilities
        pred_probs = student_t_pdf(x, pred_mean, pred_var, pred_nu)

        # Update run length probabilities
        R[0, t+1] = np.sum(R[:t+1, t] * pred_probs * hazard)
        R[1:t+2, t+1] = R[:t+1, t] * pred_probs * (1 - hazard)

        # Normalize
        R[:t+2, t+1] /= np.sum(R[:t+2, t+1])
```

## Anomaly Detection Models

### 1. Isolation Forest

**Algorithm Principle**:
- Anomalies are "few and different"
- Random partitioning isolates anomalies faster
- Path length in isolation tree indicates anomaly score

**Mathematical Foundation**:
```
Anomaly Score = 2^(-E(h(x))/c(n))

where:
E(h(x)) = average path length of x over all trees
c(n) = 2H(n-1) - (2(n-1)/n)  [average path length of BST]
H(n) = ln(n) + ³  [harmonic number]
```

**Implementation Details**:
```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # Expected anomaly rate
    random_state=42
)
scores = -model.decision_function(features)  # Higher = more anomalous
```

### 2. STL Decomposition for Anomaly Detection

**Seasonal-Trend Decomposition**:
```
X(t) = T(t) + S(t) + R(t)

where:
T(t) = trend component
S(t) = seasonal component
R(t) = residual component
```

**Loess Smoothing** (for trend and seasonal):
```
Weighted regression: w(i) = (1 - (|x - xb|/d)³)³

where d is the distance to the q-th nearest neighbor
```

**Anomaly Detection**:
```
Z-score = (R(t) - ¼c) / Ãc
Anomaly if |Z-score| > threshold (typically 2-3)
```

## Bayesian Models

### 1. Beta-Binomial Model for Suspicion Priors

**Model Specification**:
```
Suspicious Games ~ Binomial(n, p)
p ~ Beta(±, ²)

Posterior: p|data ~ Beta(± + successes, ² + failures)
```

**Prior Update with Experience**:
```
P(suspicious | rating, exp) = ± / (± + ² + extra_exp)

where extra_exp = max(0, experience - bucket_start)
```

**Implementation** (`bayesian.py:48-69`):
```python
def compute_prior(self, rating, experience):
    r_bucket = self._rating_bucket(rating)
    e_bucket = self._exp_bucket(experience)

    params = self.priors.get(r_bucket, {}).get(e_bucket)
    if params:
        alpha, beta = params["alpha"], params["beta"]
        start = int(e_bucket.split('-')[0])
        extra = max(0, experience - start)
        p = alpha / (alpha + beta + extra)
    else:
        p = self.base_prior

    return self._clamp(p)  # Ensure [0.001, 0.999]
```

### 2. Naive Bayes for Evidence Integration

**Likelihood Ratio Updates**:
```
Posterior Odds = Prior Odds × b LR(evidenceb)

where LR(e) = P(e|suspicious) / P(e|not_suspicious)
```

**Evidence-Specific Likelihood Ratios**:
```python
likelihood_rules = {
    "acpl": lambda v: 5.0 if v < 20 else 1.0,
    "match_rate": lambda v: 4.0 if v > 0.7 else 1.0,
    "time_complexity_corr": lambda v: 3.0 if v < 0.1 else 1.0,
    "lag_spike_count": lambda v: 2.0 if v > 2 else 1.0,
}
```

## Clustering Models

### 1. K-Means Clustering

**Objective Function**:
```
J = £b £|O wb| ||xb - ¼|||²

where wb| = 1 if xb assigned to cluster j, 0 otherwise
```

**Lloyd's Algorithm**:
1. Initialize k centroids randomly
2. Assign points to nearest centroid
3. Update centroids: `¼| = (1/|C||) £bC| xb`
4. Repeat until convergence

**Implementation**:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)  # [mean_move_time, avg_acpl, mean_entropy]
```

### 2. Gaussian Mixture Models

**Model Specification**:
```
P(x) = £7 À N(x | ¼, £)

where:
À = mixing coefficients (£À = 1)
N(x|¼,£) = multivariate Gaussian
```

**EM Algorithm**:

**E-Step** (Expectation):
```
³(z) = ÀN(x|¼,£) / £| À|N(x|¼|,£|)
```

**M-Step** (Maximization):
```
À = (1/N) £ ³(z)
¼ = £ ³(z)x / £ ³(z)
£ = £ ³(z)(x-¼)(x-¼)@ / £ ³(z)
```

## Statistical Process Control

### 1. Control Charts

**Shewhart Control Charts**:
```
X-bar chart:
UCL = ¼ + 3Ã/n
CL = ¼
LCL = ¼ - 3Ã/n

R chart:
UCL = D × R
CL = R
LCL = D × R
```

**EWMA (Exponentially Weighted Moving Average)**:
```
EWMA(t) = »X(t) + (1-»)EWMA(t-1)

Control Limits:
UCL/LCL = ¼ ± L(»/(2-»)[1-(1-»)²W])Ã
```

### 2. Process Capability Indices

**Capability Indices**:
```
Cp = (USL - LSL) / (6Ã)          [potential capability]
Cpk = min(CPU, CPL)              [actual capability]
CPU = (USL - ¼) / (3Ã)           [upper capability]
CPL = (¼ - LSL) / (3Ã)           [lower capability]
```

**Performance Indices**:
```
Pp = (USL - LSL) / (6s)          [overall performance]
Ppk = min(PPU, PPL)              [overall performance index]
```

## Model Selection and Validation

### 1. Automatic Model Selection

**Volatility-Based Selection** (`performance_model.py:61-68`):
```python
def auto_select_model(series):
    volatility = np.std(series.diff().dropna())

    if volatility > 1.0:
        return "garch"      # High volatility  GARCH
    elif volatility > 0.1:
        return "arima"      # Medium volatility  ARIMA
    else:
        return "kalman"     # Low volatility  Kalman
```

### 2. Information Criteria

**Akaike Information Criterion (AIC)**:
```
AIC = 2k - 2ln(L)

where k = number of parameters, L = likelihood
```

**Bayesian Information Criterion (BIC)**:
```
BIC = k×ln(n) - 2ln(L)

where n = sample size
```

### 3. Cross-Validation

**Time Series Cross-Validation**:
- Forward chaining validation
- Expanding window approach
- Walk-forward optimization

**Implementation Strategy**:
```python
def time_series_cv(series, model_func, test_size=0.2):
    train_size = int(len(series) * (1 - test_size))

    for i in range(train_size, len(series)):
        train = series[:i]
        test = series[i:i+1]

        model = model_func(train)
        pred = predict(model, steps=1)
        # Evaluate prediction vs actual
```

## Numerical Considerations

### 1. Numerical Stability

**Precision Issues**:
- Use log-space computations for probabilities
- Numerical guards against division by zero
- Robust parameter initialization

**Example** (Log-likelihood computation):
```python
def log_likelihood(data, params):
    # Avoid overflow in exp() by working in log space
    log_probs = []
    for x in data:
        log_prob = log_pdf(x, params)
        log_probs.append(log_prob)

    return sum(log_probs)  # Instead of log(prod(probs))
```

### 2. Convergence Criteria

**EM Algorithm Convergence**:
```
|L(¸^(t+1)) - L(¸^(t))| < µ

where L(¸) is the log-likelihood
```

**GARCH Parameter Constraints**:
```python
def ensure_stationarity(alpha, beta):
    if alpha + beta >= 1:
        # Rescale to ensure stationarity
        sum_coef = alpha + beta
        alpha = alpha / sum_coef * 0.99
        beta = beta / sum_coef * 0.99

    return alpha, beta
```

## Performance Optimization

### 1. Vectorization

**NumPy Broadcasting**:
```python
# Instead of loops
def vectorized_computation(data):
    return np.sum(data ** 2, axis=1)  # Vectorized

# Avoid
def loop_computation(data):
    results = []
    for row in data:
        results.append(sum(x**2 for x in row))
    return results
```

### 2. Caching Strategy

**Memoization for Expensive Computations**:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_metric(game_hash, params_hash):
    # Expensive computation
    return result
```

---

## References

1. **Bollerslev, T. (1986)**. "Generalized autoregressive conditional heteroskedasticity". Journal of Econometrics.

2. **Adams, R. P., & MacKay, D. J. (2007)**. "Bayesian online changepoint detection". arXiv preprint arXiv:0710.3742.

3. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008)**. "Isolation forest". In Data Mining, 2008. ICDM'08.

4. **Cleveland, R. B., et al. (1990)**. "STL: A seasonal-trend decomposition procedure based on loess". Journal of Official Statistics.

5. **Kalman, R. E. (1960)**. "A new approach to linear filtering and prediction problems". Journal of Basic Engineering.

6. **Page, E. S. (1954)**. "Continuous inspection schemes". Biometrika.

7. **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977)**. "Maximum likelihood from incomplete data via the EM algorithm". Journal of the Royal Statistical Society.

---

**See Also**:
- [Metrics Documentation](./metrics.md) - Implementation details
- [Performance Analysis](./performance.md) - Computational optimization
- [Analysis Modules](../modules/analysis/README.md) - Code organization