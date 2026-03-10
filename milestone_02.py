import pandas as pd
import numpy as np
from scipy import stats
import sys

try:
    df = pd.read_csv('data/finflow_users.csv')
except FileNotFoundError:
    print("ERROR: Could not find 'data/finflow_users.csv'.")
    sys.exit(1)

session_minutes = df['session_minutes'].values
score_views = df['score_views'].values

# ============================================
# PART 1: MOMENTS & SHAPE ANALYSIS
# ============================================

mean_minutes = session_minutes.mean()
variance_minutes = session_minutes.var(ddof=1)
skewness_minutes = stats.skew(session_minutes, bias=False)
kurtosis_minutes = stats.kurtosis(session_minutes, bias=False)

# ============================================
# PART 2: DISTRIBUTION FITTING
# ============================================

# Fit Poisson to score_views (lambda = mean for Poisson)
lambda_poisson = score_views.mean()

# Fit Normal to session_minutes
mu_normal, sigma_normal = stats.norm.fit(session_minutes)

# KS test for Poisson
ks_stat_poisson, p_value_poisson = stats.kstest(score_views, lambda k: stats.poisson.cdf(k, lambda_poisson))

# KS test for Normal
ks_stat_normal, p_value_normal = stats.kstest(session_minutes, 'norm', args=(mu_normal, sigma_normal))

# ============================================
# PART 3: CLT SIMULATION
# ============================================

pop_mean = session_minutes.mean()
pop_std = session_minutes.std(ddof=1)
sample_sizes = [10, 30, 100]
n_reps = 10000

np.random.seed(42)
sampling_distributions = {}
for n in sample_sizes:
    sample_means = np.array([
        np.random.choice(session_minutes, size=n, replace=True).mean()
        for _ in range(n_reps)
    ])
    sampling_distributions[n] = sample_means

empirical_ses = {n: np.std(means, ddof=1) for n, means in sampling_distributions.items()}
theoretical_ses = {n: pop_std / np.sqrt(n) for n in sample_sizes}

# Minimum n where |skewness| < 0.5
min_n_normal = None
for n in sample_sizes:
    if abs(stats.skew(sampling_distributions[n], bias=False)) < 0.5:
        min_n_normal = n
        break

# ============================================
# VALIDATION
# ============================================

assert mean_minutes > 0, "Mean must be positive"
assert variance_minutes > 0, "Variance must be positive"
assert lambda_poisson > 0, "Poisson lambda must be positive"
assert sigma_normal > 0, "Normal sigma must be positive"

for n in sample_sizes:
    assert abs(empirical_ses[n] - theoretical_ses[n]) / theoretical_ses[n] < 0.1, \
        f"SE mismatch for n={n}"

# ============================================
# OUTPUT
# ============================================

print("SESSION DURATION MOMENTS")
print("="*50)
print(f"Mean:     {mean_minutes:.2f} minutes")
print(f"Variance: {variance_minutes:.2f} (SD = {variance_minutes**0.5:.2f})")
print(f"Skewness: {skewness_minutes:.2f}")
print(f"Kurtosis: {kurtosis_minutes:.2f} (excess)")

print("\nSHAPE INTERPRETATION:")
if skewness_minutes > 0:
    print(f"  Skewness ({skewness_minutes:.2f}): Right-skewed — most users have short sessions but a small group of power users have much longer ones, pulling the mean above the median.")
else:
    print(f"  Skewness ({skewness_minutes:.2f}): Left-skewed — most sessions are long with a few unusually short ones.")

if kurtosis_minutes > 0:
    print(f"  Kurtosis ({kurtosis_minutes:.2f}): Heavy tails (leptokurtic) — more extreme session lengths than a Normal distribution would predict.")
else:
    print(f"  Kurtosis ({kurtosis_minutes:.2f}): Light tails (platykurtic) — fewer extreme values than a Normal distribution.")

print(f"\nBUSINESS IMPLICATION:")
print(f"  The skewed session distribution means the average session time is misleading as a single metric. The product team should track median session length alongside the mean, and design features that convert casual short-session users into more engaged regular users.")

print("\n" + "="*60)
print("DISTRIBUTION FITTING RESULTS")
print("="*60)
print(f"{'Distribution':<15} {'Parameter(s)':<25} {'KS Stat':<10} {'p-value':<10}")
print("-"*60)
print(f"Poisson         lambda = {lambda_poisson:.2f}{'':<12} {ks_stat_poisson:.3f}     {p_value_poisson:.3f}")
print(f"Normal          mu = {mu_normal:.2f}, sigma = {sigma_normal:.2f}  {ks_stat_normal:.3f}     {p_value_normal:.3f}")
print("="*60)

print("\nGOODNESS-OF-FIT INTERPRETATION:")
if p_value_poisson < 0.05:
    print(f"  Poisson fit: Poor fit (p={p_value_poisson:.3f} < 0.05) — score_views data does not follow a Poisson distribution well, likely due to overdispersion or user behaviour clustering.")
else:
    print(f"  Poisson fit: Acceptable fit (p={p_value_poisson:.3f} >= 0.05) — Poisson is a reasonable model for score_views.")

if p_value_normal < 0.05:
    print(f"  Normal fit: Poor fit (p={p_value_normal:.3f} < 0.05) — session_minutes is skewed and does not follow a Normal distribution; Log-Normal or Exponential would be more appropriate.")
else:
    print(f"  Normal fit: Acceptable fit (p={p_value_normal:.3f} >= 0.05) — Normal is a reasonable model for session_minutes.")

print(f"\nRECOMMENDATION FOR SIMULATION MODELS:")
print(f"  Use a Log-Normal distribution for session_minutes (captures right skew and non-negative values) and a Negative Binomial for score_views if Poisson shows overdispersion. These choices will produce more realistic simulated user behaviour than assuming Normality.")

print("\n" + "="*60)
print("CLT CONVERGENCE RESULTS")
print("="*60)
print(f"{'Sample Size (n)':<20} {'Empirical SE':<18} {'Theoretical SE':<18} {'Ratio'}")
print("-"*60)
for n in sample_sizes:
    ratio = empirical_ses[n] / theoretical_ses[n]
    print(f"{n:<20} {empirical_ses[n]:<18.2f} {theoretical_ses[n]:<18.2f} {ratio:.3f}")
print("="*60)
print(f"\nMinimum n for approximate Normality (|skew| < 0.5): {min_n_normal}")
print(f"\nBUSINESS IMPLICATION:")
print(f"  For A/B tests on session duration, use a minimum sample size of {min_n_normal} per group. Below this threshold the sampling distribution is still skewed, meaning p-values and confidence intervals may be unreliable. With n >= {min_n_normal} per variant, the CLT guarantees the sample mean is approximately Normal, making standard statistical tests valid.")