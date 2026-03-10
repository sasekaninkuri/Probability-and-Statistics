import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

# ============================================
# FILE LOADING
# ============================================

try:
    df = pd.read_csv('data/finflow_users.csv')
except FileNotFoundError:
    raise FileNotFoundError("Missing required file: data/finflow_users.csv. Please place it in the data/ subdirectory.")

try:
    ab_df = pd.read_csv('data/finflow_ab_test.csv')
except FileNotFoundError:
    raise FileNotFoundError("Missing required file: data/finflow_ab_test.csv. Please place it in the data/ subdirectory.")

# ============================================
# PART 1: CONFIDENCE INTERVALS
# ============================================

n = len(df)
mean_minutes = df['session_minutes'].mean()
sd_minutes = df['session_minutes'].std(ddof=1)
se_minutes = sd_minutes / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n - 1)
ci_mean_lower = mean_minutes - t_crit * se_minutes
ci_mean_upper = mean_minutes + t_crit * se_minutes
margin_error_mean = t_crit * se_minutes

# 95% Wilson score interval for premium conversion rate
successes = df['premium_user'].sum()
n_total = len(df)
ci_prop_lower, ci_prop_upper = proportion_confint(successes, n_total, alpha=0.05, method='wilson')
p_hat = successes / n_total
margin_error_prop = (ci_prop_upper - ci_prop_lower) / 2

# ============================================
# PART 2: BOOTSTRAP METHODS
# ============================================

session_minutes = df['session_minutes'].values
n_boot = 10000
rng = np.random.default_rng(42)

bootstrap_medians = np.zeros(n_boot)
for i in range(n_boot):
    sample = rng.choice(session_minutes, size=len(session_minutes), replace=True)
    bootstrap_medians[i] = np.median(sample)

ci_boot_lower = np.percentile(bootstrap_medians, 2.5)
ci_boot_upper = np.percentile(bootstrap_medians, 97.5)
point_estimate_median = np.median(session_minutes)

# ============================================
# PART 3: HYPOTHESIS TESTING I (T-TEST)
# ============================================

free_users = df[df['premium_user'] == 0]['session_minutes']
premium_users = df[df['premium_user'] == 1]['session_minutes']

h0_ttest = "H₀: μ_premium ≤ μ_free (premium users do not have longer sessions)"
ha_ttest = "Hₐ: μ_premium > μ_free (premium users have longer sessions)"

# Shapiro-Wilk normality test
shapiro_free_stat, shapiro_free_p = stats.shapiro(free_users)
shapiro_premium_stat, shapiro_premium_p = stats.shapiro(premium_users)
normality_ok = (shapiro_free_p > 0.05) and (shapiro_premium_p > 0.05)

# Levene's test for equal variance
levene_stat, levene_p = stats.levene(free_users, premium_users)
equal_var = levene_p > 0.05

# Welch's t-test (one-tailed)
t_stat, p_two_tail = stats.ttest_ind(premium_users, free_users, equal_var=False)
p_value_ttest = p_two_tail / 2 if t_stat > 0 else 1 - p_two_tail / 2
reject_h0_ttest = p_value_ttest < 0.05

# Cohen's d with pooled SD
n1, n2 = len(free_users), len(premium_users)
sd1, sd2 = free_users.std(ddof=1), premium_users.std(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
cohens_d = (premium_users.mean() - free_users.mean()) / pooled_sd

# Approximate sample size for 80% power (two-sample t-test formula)
# n ≈ 2 * ((z_alpha + z_beta) / d)^2  where d = Cohen's d
z_alpha = stats.norm.ppf(0.95)   # one-tailed α = 0.05
z_beta = stats.norm.ppf(0.80)    # 80% power
n_needed_ttest = int(np.ceil(2 * ((z_alpha + z_beta) / cohens_d) ** 2))

# ============================================
# PART 4: HYPOTHESIS TESTING II (CHI-SQUARE)
# ============================================

contingency_table = pd.crosstab(df['risk_profile'], df['premium_user'])
chi2_stat, p_value_chi2, dof_chi2, expected = chi2_contingency(contingency_table)

min_expected = expected.min()
assumption_met = min_expected >= 5 or (
    np.sum(expected >= 5) / expected.size >= 0.8 and min_expected >= 1
)

n_chi2 = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2_stat / (n_chi2 * min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)))

# ============================================
# PART 5: MULTIPLE COMPARISONS (A/B TEST)
# ============================================

conversion_rates = ab_df.groupby('variant')['converted'].mean()
control_rate = conversion_rates['control']

results = []
variants_to_test = ['variant_a', 'variant_b', 'variant_c', 'variant_d']
alpha = 0.05
m = len(variants_to_test)
alpha_adj = alpha / m  # Bonferroni = 0.0125

control_data = ab_df[ab_df['variant'] == 'control']
n_control = len(control_data)
successes_control = control_data['converted'].sum()

for variant in variants_to_test:
    variant_data = ab_df[ab_df['variant'] == variant]
    n_variant = len(variant_data)
    successes_variant = variant_data['converted'].sum()

    count = np.array([successes_variant, successes_control])
    nobs = np.array([n_variant, n_control])
    stat_ab, p_value_ab = proportions_ztest(count, nobs, alternative='larger')

    significant = p_value_ab < alpha_adj

    variant_rate = conversion_rates[variant]
    abs_lift = variant_rate - control_rate
    rel_lift = (variant_rate - control_rate) / control_rate

    results.append({
        'variant': variant,
        'conversion_rate': variant_rate,
        'p_value': p_value_ab,
        'significant': significant,
        'abs_lift': abs_lift,
        'rel_lift': rel_lift
    })

results_df = pd.DataFrame(results)

# ============================================
# VALIDATION CHECKS
# ============================================

assert ci_mean_lower < mean_minutes < ci_mean_upper, "Mean CI must contain point estimate"
assert ci_prop_lower < p_hat < ci_prop_upper, "Proportion CI must contain point estimate"
assert ci_boot_lower < point_estimate_median < ci_boot_upper, "Bootstrap CI must contain median"
assert cohens_d > 0, "Cohen's d should be positive (premium > free)"
assert p_value_chi2 >= 0 and p_value_chi2 <= 1, "p-value must be between 0 and 1"
assert len(results_df) == 4, "Must test 4 variants"

# ============================================
# RESULTS & INTERPRETATION
# ============================================

print("CONFIDENCE INTERVALS (95%)")
print("=" * 70)
print(f"{'Metric':<25} {'Point Estimate':<20} {'95% CI':<30} {'Margin of Error'}")
print("-" * 70)
print(f"{'Session duration (min)':<25} {mean_minutes:<20.1f} ({ci_mean_lower:.1f}, {ci_mean_upper:.1f}){'':<10} ±{margin_error_mean:.1f}")
print(f"{'Premium conversion':<25} {p_hat:<20.1%} ({ci_prop_lower:.1%}, {ci_prop_upper:.1%}){'':<8} ±{margin_error_prop:.1%}")
print("=" * 70)
print("\nBUSINESS INTERPRETATION:")
print(f"  Mean session duration: We are 95% confident the true average session length")
print(f"    falls between {ci_mean_lower:.1f} and {ci_mean_upper:.1f} minutes. Even the lower bound")
print(f"    suggests meaningful engagement time worth optimising for.")
print(f"  Premium conversion rate: The true conversion rate likely falls between")
print(f"    {ci_prop_lower:.1%} and {ci_prop_upper:.1%}. This range is narrow, indicating a")
print(f"    reliable estimate for revenue forecasting.")
print(f"\nRISK ASSESSMENT (lower bound scenario):")
print(f"  In the worst case, session duration could be as low as {ci_mean_lower:.1f} min and")
print(f"  conversion as low as {ci_prop_lower:.1%}. Plans should be stress-tested against")
print(f"  these pessimistic but plausible values.")

print("\n" + "=" * 70)
print("BOOTSTRAP VS PARAMETRIC CONFIDENCE INTERVALS")
print("=" * 70)
boot_width = ci_boot_upper - ci_boot_lower
param_width = ci_mean_upper - ci_mean_lower
print(f"{'Statistic':<15} {'Point Estimate':<20} {'95% CI Width':<20} {'Relative Width'}")
print("-" * 70)
print(f"{'Median (boot)':<15} {point_estimate_median:<20.1f} {boot_width:<20.1f} 1.00x")
print(f"{'Mean (param)':<15} {mean_minutes:<20.1f} {param_width:<20.1f} {param_width / boot_width:.2f}x")
print("=" * 70)
print("\nBOOTSTRAP INTERPRETATION:")
print(f"  Median session duration: {point_estimate_median:.1f} min (95% CI: {ci_boot_lower:.1f} to {ci_boot_upper:.1f})")
print(f"  → Interpretation: The bootstrap CI is distribution-free and robust to outliers.")
print(f"    The median represents the 'typical' user better when the distribution is skewed.")
print(f"\nRECOMMENDATION:")
print(f"  If session duration is right-skewed (as is common), use the median as the")
print(f"  'typical user' benchmark. Use the mean for total engagement volume calculations.")

print("\n" + "=" * 70)
print("HYPOTHESIS TEST: PREMIUM VS FREE SESSION DURATION")
print("=" * 70)
print(f"Hypotheses:")
print(f"  {h0_ttest}")
print(f"  {ha_ttest}")
print(f"\nAssumption Checks:")
print(f"  Normality (Shapiro-Wilk): free p={shapiro_free_p:.3f}, premium p={shapiro_premium_p:.3f} → {'OK' if normality_ok else 'VIOLATED'}")
print(f"  Equal variance (Levene):  p={levene_p:.3f} → {'OK' if equal_var else 'VIOLATED (use Welch)'}")
print(f"\nTest Results (Welch's t-test, one-tailed):")
print(f"  t = {t_stat:.2f}, p = {p_value_ttest:.4f}")
print(f"  Decision: {'REJECT H₀' if reject_h0_ttest else 'FAIL TO REJECT H₀'}")
print(f"\nEffect Size:")
print(f"  Cohen's d = {cohens_d:.2f} ({'small' if abs(cohens_d) < 0.2 else 'medium' if abs(cohens_d) < 0.5 else 'large'})")
print(f"  Required n per group for 80% power: ~{n_needed_ttest:.0f}")
print("=" * 70)
print("\nBUSINESS INTERPRETATION:")
print(f"  Premium users are associated with {'significantly' if reject_h0_ttest else 'no significantly'} longer session durations")
print(f"  than free users (Cohen's d = {cohens_d:.2f}). This is an observational finding—")
print(f"  premium status is associated with higher engagement, but we cannot conclude it")
print(f"  causes it. Confounding factors (e.g., user motivation) may explain the difference.")

print("\n" + "=" * 70)
print("CHI-SQUARE TEST: RISK PROFILE vs PREMIUM CONVERSION")
print("=" * 70)
print("Contingency Table (Observed Counts):")
print(contingency_table)
print(f"\nExpected Counts (under independence):")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).round(1))
print(f"\nAssumption Check: min expected count = {min_expected:.1f} → {'OK' if assumption_met else 'VIOLATED'}")
print(f"\nTest Results:")
print(f"  χ² = {chi2_stat:.2f}, df = {dof_chi2}, p = {p_value_chi2:.4f}")
print(f"  Cramér's V = {cramers_v:.3f} ({'very weak' if cramers_v < 0.1 else 'weak' if cramers_v < 0.3 else 'moderate' if cramers_v < 0.5 else 'strong'})")
print("=" * 70)
print("\nBUSINESS INTERPRETATION:")
if p_value_chi2 < 0.05:
    print(f"  Risk profile and premium conversion are statistically associated (p={p_value_chi2:.4f}).")
    print(f"  The effect is {('very weak' if cramers_v < 0.1 else 'weak' if cramers_v < 0.3 else 'moderate' if cramers_v < 0.5 else 'strong')} (Cramér's V={cramers_v:.3f}),")
    print(f"  meaning risk profile explains only a modest portion of conversion variance.")
else:
    print(f"  No statistically significant association between risk profile and premium")
    print(f"  conversion was detected (p={p_value_chi2:.4f}). Risk profile alone does not")
    print(f"  reliably predict who converts to premium.")
print(f"\nMARKETING RECOMMENDATION:")
if p_value_chi2 < 0.05 and cramers_v >= 0.1:
    # Find which risk profile has highest conversion
    risk_conv = df.groupby('risk_profile')['premium_user'].mean()
    best_risk = risk_conv.idxmax()
    print(f"  Prioritise targeting '{best_risk}' risk-profile users in premium upgrade campaigns,")
    print(f"  as they show the highest observed conversion rate. However, given the modest")
    print(f"  effect size, segment other behavioural signals alongside risk profile.")
else:
    print(f"  Do not rely solely on risk profile for premium targeting—the association is too")
    print(f"  weak to drive meaningful segmentation lift. Use multi-variate models combining")
    print(f"  risk profile with session engagement and product usage signals instead.")

print("\n" + "=" * 80)
print("A/B TEST RESULTS WITH BONFERRONI CORRECTION")
print("=" * 80)
print(f"Control conversion rate: {control_rate:.2%}")
print(f"Bonferroni-adjusted α: {alpha:.2f}/{m} = {alpha_adj:.4f}\n")
print(results_df.to_string(index=False,
    formatters={
        'conversion_rate': '{:.2%}'.format,
        'p_value': '{:.4f}'.format,
        'abs_lift': '{:+.2%}'.format,
        'rel_lift': '{:+.1%}'.format
    }))
print("=" * 80)

significant_variants = results_df[results_df['significant']]
if len(significant_variants) > 0:
    best = significant_variants.loc[significant_variants['abs_lift'].idxmax()]
    print(f"\nDEPLOY VARIANT {best['variant'].upper()}:")
    print(f"   Lift: {best['abs_lift']:+.2%} absolute ({best['rel_lift']:+.0%} relative)")
    print(f"   p-value: {best['p_value']:.4f} < {alpha_adj:.4f} (Bonferroni-adjusted)")
    print(f"\n   JUSTIFICATION: This variant survived family-wise error control with Bonferroni")
    print(f"   correction, meaning the observed lift is unlikely due to chance even after")
    print(f"   accounting for testing {m} variants simultaneously. The absolute lift of")
    print(f"   {best['abs_lift']:+.2%} translates directly to incremental revenue at scale.")
    print(f"   Recommend a staged rollout (10% → 50% → 100%) while monitoring guardrail")
    print(f"   metrics (session duration, churn rate) before full deployment.")
else:
    print(f"\nNO VARIANTS SURVIVE BONFERRONI CORRECTION")
    best_row = results_df.loc[results_df['abs_lift'].idxmax()]
    print(f"   Best performer: {best_row['variant']} (p={best_row['p_value']:.4f}, lift={best_row['abs_lift']:+.2%})")
    print(f"   RECOMMENDATION: Do not deploy any variant. Consider running a larger test")
    print(f"   (increase sample size to achieve 80% power) or iterating on variant designs")
    print(f"   to produce a stronger signal before committing to rollout.")