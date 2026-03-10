import pandas as pd
import sys


# Load dataset
try:
    df = pd.read_csv('data/finflow_users.csv')
except FileNotFoundError:
    print("ERROR: Could not find 'data/finflow_users.csv'. "
        "Please ensure the dataset is placed in the 'data/' subdirectory "
        "relative to this script.")
    sys.exit(1)

# ============================================
# PART 1: SAMPLE SPACES & BASIC PROBABILITY
# ============================================

# Define sample space for premium_user
sample_space_premium = set(df['premium_user'].unique())

# P(premium_user = 1)
p_premium = (df['premium_user'] == 1).mean()

# P(score_views >= 5)
p_high_engagement = (df['score_views'] >= 5).mean()

# P(risk_profile = 'aggressive')
p_aggressive = (df['risk_profile'] == 'aggressive').mean()

# Joint probability P(score_views >= 5 AND premium_user = 1)
p_joint = ((df['score_views'] >= 5) & (df['premium_user'] == 1)).mean()

# ============================================
# PART 2: CONDITIONAL PROBABILITY & BAYES
# ============================================

# P(premium = 1 | score_views >= 3)
engaged_mask = df['score_views'] >= 3
p_premium_given_engaged = df.loc[engaged_mask, 'premium_user'].mean()

# P(score_views >= 3 | premium = 1)
premium_mask = df['premium_user'] == 1
p_engaged_given_premium = (df.loc[premium_mask, 'score_views'] >= 3).mean()

# P(score_views >= 3)
p_engaged = engaged_mask.mean()

# Verify Bayes' theorem
# P(premium | engaged) = P(engaged | premium) * P(premium) / P(engaged)
bayes_check = (p_engaged_given_premium * p_premium) / p_engaged

# Odds ratio
# odds_ratio = [P(premium|engaged) / (1 - P(premium|engaged))] / [P(premium) / (1 - P(premium))]
odds_premium_given_engaged = p_premium_given_engaged / (1 - p_premium_given_engaged)
odds_premium = p_premium / (1 - p_premium)
odds_ratio = odds_premium_given_engaged / odds_premium

# ============================================
# PART 3: RANDOM VARIABLE CLASSIFICATION
# ============================================

classifications = {
    'days_active': {
        'type': 'discrete',
        'support': 'non-negative integers {0, 1, 2, ..., 365} (whole days)',
        'distribution': 'Poisson or Negative Binomial',
        'justification': 'Days active is a countable whole-number quantity bounded by the calendar period, making it a discrete count variable.'
    },
    'score_views': {
        'type': 'discrete',
        'support': 'non-negative integers {0, 1, 2, ...}',
        'distribution': 'Poisson',
        'justification': 'Count of independent events (score views) over fixed period.'
    },
    'session_minutes': {
        'type': 'continuous',
        'support': 'non-negative real numbers [0, ∞)',
        'distribution': 'Log-Normal or Exponential',
        'justification': 'Session duration is measured in real-valued time units and can take any positive fractional value, making it inherently continuous.'
    },
    'risk_profile': {
        'type': 'categorical',
        'support': 'finite unordered set {conservative, moderate, aggressive}',
        'distribution': 'Categorical (Multinomial)',
        'justification': 'Risk profile represents mutually exclusive named categories with no natural numeric ordering, best modelled as a categorical distribution.'
    },
    'premium_user': {
        'type': 'binary',
        'support': '{0, 1} (0 = non-premium, 1 = premium)',
        'distribution': 'Bernoulli',
        'justification': 'Premium user status is a binary outcome (subscribed or not) with a fixed probability of success, defining a Bernoulli trial.'
    }
}

# ============================================
# VALIDATION CHECKS (do not modify)
# ============================================
assert 0 <= p_premium <= 1, "P(premium) must be between 0 and 1"
assert 0 <= p_high_engagement <= 1, "P(high engagement) must be between 0 and 1"
assert 0 <= p_aggressive <= 1, "P(aggressive) must be between 0 and 1"
assert 0 <= p_joint <= 1, "Joint probability must be between 0 and 1"
assert abs(p_premium_given_engaged - bayes_check) < 0.01, "Bayes' theorem verification failed"
assert odds_ratio > 0, "Odds ratio must be positive"
assert all(classifications[var]['type'] for var in classifications), "All variables must be classified"

# ============================================
# BUSINESS INTERPRETATION
# ============================================

print("="*70)
print("PART 1: BASIC PROBABILITIES")
print("="*70)

print(f"P(premium user): {p_premium:.1%}")
print(f"  → Interpretation: Approximately {p_premium:.1%} of FinFlow users have "
    f"converted to a premium subscription, establishing our baseline conversion rate "
    f"for benchmarking engagement-driven campaigns.")

print(f"\nP(high engagement — score_views ≥ 5): {p_high_engagement:.1%}")
print(f"  → Interpretation: {p_high_engagement:.1%} of users view five or more credit "
    f"self-monitoring and therefore prime candidates for premium upsell.")

print(f"\nP(aggressive risk profile): {p_aggressive:.1%}")
print(f"  → Interpretation: {p_aggressive:.1%} of users self-identify as aggressive "
    f"investors, representing a growth-oriented segment that may respond strongly "
    f"to premium features like advanced portfolio analytics.")

print(f"\nJoint P(score_views ≥ 5 AND premium): {p_joint:.1%}")
print(f"  → Interpretation: Only {p_joint:.1%} of all users are both highly engaged "
    f"and premium subscribers; comparing this to marginal probabilities reveals "
    f"whether high engagement and premium status co-occur more than by chance.")

print("\n" + "="*70)
print("PART 2: CONDITIONAL PROBABILITY & BAYES")
print("="*70)

print(f"P(premium | score_views ≥ 3): {p_premium_given_engaged:.1%}")
print(f"  → Among users who view three or more scores, {p_premium_given_engaged:.1%} "
    f"are premium subscribers — substantially higher than the baseline, confirming "
    f"that engagement is a strong leading indicator of conversion.")

print(f"\nP(score_views ≥ 3 | premium): {p_engaged_given_premium:.1%}")
print(f"  → {p_engaged_given_premium:.1%} of premium users view three or more scores, "
    f"suggesting that engagement is not just a predictor of conversion but also a "
    f"characteristic behaviour of the premium cohort.")

print(f"\nBayes' theorem verification: {bayes_check:.4f} ≈ {p_premium_given_engaged:.4f} ✓")
print(f"  → The computed Bayes check matches the direct conditional probability "
    f"within tolerance, confirming arithmetic consistency.")

print(f"\nOdds ratio: {odds_ratio:.2f}x")
print(f"  → Interpretation: Users who view three or more scores are {odds_ratio:.2f}x "
    f"more likely (in odds terms) to be premium subscribers than the average user. "
    f"Recommendation: Invest in product features and nudges that encourage users to "
    f"reach the 3+ score-view threshold (e.g., personalised score alerts or "
    f"educational prompts), as crossing this engagement threshold is strongly "
    f"associated with premium conversion.")

print("\n" + "="*70)
print("PART 3: VARIABLE CLASSIFICATION")
print("="*70)
print(f"{'Variable':<20} {'Type':<15} {'Distribution':<25} {'Support'}")
print("-"*80)
for var, props in classifications.items():
    print(f"{var:<20} {props['type']:<15} {props['distribution']:<25} {props['support']}")

print("\nJustifications:")
for var, props in classifications.items():
    print(f"  {var}: {props['justification']}")

print("="*70)

# Critical thinking question
print("\n❓ Why is score_views discrete but session_minutes continuous?")
print(
    "   → Answer: score_views is discrete because it counts whole, indivisible events "
    "(each score lookup is either viewed or not — you cannot view 2.7 scores). "
    "Its support is the set of non-negative integers {0, 1, 2, ...}. "
    "session_minutes, by contrast, is continuous because time is measured on a real-valued "
    "scale: a session can last 4.37 minutes or 12.891 minutes — any positive real number "
    "is theoretically possible. The fundamental distinction is that counts are indivisible "
    "discrete units whereas durations are divisible real-valued measurements."
)