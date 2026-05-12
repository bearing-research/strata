# @name Survival rates by class and sex
survival_rates = df.groupby(["pclass", "sex"])["survived"].mean().round(3)
print(survival_rates)
print(f"\nOverall: {df['survived'].mean():.3f}")
