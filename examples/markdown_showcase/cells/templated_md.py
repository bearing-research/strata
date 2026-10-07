# @name Templated report from a DataFrame
#
# A more realistic dynamic-markdown pattern: take some tabular result
# and render a small report. Useful for "summary at the end of a
# notebook" cells where you want prose + data in one display, not
# stacked output blocks.

import pandas as pd

df = pd.DataFrame(
    {
        "model_id": [
            "claude-sonnet-4-6",
            "gpt-5.4",
            "gemini-3-flash",
            "mistral-large-latest",
        ],
        "mmlu": [0.8412, 0.8205, 0.7891, 0.7634],
        "gsm8k": [0.9421, 0.9387, 0.9012, 0.8723],
        "humaneval": [0.8521, 0.8234, 0.7823, 0.7123],
    }
)

# Score columns to summarize.
score_cols = [c for c in df.columns if c != "model_id"]
df["mean"] = df[score_cols].mean(axis=1)
ranked = df.sort_values("mean", ascending=False).reset_index(drop=True)

best = ranked.iloc[0]
worst = ranked.iloc[-1]
spread = (best["mean"] - worst["mean"]) * 100

# Pretty table: markdown table syntax with right-aligned scores.
table_lines = [
    "| Rank | Model | MMLU | GSM8K | HumanEval | Mean |",
    "|:----:|-------|-----:|------:|----------:|-----:|",
]
for i, row in ranked.iterrows():
    table_lines.append(
        f"| {i + 1} | `{row['model_id']}` "
        f"| {row['mmlu']:.1%} | {row['gsm8k']:.1%} "
        f"| {row['humaneval']:.1%} | {row['mean']:.1%} |"
    )
table = "\n".join(table_lines)

display(
    Markdown(
        f"""
## Mock leaderboard report

**{len(ranked)} models** scored across **{len(score_cols)} evals**. Top
performer: `{best['model_id']}` at `{best['mean']:.1%}` mean. Spread
between best and worst: **{spread:.1f} points**.

{table}

---

*Report generated dynamically from `pd.DataFrame` → `Markdown`,
exactly the pattern you'd use for "summary at end of notebook" cells.*
"""
    )
)
