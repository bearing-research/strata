# @name triage_summary
# Downstream cells consume the LLM output as structured data: no
# parsing, no fallbacks, no "did the model actually return JSON this
# time?" defensive code. The schema guarantees the shape.
import pandas as pd

rows = triage["items"]
df = pd.DataFrame(rows)

print("Per-review triage:")
for row, review in zip(rows, reviews):
    tags = ", ".join(row["tags"])
    print(
        f"  [{row['review_index']}] "
        f"{row['sentiment']:>8} / {row['priority']:>6} "
        f"({tags}): {review[:60]}..."
    )

print()
print("By priority:")
print(df["priority"].value_counts().to_string())

print()
print("Sentiment distribution:")
print(df["sentiment"].value_counts().to_string())

# The final expression becomes the cell's display value: a compact
# record of what the triage flagged as high-priority.
high_priority = df[df["priority"] == "high"]
{
    "total_reviews": len(df),
    "needs_escalation": len(high_priority),
    "negative_rate": float((df["sentiment"] == "negative").mean()),
}
