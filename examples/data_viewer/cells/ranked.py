# @name Rank by revenue
# A derived frame gets the same viewer. The grid's column sort is applied
# server-side over the *whole* frame, so it re-orders independently of this
# initial sort — click any header to prove it.
ranked = transactions.sort_values("revenue", ascending=False).reset_index(drop=True)

print(f"Top revenue: ${ranked['revenue'].iloc[0]:,.2f}")
ranked
