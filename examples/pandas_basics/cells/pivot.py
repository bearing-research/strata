# @name Monthly revenue pivot
monthly_pivot = sales.pivot_table(
    values="revenue",
    index="month",
    columns="region",
    aggfunc="sum",
).round(2)

print(monthly_pivot)
