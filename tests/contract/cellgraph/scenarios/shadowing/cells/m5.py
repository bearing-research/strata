# @mount input file://@NB@/data ro
print(sorted(p.name for p in input.iterdir()))
