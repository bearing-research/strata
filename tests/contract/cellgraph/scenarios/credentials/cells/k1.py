# @mount raw file://@NB@/data ro credential=lab
names = sorted(p.name for p in raw.iterdir())
print(names)
