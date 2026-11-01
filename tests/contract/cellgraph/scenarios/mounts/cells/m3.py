# @mount src file://@NB@/data/a ro
from_annotation = sorted(p.name for p in src.iterdir())
print(from_annotation)
