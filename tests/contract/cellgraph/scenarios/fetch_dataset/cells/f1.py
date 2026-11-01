# @fetch zones http://127.0.0.1:47613/zones.csv sha256=170131825600846a79ab8a08b8e1f7509c8f0317fe17ae6b52d9a1706fd8addd
rows = zones.read_text().splitlines()
print(len(rows))
