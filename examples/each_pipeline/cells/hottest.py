# @name Hottest stations
# @sql connection=lake
# @after summarize
SELECT station, name, days, hottest_f
FROM lake.gsod.stations
ORDER BY hottest_f DESC
LIMIT 10
