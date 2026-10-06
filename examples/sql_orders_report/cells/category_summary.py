# @sql connection=warehouse
# @name category_summary
# @cache forever
# @after seed
-- The product catalog is reference data: it changes rarely and
-- the user's asserting "treat this as static". `forever` skips the
-- freshness probe; only an edit to the SQL body itself invalidates
-- the cache.
SELECT
    p.category,
    COUNT(DISTINCT p.sku) AS sku_count,
    COUNT(o.id)           AS order_count,
    ROUND(SUM(o.amount), 2) AS total_revenue
FROM products AS p
LEFT JOIN orders AS o USING (sku)
GROUP BY p.category
ORDER BY total_revenue DESC NULLS LAST
