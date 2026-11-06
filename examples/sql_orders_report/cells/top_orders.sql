# @sql connection=warehouse
# @name top_orders
# @cache fingerprint
# @after seed
SELECT
    o.id,
    o.customer,
    o.sku,
    p.category,
    o.amount,
    o.ordered_at
FROM orders AS o
JOIN products AS p USING (sku)
WHERE o.amount > :min_amount
ORDER BY o.amount DESC
LIMIT 5
