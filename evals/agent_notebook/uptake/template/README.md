# harbor

Order events, pricing and shipping for the harbor marketplace.

- `harbor.data.load_events()`: every order event, as a DataFrame.
- `harbor.pricing.quote(product, qty, region)`: a price quote.
- `harbor.shipping.eta_days(origin, dest)`: days to ship between two regions.
- `harbor.scoring.risk_score(customer_id)`: a customer's risk, from 0 to 1.
- `fleet` (in `packages/fleet`): regions, distances and rates.

Run the tests with `uv run pytest`.
