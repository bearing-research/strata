# @name daily_note
# @temperature 0.2
# @max_tokens 800
# @system You are a trading journal assistant. Your job is to produce a concise, honest end-of-day reflection on a small paper-trading strategy. Do not sugarcoat: if the strategy is losing money or being eaten by costs, say so directly.
# @output_schema {"type": "object", "properties": {"one_liner": {"type": "string"}, "biggest_winner": {"type": "string"}, "biggest_loser": {"type": "string"}, "cost_health": {"type": "string", "enum": ["green", "yellow", "red"]}, "cost_comment": {"type": "string"}, "lessons": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5}, "tomorrow_watch": {"type": "array", "items": {"type": "string"}}}, "required": ["one_liner", "biggest_winner", "biggest_loser", "cost_health", "cost_comment", "lessons", "tomorrow_watch"]}
# @validate_retries 2
#
# Last cell in the notebook. Produces a structured end-of-day journal
# entry from the P&L summary and cost ledger, the kind of thing
# you'd want to read first the next morning.

Trading day recap. Produce an honest, structured summary.

## P&L

{{ pnl_summary }}

## Costs

{{ cost_summary }}

## Today's orders

{{ submitted }}

## Rules for the journal

- `one_liner`: one sentence summarizing the day. Lead with the
  cost-adjusted P&L sign.
- `biggest_winner` / `biggest_loser`: name the ticker and amount;
  use "none" if no closed trades yet.
- `cost_health`: "green" when costs are < 10% of gross P&L,
  "yellow" at 10–40%, "red" at >40% or when P&L is negative.
- `cost_comment`: one sentence explaining the color.
- `lessons`: 1-5 short imperatives ("cut losers faster", "avoid
  low-confidence signals", etc.). If the sample size is too small to
  conclude anything, say so in one of the lessons.
- `tomorrow_watch`: specific things to look at tomorrow. Can be
  empty if there's nothing actionable yet.

Be terse. No filler.
