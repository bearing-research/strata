# @name signals_batch
# @temperature 0.0
# @max_tokens 4096
# @system You are a quantitative-trading analyst extracting market-impact signals from news headlines. Return a row for every input article: the risk filter downstream drops weak signals, so do not pre-filter here.
# @output_schema {"type": "object", "properties": {"signals": {"type": "array", "items": {"type": "object", "properties": {"article_id": {"type": "integer"}, "ticker": {"type": "string"}, "sentiment": {"type": "number", "minimum": -1, "maximum": 1}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "theme": {"type": "string", "enum": ["earnings", "guidance", "product", "regulatory", "macro", "m_and_a", "legal", "people", "other"]}, "reasoning": {"type": "string"}}, "required": ["article_id", "ticker", "sentiment", "confidence", "theme", "reasoning"]}}}, "required": ["signals"]}
# @validate_retries 3
#
# Extracts structured sentiment for each article in the unprocessed
# batch. The schema enums are deliberately narrow: a tighter enum
# means cleaner aggregation SQL downstream. The validate-and-retry
# loop kicks in when the provider falls back to json_object mode
# (Anthropic, Gemini, Mistral) and the response doesn't quite match.
#
# Prompt-design note: earlier prompts told the model to "err toward
# low confidence on ambiguous headlines." Models read that as "omit
# the row" and returned empty arrays on unglamorous news batches.
# The current phrasing is explicit: return one row per article,
# encode uncertainty in the confidence score, don't drop articles.
# Config.MIN_CONFIDENCE + MIN_ABS_SENTIMENT at the risk layer are
# where the actual trading threshold lives.

Return exactly one signal per input article; do NOT omit rows. Encode
uncertainty in the `confidence` field. The downstream risk filter
drops low-confidence rows before any trade is placed, so your job
here is just to characterize every headline we fetched.

For each article, emit:

- `article_id` and `ticker`: copied verbatim from the input row.
- `sentiment` in [-1, 1]: -1 = clearly bearish for this ticker,
  +1 = clearly bullish, 0 = neutral / not directionally relevant.
- `confidence` in [0, 1]:
  - > 0.8 for clear-cut material news (earnings beat, explicit
    guidance, confirmed M&A, major product launch, regulatory
    action with a named target).
  - 0.5–0.8 for solid analyst coverage, sector news with clear
    ticker impact, leadership moves with stated effect.
  - 0.2–0.5 for rumors, side mentions, ambiguous macro.
  - < 0.2 for headlines that aren't really about this ticker
    (cross-references, generic market commentary). Still return
    the row; use 0 sentiment and explain in `reasoning`.
- `theme`: pick the single best enum value.
- `reasoning`: one short sentence explaining the score.

Do NOT infer signals for tickers other than the one on each row.

Articles:

{{ unprocessed }}
