# Read before running

This notebook is the first thing you see when you open the project. It does
nothing computationally, the real work starts in `helpers`.

## Key facts

1. **Real money mode requires two opt-ins.** This notebook can place real
   orders on your Alpaca account when both `Config.MODE == "live"` and
   `Config.I_UNDERSTAND_THIS_IS_REAL_MONEY == True`. Both live in
   `helpers.py` and default to `"paper"` / `False`. You must edit both on
   purpose to trade live.

2. **Every LLM call costs real money.** Every market-data call counts
   against your Alpaca rate limit (200/min free tier). The `costs` cell
   tracks both, check it before you leave the notebook running unattended.

3. **Paper fills aren't realistic.** Paper fills on Alpaca are instantaneous
   and slippage-free, which is not how real markets work. We apply a modeled
   slippage cost even in paper mode so the cost-adjusted P&L doesn't lie
   when you eventually flip to live.

4. **LLM outputs are non-deterministic.** A hostile news headline could in
   principle influence a signal. We enforce a ticker whitelist
   (`Config.TICKER_WHITELIST`) at the order layer, the LLM cannot place
   orders in tickers outside that list regardless of what the signal says.

5. **Nothing here is financial advice.** The strategy is a demo. Backtest on
   historical data (see `pnl_report`) before trusting it with capital.
