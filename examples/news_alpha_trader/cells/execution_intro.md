# Place orders, reconcile, report

Signal extraction and risk checks land cached artifacts upstream.
Below is the execution side: send the trade plan to the broker,
reconcile what filled, attribute trading costs, and write the
end-of-day P&L into a structured report.

Editing any analysis cell above invalidates the downstream
execution cells through provenance, so a re-run never sends stale
orders. The very last cell is a prompt cell that summarizes the
day's session for human review.
