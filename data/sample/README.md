# Development sample

Small, reproducible subset of `data/01_raw/` for developing and testing the pipeline. All 91 days of each scenario window are kept; only the users are subsampled. Git-ignored (only `.gitkeep` is tracked). Full runs use the raw data.

## How it was built

Script: `scripts/make_dev_sample.py` (shared helpers in `scripts/raw_data_common.py`). Seed: `20240315` (`numpy.random.default_rng`); target: 300 users per scenario. Re-running with the same raw data and seed gives the same sample.

Strata, in this order (a user is counted once; candidates are sorted before drawing):

1. `liquidated`: users with >= 1 liquidation in the window that appear in the positions, capped at 100. If there are more, users that are approximate A/B at some day are drawn first.
2. `repay_to_zero`: ~80 users, not liquidated, with a repay on day D, debt >= $100 at the D snapshot and < $1 (or absent) at the D+1 snapshot. Same A/B-first rule.
3. `profile_ab`: the rest, random among approximate A/B users (about one third from profile B when available, because B is much smaller than A).
4. `borrower`: random borrowers, only if the A/B pool runs out.

Approximate A/B (diagnostic, not the thesis methodology): at least once in the window, >= 85% of the collateral (oracle USD, reserves enabled as collateral) in WETH/wstETH/rETH/cbETH/weETH/osETH/ETHx (A) or UNI+LINK (B), and stablecoin-only debt. HF uses the reserve LT, or the eMode category LT for eMode users and category assets.

Positions and events are filtered by user with a byte-exact line copy (same columns, text and CRLF line endings as the raw files). `protocol/` and `prices/` are copied unchanged. `thegraph_*/metadata.json` are the raw ones with the sample counts (`source_*` keep the originals). `<folder>/sample_users.csv` lists the users with their stratum and approximate profile flags.

## Counts

| scenario | users | liquidated | repay_to_zero | profile_ab | borrower | position rows | liquidation events | repay events | users/day (min-max) | ever A | ever B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_mar2024 | 300 | 100 | 80 | 120 | 0 | 45578 | 151 | 614 | 179-211 | 242 | 62 |
| stress_aug2024 | 300 | 100 | 80 | 120 | 0 | 48887 | 272 | 837 | 193-223 | 243 | 58 |

Candidate pools in the full raw data:

| scenario | liquidated users in positions | repay-to-zero users | ever A (not B) | ever B | all borrowers |
|---|---|---|---|---|---|
| baseline_mar2024 | 350 | 2322 | 4995 | 498 | 10810 |
| stress_aug2024 | 1460 | 2679 | 7345 | 599 | 15958 |

## Caveats

- The sample is stratified, not representative: liquidated and repaid wallets are over-represented. Do not estimate frequencies (PD, state shares) from it.
- Entries and exits of the sampled users are real; the daily universe of the sample is not the daily universe of the market.
- `scripts/check_raw_data.py --raw-dir data/sample` runs the structural checks on the sample.
