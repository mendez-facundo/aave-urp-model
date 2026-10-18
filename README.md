# User Risk Premium Model

**Stochastic Expected Credit Loss engine to calibrate the Aave v4 User Risk Premium, built on discrete-time Markov chains (PD) and extreme value theory (LGD).**

Actuarial thesis, Universidad de Buenos Aires (UBA).

## Overview

Aave v4 moves to a Hub & Spoke architecture in which a **User Risk Premium (URP)** prices each borrower's risk and funds the Liquidity Hub's reserves against bad debt. Today the risk parameters behind that pricing come from static, high-latency DAO governance votes: they penalise safe collateral in calm markets and undercapitalise the protocol under stress.

This repository implements an Expected Credit Loss (ECL) engine that replaces the static heuristic with a continuous, risk-based actuarial provision:

```text
ECL = EAD x PD x LGD   ->   User Risk Premium
```

| Component | Estimated with | Question it answers |
| --------- | -------------- | ------------------- |
| **EAD** | Debt of each wallet, valued at the Aave oracle price | How much is exposed? |
| **PD** | Discrete-time Markov chain over daily Health Factor states | How likely is a liquidation event within 30 days? |
| **LGD** | Extreme value theory on collateral log-losses (POT/GPD, Expected Shortfall) | How much is lost if the liquidation happens? |

Aave v3 on Ethereum is used as the observable analogue of the v4 Hub. IFRS 9 is a conceptual inspiration for the expected-loss framing only: the project implements neither stages nor SICR. Expected results are stated as testable hypotheses, not as conclusions.

## Methodology

### Probability of default: Markov chain on Health Factor states

Each wallet-day is mapped to one of seven states.

| State | Definition | Type |
| ----- | ---------- | ---- |
| **S1** | Repayment observed on-chain that closes the position (residual debt at or below the dust threshold) | Absorbing |
| **S2** | HF > 2.0 | Transient |
| **S3** | 1.5 < HF <= 2.0 | Transient |
| **S4** | 1.25 < HF <= 1.5 | Transient |
| **S5** | 1.10 <= HF <= 1.25 | Transient |
| **S6** | 1.0 < HF < 1.10 | Transient |
| **S7** | Liquidation event (technical default): HF <= 1 in the daily snapshot, or an on-chain `liquidationCall` on the wallet | Absorbing |

- S1 and S7 are absorbing by definition. **Right-censoring** applies only to wallets whose outcome is not observed: the window ends, or the wallet leaves the profile filter while it still has debt.
- Transitions are never counted across unobserved days (no forward-fill).
- The 30-day PD of each state is the S7 column of **P^30**.
- Uncertainty and robustness: absolute count matrices, wallet-level bootstrap confidence intervals, a time-homogeneity check, and sensitivity to the HF cut-offs and to the minimum-debt filter.
- The choice of the calibration window is under study (out-of-sample selection).

### Loss given default: EVT on collateral returns

- Daily log-losses of each collateral asset are modelled with **Peaks-Over-Threshold**: a Generalized Pareto Distribution fitted on a 730-day window, with threshold diagnostics, Q-Q plots and a bootstrap confidence interval.
- The **Expected Shortfall** (99%) is converted into a price shock as `E[1 - exp(-X) | X > VaR]`, the tail-averaged price decline, consistent with log-losses (Jensen).
- The LGD is **conditional on the liquidation event** and uses the real liquidation threshold (LT) and liquidation bonus (LB) of each reserve:

  ```text
  LGD_w = max(0, 1 - (HF_entry / LT_w) * sum_k s_k (1 - shock_k) / (1 + LB_w))
  ```

  where `s_k` is the share of collateral asset `k` in wallet `w`.
- Per-asset shocks are **comonotonic** (all collateral falls at once, each with its own ES). LST shocks equal the ETH shock.
- Notebook 04 also holds an exploratory menu of LGD variants (confidence level, multi-day horizon, stablecoin de-peg).

### Validation

1. In-sample backtest on the day-1 cohort: expected liquidations (sum of `n_i * PD_i`) against observed entries to S7.
2. Out-of-time backtest: the portfolio at `t` against liquidations observed in `(t, t+30]`.
3. Blind application on the full set of indebted wallets at block `t` from an archive RPC (Alchemy), classified by profile with the on-chain data.

Methodological decisions and their rationale are logged in [`docs/decisions.md`](docs/decisions.md).

## Scenarios and profiles

Each scenario is a valuation date `t`. Daily data use the first block at or after 00:00 UTC of each date.

| Scenario | Folder | Valuation date `t` | Block at `t` | Role |
| -------- | ------ | ------------------ | ------------ | ---- |
| **Baseline** | `baseline_mar2024` | 2024-03-15 | `19,436,645` | Benign collateral and funding regime |
| **Stress** | `stress_aug2024` | 2024-08-05 | `20,458,638` | Stressed collateral regime for the downturn LGD |

Wallets are segmented into two profiles, both with **stablecoin debt only**:

| Profile | Collateral composition |
| ------- | ---------------------- |
| **A** | At least 85% ETH or liquid staking tokens (LSTs) |
| **B** | At least 85% UNI + LINK |

## Data pipeline

All extractors live in `src/extraction/` and write under `data/01_raw/<scenario>/`. Each scenario window runs from `t-60` to `t+30` days, one snapshot per UTC day.

| Source | Extractor | Output |
| ------ | --------- | ------ |
| **The Graph**, Aave v3 Ethereum subgraph, queried with block time-travel (`block: {number: B}`) | `aave_graph_agent.py` | `thegraph_daily/` (positions per day), `thegraph_events/` (`liquidationCall` and `repay` events), `protocol/` (reserve parameters incl. LT, LB, LTV and eMode, plus `daily_blocks.csv`) |
| **Aave oracle** on chain, `AaveOracle.getAssetsPrices` at each daily block through JSON-RPC | `aave_oracle_agent.py` | `prices/aave_oracle_daily.csv` (stablecoins and LSTs at their oracle price) |
| **Binance** daily klines, 730 days | `binance_agent.py` | `prices/binance_daily.csv` (input of the EVT notebook) |

Balances are rebuilt from the scaled balances and the liquidity and borrow indices at each block, because the subgraph's `current*` fields are frozen at each user's last own event. Positions are valued at the Aave oracle price, which is the price the protocol itself uses to liquidate. Binance prices are used only for EVT.

`aave_rpc_agent.py` and `extract_active_wallets.py` are the earlier Alchemy snapshot tools. They are kept for reference and will be rewritten (Multicall3) for the blind application.

## Repository structure

Folders marked `(git-ignored)` are materialised locally and not versioned.

```text
.
├── data/                                  (git-ignored contents)
│   ├── 01_raw/
│   │   ├── baseline_mar2024/
│   │   │   ├── thegraph_daily/            # positions_<date>.csv.gz + metadata.json
│   │   │   ├── thegraph_events/           # liquidation_calls.csv, repays.csv
│   │   │   ├── protocol/                  # reserve_params_daily, emode_categories_daily, daily_blocks
│   │   │   └── prices/                    # aave_oracle_daily.csv, binance_daily.csv
│   │   └── stress_aug2024/                # same layout
│   ├── 02_interim/                        # notebook exports (*_not_for_prod.csv)
│   ├── 03_processed/                      # reserved for the production pipeline
│   └── sample/                            # small stratified dev sample (see data/sample/README.md)
├── docs/
│   ├── decisions.md                       # methodological decision log (Spanish)
│   └── findings_graph_extraction.md       # diagnosis of the original extraction error (Spanish)
├── notebooks/
│   ├── 01_eda_market_data.ipynb           # EDA of Binance prices
│   ├── 02_markov_calibration.ipynb        # Markov calibration (PD)
│   ├── 03_evt_gpd_fitting.ipynb           # EVT / GPD (LGD inputs)
│   ├── 04_ecl_urp_results.ipynb           # integrated ECL table, backtest, LGD variants
│   └── py/                                # jupytext py:percent pairs of the notebooks
├── scripts/
│   ├── check_raw_data.py                  # structural checks of the raw data
│   ├── make_dev_sample.py                 # builds data/sample/
│   └── raw_data_common.py                 # helpers shared by the two scripts above
├── src/
│   ├── extraction/                        # data agents (see Data pipeline)
│   ├── processing/                        # empty placeholders, by design (see Status)
│   └── utils/
│       ├── blocks.py                      # UTC date -> Ethereum block
│       ├── rpc.py                         # requests-based JSON-RPC client (web3 not required)
│       ├── protocol_params.py             # single source of reserve parameters (LT, LB, LTV, eMode)
│       ├── stats_compat.py                # fallback when scipy.stats cannot be imported
│       └── logger.py
├── tests/                                 # plain-assert tests, run directly with Python
├── .env.example
├── requirements.txt
└── README.md
```

## Reproducing the results

### Prerequisites

- **Python 3.11+**.
- A **The Graph** API key (`GRAPH_API_KEY`).
- An **archive-capable Ethereum JSON-RPC endpoint**, for example Alchemy (`ALCHEMY_RPC_URL`). Historical `eth_call` fails on a non-archive node.

### 1. Environment

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Secrets

Copy `.env.example` to `.env` (never commit it) and fill in both values. The agents load it through `python-dotenv`.

```env
GRAPH_API_KEY=<your The Graph API key>
ALCHEMY_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<your API key>
```

### 3. Extract the raw data

Run from the repository root. `--scenario` accepts `baseline`, `stress` or `all` (the default). The Graph and oracle agents are resumable: re-running the same command skips the days already on disk.

```bash
python src/extraction/aave_graph_agent.py  --scenario baseline
python src/extraction/aave_oracle_agent.py --scenario baseline
python src/extraction/binance_agent.py
```

### 4. Check the raw data

```bash
python scripts/check_raw_data.py
```

Verifies that dates and blocks agree across files, that every position has an oracle price and reserve parameters, that the universe changes from day to day, and that events fall inside the window.

### 5. Run the notebooks in order

```bash
jupyter lab
```

1. `01_eda_market_data.ipynb`
2. `02_markov_calibration.ipynb`
3. `03_evt_gpd_fitting.ipynb`
4. `04_ecl_urp_results.ipynb`

The notebooks are paired with `notebooks/py/*.py` (jupytext, `py:percent`). Edit the `.py` file and run `jupytext --sync notebooks/py/<file>.py` to update the notebook. Exports go to `data/02_interim/*_not_for_prod.csv`.

### Development on the sample

`python scripts/make_dev_sample.py` builds a small stratified sample in `data/sample/` from the raw data. To run notebook 02 on it, set `MARKOV_RAW_DATA_DIR=../data/sample`. Its exports are written under `data/sample/02_interim/`, so they cannot be confused with a full run.

### Tests

The tests use plain `assert` statements and run directly, without pytest:

```bash
python tests/test_protocol_params.py
python tests/test_stats_compat.py
python tests/test_aave_graph_agent.py
```

## Status and roadmap

**Done**

- Data pipeline re-extracted with block time-travel, Aave oracle prices and on-chain events (replaces the invalid first panel, see below).
- Notebook 02: Markov calibration on the daily snapshots, with count matrices, bootstrap intervals and sensitivities.
- Notebook 03: EVT/GPD fits, ES at several levels and horizons, liquidation break-even shocks.
- Notebook 04: integrated ECL table per scenario and profile, in-sample backtest, exploratory LGD variants.

**Pending**

- Selection of the calibration window (out-of-sample).
- Out-of-time backtest on `(t, t+30]`.
- Choice of the final LGD, possibly a point-in-time GARCH-EVT.
- Smoothing of profile B, whose sample is sparse.
- Blind application on Alchemy data (rewrite of the RPC snapshot with Multicall3).
- Production pipeline in `src/processing/`, built once the methodology is final (the module files are intentionally empty until then).
- URP calibration.

## Documentation

- [`docs/decisions.md`](docs/decisions.md): methodological decision log with the rationale of each choice, and the open questions (Spanish).
- [`docs/findings_graph_extraction.md`](docs/findings_graph_extraction.md): diagnosis of the original extraction error. Queries without block time-travel returned 2026 balances instead of the 2024 state, which invalidated the first panel (Spanish).
- [`data/sample/README.md`](data/sample/README.md): how the development sample is built.
