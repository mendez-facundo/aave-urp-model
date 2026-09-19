# User Risk Premium Model

**Stochastic IFRS 9 calibration of the User Risk Premium and Liquidity Hub provisioning for Aave v4, using discrete-time Markov chains and extreme value theory.**

## Abstract / Architecture

Aave v4 transitions toward a Hub & Spoke architecture where a User Risk Premium (URP) prices individual borrower risk to fund the Liquidity Hub's insolvency reserves. While this mechanism is designed to absorb bad debt, the underlying risk parameters currently depend on static, high-latency DAO governance votes. This rigidity reproduces structural flaws: it penalizes safe collateral during stability and undercapitalizes the protocol during severe market stress, forcing reliance on the Safety Module and risking procyclical death spirals.

This repository presents a stochastic, IFRS 9-compliant Expected Credit Loss (ECL) engine designed to autonomously calibrate the URP. By decoupling Probability of Default (PD) via discrete-time Markov chains and Loss Given Default (LGD) via Extreme Value Theory, this framework transforms the URP from a static heuristic into a continuous, risk-based actuarial provision.

This repository implements a two-pillar quantitative architecture that replaces static premia with scenario-contingent risk measures:

1. **Discrete-Time Markov Chains (DTMC).** Health-factor and repayment paths are mapped onto a finite state space. Competing risks are isolated so that **default probabilities** are not confounded with **voluntary (competitive) prepayment**. The resulting transition matrix yields IFRS 9 stage-consistent PDs for the Liquidity Hub book.
2. **Extreme Value Theory (EVT).** Collateral returns are modelled with a **Peaks-Over-Threshold (POT)** approach and a **Generalized Pareto Distribution (GPD)**. The fitted tail supplies a **downturn LGD**—a stressed severity consistent with IFRS 9 lifetime-loss and downturn-collateral guidance.

Empirical identification currently uses **Aave v3 on Ethereum** as the observable analogue of the v4 Hub. Two cross-sections pin the design:


| Scenario     | Valuation date | Archive block | Role                                        |
| ------------ | -------------- | ------------- | ------------------------------------------- |
| **Baseline** | 15 March 2024  | `19,433,100`  | Benign collateral and funding regime        |
| **Stress**   | 5 August 2024  | `20,457,180`  | Stressed collateral regime for downturn LGD |


Market tails (BTC, ETH, UNI) are extracted from Binance daily candles. Protocol exposures are reconstructed from archive `eth_call` snapshots of `AaveProtocolDataProvider.getUserReserveData`. Downstream notebooks calibrate the Markov and GPD layers and translate PD × EAD × downturn LGD into a **dynamic URP** and Hub ECL.

**Current status.** The **data-extraction pipeline** is operational. Processing modules (`src/processing/`) and result notebooks for Markov, EVT, and ECL are scaffolded and will be populated as calibration proceeds.

## Directory Structure

```text
aave-urp-model/
├── data/
│   ├── 01_raw/
│   │   ├── baseline_mar2024/      # OHLCV CSVs, wallet universe, RPC snapshot
│   │   └── stress_aug2024/
│   ├── 02_interim/
│   └── 03_processed/
├── notebooks/
│   ├── 01_eda_market_data.ipynb
│   ├── 01b_eda_aave_snapshots.ipynb
│   ├── 02_markov_calibration.ipynb
│   ├── 03_evt_gpd_fitting.ipynb
│   └── 04_ecl_urp_results.ipynb
├── src/
│   ├── extraction/
│   │   ├── binance_agent.py           # Daily BTC/ETH/UNI klines
│   │   ├── extract_active_wallets.py  # Wallet universe (The Graph, time-travel)
│   │   ├── aave_rpc_agent.py          # Archive RPC collateral/debt snapshot
│   │   └── aave_graph_agent.py        # 30-day GraphQL position/liquidation history
│   ├── processing/
│   │   ├── markov_model.py
│   │   ├── evt_model.py
│   │   └── ecl_calculator.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── requirements.txt
└── README.md
```

Raw extracts are git-ignored (`data/01_raw/*`). Re-run the agents below to materialise local copies.

## Setup & Installation



### Prerequisites

- **Python 3.11+** (the reference environment uses 3.14)
- An **archive-capable Ethereum RPC** (Alchemy or equivalent). Historical `eth_call` at blocks `19,433,100` and `20,457,180` will fail on a non-archive node.
- A **The Graph** API key, required to build the wallet universe before the RPC snapshot.



### 1. Clone and create a virtual environment

**Windows (PowerShell)**

```powershell
git clone <repository-url> aave-urp-model
cd aave-urp-model
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
git clone <repository-url> aave-urp-model
cd aave-urp-model
python3 -m venv venv
source venv/bin/activate
```



### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```



### 3. Configure environment variables

Create a `.env` file in the project root (do not commit it). The extraction agents load it via `python-dotenv`.

```env
# Archive Ethereum JSON-RPC (Alchemy recommended)
ALCHEMY_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/<YOUR_API_KEY>

# The Graph decentralized network gateway
GRAPH_API_KEY=<YOUR_GRAPH_API_KEY>
```

`ALCHEMY_RPC_URL` is mandatory for `aave_rpc_agent.py`. `GRAPH_API_KEY` is mandatory for `extract_active_wallets.py` (and for the optional GraphQL history agent).

## Usage: Data Extraction Pipeline

All commands are executed from the **repository root**. Each agent writes into `data/01_raw/baseline_mar2024/` and `data/01_raw/stress_aug2024/`.

### Step 1 — Market prices (Binance)

Downloads approximately two years of daily OHLCV for `BTCUSDT`, `ETHUSDT`, and `UNIUSDT`, ending on **2024-03-15** (baseline) and **2024-08-05** (stress). No API key is required.

```bash
python src/extraction/binance_agent.py
```

**Outputs**

- `data/01_raw/baseline_mar2024/binance_prices_{btc,eth,uni}.csv`
- `data/01_raw/stress_aug2024/binance_prices_{btc,eth,uni}.csv`



### Step 2 — Active wallet universe (The Graph)

Time-travels the Aave v3 Ethereum subgraph to the two archive blocks and writes a unique address list. This file is the input universe for the RPC snapshot.

```bash
python src/extraction/extract_active_wallets.py
```

**Outputs**

- `data/01_raw/baseline_mar2024/all_active_wallets.json`
- `data/01_raw/stress_aug2024/all_active_wallets.json`



### Step 3 — Cross-sectional Aave snapshot (archive RPC)

Queries `AaveProtocolDataProvider.getUserReserveData` at the pinned blocks for every wallet × target reserve (WETH, WBTC, UNI, USDC, USDT, DAI, FRAX, wstETH, rETH, cbETH). Empty positions are omitted. Amounts are stored as **uint256 wei strings**.

```bash
python src/extraction/aave_rpc_agent.py
```

**Outputs**

- `data/01_raw/baseline_mar2024/alchemy_snapshot.json` — block `19,433,100`
- `data/01_raw/stress_aug2024/alchemy_snapshot.json` — block `20,457,180`

The job issues on the order of `n_wallets × n_assets` archive calls, with bounded concurrency and exponential backoff on HTTP 429. Runtime depends on Alchemy compute-unit capacity.

### Step 4 — 30-day GraphQL history

Position and liquidation paths for Health Factor reconstruction (Markov inputs):

```bash
python src/extraction/aave_graph_agent.py
```

**Outputs:** `thegraph_hf_30d.json` under each scenario directory.

### Exploratory notebooks

After extraction:

1. `notebooks/01_eda_market_data.ipynb` — Binance series integrity
2. `notebooks/01b_eda_aave_snapshots.ipynb` — RPC cross-sections
3. `notebooks/02_markov_calibration.ipynb` — GraphQL payloads / DTMC preparation

Launch Jupyter from the activated environment:

```bash
jupyter lab
```

