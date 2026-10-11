# Diagnóstico: el panel de The Graph no reflejaba el estado de 2024

Fecha: 2026-10-11. Afecta a `data/01_raw/*/thegraph_hf_30d.json` (extraídos el 2026-09-15) y, por lo tanto, a toda la calibración de Markov de `02_markov_calibration.ipynb` (commit `08edbdc`).

## Causa
`USERS_QUERY` de `src/extraction/aave_graph_agent.py` (versión anterior):
1. **No usa time-travel** (no tiene argumento `block:`). El universo `users(where: {borrowedReservesCount_gt: 0})` son los prestatarios **a la fecha de extracción**, y por eso ambas ventanas tienen exactamente 49.299 usuarios.
2. Los campos `currentATokenBalance` / `currentVariableDebt` de cada user-reserve son los **actuales (2026)**.
3. Los historiales (`aTokenBalanceHistory`, `vTokenBalanceHistory`) se filtran con `timestamp_gte: start`, así que **falta el saldo de apertura**.
4. En el 02, la regla "si no hay historial en la ventana, el saldo es constante e igual a `current*`" proyectaba saldos de 2026 sobre 2024, y las posiciones que hoy están en 0 se descartaban.

## Evidencia (baseline, 8.566 wallets del panel A/B)
| Métrica | Valor |
|---|---|
| Wallets sin ningún evento en la ventana (toda la posición sale de `current*`) | 7.951 (93%) |
| Wallets cuyas reserves tienen todas saldos posteriores a la ventana | 7.544 (88%) |
| Reserves con `lastUpdateTimestamp` posterior al fin de la ventana | 166.043 de 170.953 |
| Wallets en S7 el día 1 | 203 (63 con colateral sin saldo de apertura; 51 con LST) |
| "Ingresos tardíos" con deuda sin saldo de apertura (no son nuevos prestatarios) | 449 de 569 (79%) |

## Otros problemas detectados
- **LT fijos en `config.py`** vs. los del subgraph (valores actuales, a verificar para 2024): UNI 0,74 y LINK 0,71 (config: 0,65); WBTC 0,78 (config: 0,75); WETH 0,83; wstETH 0,81; rETH/cbETH 0,79.
- **Los LST se valuaban a ETHUSDT.** wstETH vale ≈ 1,15–1,17 ETH, y el HF quedaba subestimado. Binance no lista wstETH, rETH ni cbETH.
- **El snapshot de Alchemy** (`aave_rpc_agent.py`) usa una lista fija de 10 activos, sin LINK. Un colateral fuera de la lista es invisible, y la wallet puede parecer "pura" cuando no lo es.

## Consecuencias
- Los conteos de wallets y transiciones enviados al tutor (7.978 / 588 / 8.087 / 570 wallets; 232.232 / 17.008 / 231.189 / 15.942 transiciones) **van a cambiar**.
- La metodología y el código de 02 y 03 se mantienen; cambian los datos.
- El notebook 03 (EVT con Binance) no se ve afectado.

## Resultados preliminares (con los datos inválidos, solo como referencia)
- Backtest in-sample de la cohorte del día 1 (liquidaciones esperadas `Σ n_i·PD_i` vs entradas a S7): baseline A 21,1 vs 23 · B 5,9 vs 7 · stress A 354,6 vs 125 · B 56,7 vs 26. En stress, la cadena no es homogénea (el crash del 5/8 cae al final de la ventana).
- Shock necesario para que haya pérdida, partiendo de HF = 1, sin bono: 1 − LT. Con bono: 1 − LT·(1 + LB). Ejemplos: ETH 17% → 12,9%; UNI 26% → 18,6%. ES99 diario: 9–16%.
