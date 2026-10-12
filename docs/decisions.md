# Registro de decisiones metodológicas

Cada entrada: fecha · decisión · justificación · origen. Las decisiones pendientes están al final.

## Definición de estados (Markov)

**2026-10-11 · S7 = evento de liquidación (incumplimiento técnico), no "default económico".**
La matriz estima la probabilidad de alcanzar el evento de liquidación, y la LGD determina si esa liquidación genera pérdida para el protocolo. Así la descomposición EAD × PD × LGD queda coherente. *Origen: correo del tutor (6/10).*

**2026-10-11 · S7 se detecta con HF ≤ 1 en el snapshot diario O con un `liquidationCall` de la wallet ese día.**
Las liquidaciones reales ocurren minutos después de cruzar HF = 1 y restauran el HF antes del snapshot siguiente. Con solo el HF diario, se perderían eventos y la PD quedaría subestimada.

**2026-10-11 · S1 = repago observado on-chain** (un evento `repay` que deja la deuda ≈ 0, con tolerancia de polvo).
Reemplaza la inferencia "la wallet desapareció del panel". Supuesto: un re-endeudamiento posterior dentro de la ventana es una posición nueva. *Origen: correo del tutor.*

**2026-10-11 · S1 y S7 son estados absorbentes por definición; no son censura.**
La censura por la derecha es la de las wallets cuyo desenlace no se observa: siguen transitorias al cierre de la ventana, o salen del filtro de perfil con deuda > 0. En ese segundo caso la trayectoria se corta en la última observación, sin S1 y sin forward-fill. Hay que chequear si la censura es informativa (HF en la salida). *Origen: correo del tutor.*

**2026-10-11 · No se cuentan transiciones a través de días no observados.**
El forward-fill en los gaps fabricaba transiciones de permanencia.

**2026-10-12 · Un `liquidationCall` del día d lleva a S7 en el snapshot de d+1 (calibración causal).**
Los snapshots son de las 00:00 UTC, así que un evento fechado d ocurre entre el snapshot d y el d+1: se asocia a la transición d → d+1 (`LIQUIDATION_DAY_OFFSET = 0`) y la wallet es S7 en d+1, donde el HF ya está restaurado. Los repagos se leen igual (S1 se define por la deuda del snapshot siguiente). Consecuencia: la calibración [t−30, t] solo usa información hasta el snapshot de t; las liquidaciones del propio día t (en stress, el crash del 2024-08-05) caen en el período out-of-time. En `hf_at_s7_entry`, una entrada por `liquidationCall` lleva el HF del snapshot anterior (> 1) y el LGD la trata con HF_entry = 1 usando `s7_trigger`. *Origen: revisión del coordinador (12/10); alternativa "mismo día" disponible con `LIQUIDATION_DAY_OFFSET = 1`.*

**2026-10-12 · Un repago que cierra la posición (S1) cierra un *spell*; un nuevo endeudamiento dentro de la ventana abre un spell nuevo.**
El spell nuevo es una trayectoria nueva y cuenta como ingreso tardío desde su día de reingreso (`spell_id` interno). Después de S7 la wallet no se vuelve a seguir dentro de la ventana (regla del proyecto): sus observaciones posteriores se ignoran para N y se informa cuántas son. El bootstrap sigue remuestreando wallets, así que los spells de una wallet quedan juntos. *Origen: revisión del coordinador (12/10).*

**2026-10-12 · Umbral de polvo para S1: `DUST_DEBT_USD` = USD 1.**
S1 = un evento `repay` tras el cual la deuda total del usuario es ≤ USD 1 en el snapshot siguiente, o el usuario desaparece de ese snapshot habiendo tenido un `repay` ese día. Una deuda residual entre USD 1 y el mínimo del panel no es S1: la wallet sale del filtro (censura por salida del filtro). *Origen: confirmación del usuario (12/10).*

## Datos

**2026-10-11 · El panel de Markov se construye con snapshots diarios con time-travel de The Graph** (`block: {number: B}`), uno por día de `[t−30, t+30]`.
El extractor anterior devolvía posiciones de 2026 (ver `docs/findings_graph_extraction.md`). El tramo `(t, t+30]` es para la validación out-of-time.

**2026-10-12 · Los wrappers de stablecoins con rendimiento (sDAI, sUSDe) cuentan como colateral estable.**
Entran en la regla del resto de los perfiles (A: estables o WBTC; B: estables, WETH, LST o WBTC) y en `stable_usd`, valuados al precio del oráculo. No son stablecoins de deuda: la regla de deuda 100% en stablecoins USD no los incluye. *Origen: revisión del coordinador (12/10).*

**2026-10-12 · Deuda mínima de USD 100 para entrar al panel de perfiles.**
Un usuario-día entra al panel solo si su deuda total es ≥ USD 100: las wallets con HF ≤ 1 persistente eran casi todas polvo que los liquidadores ignoran (ninguna superaba USD 941 de deuda en el diagnóstico de la Fase 1). Como sensibilidad, el 02 (§17.5) reporta N y PD a 30 días por estado con un mínimo de USD 1.000, usando las mismas funciones. *Origen: confirmación del usuario (12/10), a partir del diagnóstico de la Fase 1.*

**2026-10-11 · Los precios para el HF salen del oráculo de Aave** (`AaveOracle.getAssetsPrices`, un `eth_call` por bloque diario).
Es el precio que usa el protocolo para liquidar, y resuelve la valuación de los LST (Binance no lista wstETH, rETH ni cbETH). Binance se mantiene para EVT.

**2026-10-11 · LT, bono de liquidación, LTV y eMode se toman del protocolo en cada bloque.**
Se guardan en `data/01_raw/<esc>/protocol/` y se leen con `src/utils/protocol_params.py`, que es la única fuente. Los LT fijos de `config.py` quedan deprecados.

## Severidad (EVT y LGD)

**2026-10-11 · EVT incondicional (POT/GPD, 730 días, ES 99%) por ahora.** GARCH-EVT (point-in-time) queda pendiente de consultar con el tutor por la complejidad matemática que suma al documento.

**2026-10-11 · LGD condicional al evento de liquidación, incluyendo el bono de liquidación:**
`LGD_w = max(0, 1 − (HF_entry / LT_w) · Σ_k s_k (1 − shock_k) / (1 + LB_w))`.
Justificación: una liquidación mejora C/D si C/D > 1 + LB y lo empeora si C/D < 1 + LB. El umbral no depende de si la liquidación es parcial o total, así que no hace falta modelar las liquidaciones parciales.

**2026-10-11 · PD a 30 días × LGD de 1 día es coherente.** La PD ya incluye la trayectoria de precios hasta el disparo; la LGD mide lo que pasa entre el disparo y la liquidación. Un shock a 30 días en la LGD contaría dos veces la trayectoria.

**2026-10-11 · Shocks por activo, ponderados por la composición de cada wallet, con dependencia comonótona** (todos los activos caen a la vez con su propio ES). Es un supuesto conservador, explícito, y reemplaza a las cópulas.

**2026-10-11 · Shock de los LST = shock de ETH.** El precio de un LST en el oráculo es ETH/USD × la tasa de canje del wrapper, que crece de forma suave. Opcional: verificarlo con la serie del oráculo.

**2026-10-11 · Paridad de las stablecoins se mantiene.** La pérdida se mide en unidades de la stablecoin de la deuda, que es la moneda del Deficit Buffer, así que la paridad de la deuda no influye. Se hace una sensibilidad sobre la parte estable del colateral (referencia: USDC ≈ 0,88 en marzo de 2023).

**2026-10-11 · Datos intradiarios descartados para EVT.** Dan el ES de un horizonte horario, no un mejor ES diario.

## Validación

**2026-10-11 · En dos pasos:** (1) backtest in-sample con la cohorte del día 1, sin ingresos tardíos, que se envía al tutor; (2) backtest out-of-time: la cartera en t contra las liquidaciones observadas en (t, t+30].

**2026-10-11 · Alchemy es la aplicación ciega:** todas las wallets con deuda en el bloque t, clasificadas por perfil con los datos de Alchemy. La separación es temporal (calibrar con el pasado, evaluar con el futuro), no por wallets.

## Terminología
- NIIF 9 como **inspiración conceptual** (pérdida esperada), no como implementación: sin stages ni SICR. *Origen: correo del tutor.*
- Las "conclusiones esperadas" pasan a ser **hipótesis contrastables**.

## Pendientes (para consultar con el tutor)
- GARCH-EVT (PiT) para que la severidad diferencie escenarios, y la profundidad matemática esperada del documento.
- Si la LGD base sale ≈ 0: horizonte de liquidación de h días (medido con `liquidationCalls`) y/o ES 99,5% como sensibilidad.
- ES de stress < ES de baseline por el solapamiento de las ventanas de 730 días: ¿alcanza con documentarlo?
