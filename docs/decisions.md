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

**2026-10-17 · La longitud de la ventana de calibración se elige por validación fuera de muestra: W = 30.**
Reemplaza a la prueba formal de homogeneidad (G² de Anderson-Goodman), que se rechaza siempre con muestras grandes y dependencia dentro de cada wallet, y no guía ninguna decisión. La comparación por subventanas que pidió el tutor se mantiene como diagnóstico, expresada en PD30 por estado (mitades con IC bootstrap y ventanas móviles de 10 días).
- **Criterio:**
  - Para cada fecha s ∈ [t−30, t−1] y cada W ∈ {5, 7, 10, 14, 21, 30}, la cadena se reconstruye con las mismas reglas sobre el panel truncado a [s−W, s+1].
  - P̂_W se estima con las transiciones de origen en [s−W, s−1], con un pseudoconteo de Jeffreys de 0,5 por celda de cada fila transitoria.
  - Se evalúa la log-verosimilitud de las transiciones de s → s+1 (log score; el Brier, como métrica secundaria). Solo usa información hasta t.
  - La comparación es pareada, sobre las transiciones de evaluación comunes a todas las W.
  - Error estándar por bootstrap de wallets (B = 500).
  - **Regla de 1 EE:** la W más larga cuyo score queda dentro de un EE pareado del mejor. Favorece la estabilidad y no castiga al Perfil B.
  - **Una sola W para ambos perfiles y ambos escenarios** (score total).
- **Resultado (corrida completa, 207.320 transiciones de evaluación):**
  - El mejor log score y el mejor Brier los da W = 21.
  - W = 30 queda a 0,00006 del mejor, con un EE pareado de 0,00012, así que **W* = 30**, que coincide con la ventana de producción.
  - W = 5 y W = 7 predicen claramente peor (unos 17 EE).
  - Por segmento, estrés A, estrés B y baseline B prefieren 30; baseline A, 21.
- **Consecuencia:** `CALIBRATION_DAYS` sigue en 30. La ventana, elegida al principio de forma arbitraria, queda respaldada empíricamente.
- **Advertencia:** el criterio predice a un día. La homogeneidad de la proyección a 30 días la evalúa el backtest out-of-time.
- *Origen: propuesta del usuario (17/10), en respuesta al punto 2 del correo del tutor del 5/10; queda a confirmar con el tutor.*

## Datos

**2026-10-11 · El panel de Markov se construye con snapshots diarios con time-travel de The Graph** (`block: {number: B}`), uno por día de `[t−30, t+30]`.
El extractor anterior devolvía posiciones de 2026 (ver `docs/findings_graph_extraction.md`). El tramo `(t, t+30]` es para la validación out-of-time.

**2026-10-17 · La ventana diaria se extiende a [t−60, t+30] (91 snapshots) en los mismos archivos raw.**
Los 30 días previos a t−30 sirven solo para elegir la longitud de la ventana de calibración con validación fuera de muestra (cada fecha s ∈ [t−30, t−1] necesita hasta 30 días previos). El modelo se ajusta solo con las transiciones de la ventana elegida. Los datos raw son append-only: la extensión agrega días y filas después de un backup (`_backup_2026-10-17/`) y no altera las filas existentes; los eventos se combinan por id en lugar de sobrescribirse. *Origen: decisión del usuario (17/10).*

**2026-10-12 · Los wrappers de stablecoins con rendimiento (sDAI, sUSDe) cuentan como colateral estable.**
Entran en la regla del resto de los perfiles (A: estables o WBTC; B: estables, WETH, LST o WBTC) y en `stable_usd`, valuados al precio del oráculo. No son stablecoins de deuda: la regla de deuda 100% en stablecoins USD no los incluye. *Origen: revisión del coordinador (12/10).*

**2026-10-12 · Deuda mínima de USD 100 para entrar al panel de perfiles.**
Un usuario-día entra al panel solo si su deuda total es ≥ USD 100: las wallets con HF ≤ 1 persistente eran casi todas polvo que los liquidadores ignoran (ninguna superaba USD 941 de deuda en el diagnóstico de la Fase 1). Como sensibilidad, el 02 (§14) reporta N y PD a 30 días por estado con un mínimo de USD 1.000, usando las mismas funciones. *Origen: confirmación del usuario (12/10), a partir del diagnóstico de la Fase 1.*

**2026-10-11 · Los precios para el HF salen del oráculo de Aave** (`AaveOracle.getAssetsPrices`, un `eth_call` por bloque diario).
Es el precio que usa el protocolo para liquidar, y resuelve la valuación de los LST (Binance no lista wstETH, rETH ni cbETH). Binance se mantiene para EVT.

**2026-10-11 · LT, bono de liquidación, LTV y eMode se toman del protocolo en cada bloque.**
Se guardan en `data/01_raw/<esc>/protocol/` y se leen con `src/utils/protocol_params.py`, que es la única fuente. Los LT fijos de `config.py` quedan deprecados.

## Severidad (EVT y LGD)

**2026-10-11 · EVT incondicional (POT/GPD, 730 días, ES 99%) por ahora.** GARCH-EVT (point-in-time) queda pendiente de consultar con el tutor por la complejidad matemática que suma al documento.

**2026-10-11 · LGD condicional al evento de liquidación, incluyendo el bono de liquidación:**
`LGD_w = max(0, 1 − (HF_entry / LT_w) · Σ_k s_k (1 − shock_k) / (1 + LB_w))`.
Justificación: una liquidación mejora C/D si C/D > 1 + LB y lo empeora si C/D < 1 + LB. El umbral no depende de si la liquidación es parcial o total, así que no hace falta modelar las liquidaciones parciales.

**Shock de precio = E[1 − e^(−X) | X > VaR], no 1 − e^(−ES)** (decidido al construir el notebook 03; registrado el 2026-10-17).
El ES se estima en log-pérdidas X = −r, pero el colateral se estresa en precio. Como 1 − e^(−x) es cóncava, por la desigualdad de Jensen E[1 − e^(−X) | cola] ≤ 1 − e^(−ES): transformar el ES sobreestima levemente la caída esperada del precio. Se usa la caída de precio promediada sobre la cola (`es_price_shock`), que es la caída esperada exacta condicionada a la cola. Para ETH al 99% a 1 día: 13,88% contra 13,99% en baseline y 12,05% contra 12,12% en stress. La diferencia es chica, pero la LGD está cerca del break-even de WETH (12,85%). La forma P·(1 − ES) daría 15,1% y queda descartada. *Origen: decisión del usuario; responde al punto 3 del correo del tutor del 5/10.*

**2026-10-11 · PD a 30 días × LGD de 1 día es coherente.** La PD ya incluye la trayectoria de precios hasta el disparo; la LGD mide lo que pasa entre el disparo y la liquidación. Un shock a 30 días en la LGD contaría dos veces la trayectoria.

**2026-10-11 · Shocks por activo, ponderados por la composición de cada wallet, con dependencia comonótona** (todos los activos caen a la vez con su propio ES). Es un supuesto conservador, explícito, y reemplaza a las cópulas.

**2026-10-11 · Shock de los LST = shock de ETH.** El precio de un LST en el oráculo es ETH/USD × la tasa de canje del wrapper, que crece de forma suave. Opcional: verificarlo con la serie del oráculo.

**2026-10-11 · Paridad de las stablecoins se mantiene.** La pérdida se mide en unidades de la stablecoin de la deuda, que es la moneda del Deficit Buffer, así que la paridad de la deuda no influye. Se hace una sensibilidad sobre la parte estable del colateral (referencia: USDC ≈ 0,88 en marzo de 2023).

**2026-10-17 · Precisión sobre la paridad y sensibilidad del lado de la deuda.**
- **Valuación en el HF:** en el HF (notebook 02), las estables y los LST se valúan a su precio del oráculo de Aave. Las estables quedaron en 0,9988–1,0024 en baseline y 0,9992–1,0009 en stress. Los LST siguen la tasa de canje del wrapper (wstETH/WETH 1,157–1,178).
- **Supuesto de paridad:** se mantiene solo en la LGD.
- **Canal de la deuda:** la paridad de la deuda influye por su movimiento relativo u entre el disparo y la liquidación (la retención en unidades de deuda pasa a ser Σ s_k(1 − shock_k)/(1 + u)).
  - Un movimiento al alza actúa como un shock extra; uno a la baja (el caso histórico, 0,88) reduce la pérdida.
  - Variantes `S_depeg_debt_*` del 04 (u = +0,5 / +1 / +2 / −12%) y u* de break-even por activo al ES99 de 1 día. WETH da −1,19% en baseline y +0,92% en stress; LINK, +15,9% y +18,4%.
- **Sensibilidad del colateral:** es inmaterial, porque las estables son el 0,03–0,16% del colateral de las carteras.
- *Origen: pedido del tutor (5/10, punto 4).*

**2026-10-11 · Datos intradiarios descartados para EVT.** Dan el ES de un horizonte horario, no un mejor ES diario.

## Validación

**2026-10-11 · En dos pasos:** (1) backtest in-sample con la cohorte del día 1, sin ingresos tardíos, que se envía al tutor; (2) backtest out-of-time: la cartera en t contra las liquidaciones observadas en (t, t+30].

**2026-10-11 · Alchemy es la aplicación ciega:** todas las wallets con deuda en el bloque t, clasificadas por perfil con los datos de Alchemy. La separación es temporal (calibrar con el pasado, evaluar con el futuro), no por wallets.

## Terminología
- NIIF 9 como **inspiración conceptual** (pérdida esperada), no como implementación: sin stages ni SICR. *Origen: correo del tutor.*
- Las "conclusiones esperadas" pasan a ser **hipótesis contrastables**.

## Pendientes (para consultar con el tutor)
- Confirmar la selección de la ventana de calibración fuera de muestra (W = 30) como reemplazo de la prueba formal de homogeneidad.
- Suavizado del Perfil B (sin transiciones a S7 en baseline): credibilidad tipo Bühlmann o prior bayesiano hacia la matriz conjunta A+B.
- GARCH-EVT (PiT) para que la severidad diferencie escenarios, y la profundidad matemática esperada del documento.
- Si la LGD base sale ≈ 0: horizonte de liquidación de h días (medido con `liquidationCalls`) y/o ES 99,5% como sensibilidad.
- ES de stress < ES de baseline por el solapamiento de las ventanas de 730 días: ¿alcanza con documentarlo?
