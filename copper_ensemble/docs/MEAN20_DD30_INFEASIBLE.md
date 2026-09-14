# Mean ≥ 20% yearly + max DD < 30% — search outcome

**Verdict: infeasible** on Yahoo HG 2008→now under the same ATR TP/stop + system DD halt + ensemble (± yearly lock) methodology.

## Evidence

| Search | Configs | Dual hits (mean≥20% & DD<30%) |
|---|---|---|
| Lock on/off grid (`scripts/probe_mean20_dd30.py`) | 1,050 | **0** |
| Soft yearly lock 8–25% (`scripts/probe_softlock_mean20_dd30.py`) | 384 | **0** |

## Frontier (full sample)

| Cap | Best mean yearly | Notes |
|---|---|---|
| DD < 30% | ~**4.1%** | Requires tight yearly lock |
| DD < 50% | ~**15.9%** | Still below 20% |
| Mean ≥ 20% | min DD ~**89%** | Lock off; path essentially ruined |

## Production

Left unchanged: DD-aware yearly-lock book (~3% mean yearly, DD ~26.5%, all calendar years green). Shipping a “20% / DD<30%” config would be false — no such point exists on this Pareto surface.

Artifacts:
- `/opt/cursor/artifacts/copper_mean20_dd30_probe.json`
- `/opt/cursor/artifacts/copper_mean20_dd30_probe.log`
- `/opt/cursor/artifacts/copper_mean20_dd30_softlock.json`
- `/opt/cursor/artifacts/copper_mean20_dd30_softlock.log`
