# Simulation model, units and metrics

## Symbols

| Symbol | Code | Unit |
|---|---|---|
| `P_pk` | `system.peak_power` | kWp |
| `g[i]` | `solar_irradiance.get_value(i)` | kW/kWp |
| `Δt` | `solar_irradiance.get_period()` | minutes |
| `L` | `consumption.get_value*()` | kW |
| `C` | `system.battery_capacity` | kWh (nominal) |
| `η_c, η_d` | `charging_efficiency`, `discharging_efficiency` | % |
| `DOD` | `system.DOD` | % |
| `S` | `system.battery_profile` | % state of charge |

By default `charging_efficiency = discharging_efficiency = battery_efficiency` (92 %),
so a full charge/discharge round trip has efficiency ≈ 0.92² ≈ 85 %.

## One time step (`system._battery_step`)

```
E_gen  = P_pk · g[i] · Δt/60
E_load = L · Δt/60
E_net  = E_gen − E_load

if E_net > 0:   S' = min(S + 100 · E_net · η_c/100 / C , 100)
else:           S' =     S + 100 · E_net / (η_d/100) / C
```

Charging losses reduce what enters the battery; discharging losses inflate what must
leave it. Energy that would push `S` above 100 % is lost (no curtailment bookkeeping).
`g[i] < 0` returns `None`, the sentinel for missing data.

## Deficit handling (`simulate_battery`)

```
floor = 100 − DOD

if S < floor:
    energy_from_grid += C/100 · (floor − S)
    S = floor
    black_out_time  += Δt/60
    black_outs      += 1   (only on the transition into deficit)
```

Interpretation:

* **Stand-alone system** — `energy_from_grid` is *unserved energy* (energy the load
  demanded but the system could not deliver); `black_out_time` and the event count are the
  reliability metrics.
* **Hybrid/grid-tied system** — the same quantity is the energy that must be imported, and
  the "black-out" metrics become "grid-support" metrics.

Simulations start with a full battery (`S₀ = 100 %`), so discard a warm-up period if your
site would realistically start empty, or simply rely on a multi-year data set.

## Annualisation

Every metric returned by `simulate_battery`, `simulate_battery_without_profile` and
`run_battery_parallel` is scaled from the simulated window to a calendar year:

```
per_year = raw / n_samples · (24 · 60 · 365.25 / Δt)
```

Consequently, a 3-year input file yields a *per-year average*, not a total.

Useful threshold recipe (as used in `main.py`): allow 5 % of annual demand unserved:

```python
threshold = daily_kWh * 365.25 * 0.05
```

## Cost model

```
cost = fixed_cost + price_solar · peak_power + price_battery · battery_capacity
```

Pure CAPEX in one currency; `fixed_cost` absorbs inverters, cabling, transport,
installation and other size-independent items. There is no O&M, no battery replacement,
no discounting; a cost/reliability Pareto front is therefore an *investment* front, not an
LCOE front.

## Deliberate simplifications

| Not modelled | Practical consequence |
|---|---|
| Inverter / MPPT power limits, clipping | Oversized PV arrays look better than they are |
| Battery C-rate limits | Very small batteries charge/discharge unrealistically fast |
| Temperature, soiling, ageing, self-discharge | Fold into `multiplication`; add margin on capacity |
| Load shedding, demand response, priority loads | Deficits are counted, never avoided |
| Generator / grid dispatch strategy | Grid is an infinite, instantaneous backstop |
| Data gaps, DST shifts | Constant step assumed from the first two rows |

## Validation workflow

1. Plot one modelled day against measured inverter output
   (`examples/compare_production_reality.py`) to verify tilt, azimuth and time zone.
2. Tune `multiplication` until modelled and measured specific yields agree on clear days.
3. Check the measured daily energy of the consumption profile against billing/logger data.
4. Only then run the optimiser — sizing errors are dominated by input-data errors, not by
   the optimiser's tolerance.