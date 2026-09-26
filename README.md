# SolarBatterySimulator

Size and validate **off-grid / hybrid solar-plus-battery systems** from historical
irradiance data and a realistic electricity-consumption profile.

The simulator answers the two questions that dominate rural-electrification and
back-up-power design:

1. *For a given PV peak power and battery capacity, how often and how badly does the
   system fail?* (unserved energy, black-out hours, number of black-outs per year)
2. *What is the cheapest combination of PV and battery that keeps failures below a
   target threshold?* (single-objective optimisation and full cost/reliability
   Pareto front)

Consumption can be supplied either as a **measured time series** (typical day or week,
repeated periodically) or built up **device-by-device** from a library of appliances with
deterministic or stochastic duty profiles (fridges, laptops, printers, lights, …).

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Input data](#input-data)
- [Simulation model](#simulation-model)
- [Optimisation](#optimisation)
- [Parallel execution](#parallel-execution)
- [Plots and outputs](#plots-and-outputs)
- [Validating against measured data](#validating-against-measured-data)
- [Repository layout](#repository-layout)
- [Known issues and limitations](#known-issues-and-limitations)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

---

## Features

| Capability | Where |
|---|---|
| Read irradiance/production CSV files (Solcast, SoDa/HelioClim, PVGIS, Victron exports) with auto-detected headers, separate date/time columns, `24:00` timestamps, `dd/mm/yyyy` dates | `solar_irradiance.load_csv_file` |
| Consumption from a CSV time series (interpolated, periodically repeated) | `consumption.consumption_determined_by_time_from_file` |
| Consumption synthesised from an appliance list, incl. randomised duty cycles | `consumption.consumption_from_consumers`, `consumer` |
| Quarter-hourly (or any period) battery state-of-charge simulation with charge/discharge efficiency and depth-of-discharge limit | `system.simulate_battery` |
| Reliability metrics annualised to kWh/yr, h/yr, events/yr | `system.simulate_battery(_without_profile)` |
| Multiprocessing over time slices | `system.run_battery_parallel` |
| Cheapest system meeting a reliability threshold, via iso-cost inner optimisation + Gaussian-process root finding | `system_optimization.find_optimal_system` |
| Brute-force cost/reliability Pareto front with interactive annotations | `system_optimization.brute_force_pareto`, `plot_pareto` |
| Stand-alone Gaussian-process regression / Bayesian-optimisation helpers (UCB acquisition, optional MPI) | `gaussian_process.py` |

---

## Installation

Python ≥ 3.9 is recommended (`datetime.fromisoformat` is used heavily).

```bash
git clone https://github.com/<your-user>/SolarBatterySimulator.git
cd SolarBatterySimulator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
scipy
matplotlib
mpi4py   # optional: only needed for gaussian_process.next_point(processes > 1)
```

Notes:

* `tkinter` is used only for the *"select a file"* dialogs. It ships with most CPython
  builds; on Debian/Ubuntu install `python3-tk`. If you always pass `file_path=...`
  explicitly, tkinter is never touched at run time (but the import at the top of
  `system.py` still needs it — install it, or delete the two tkinter imports).
* On Windows/macOS, `multiprocessing` requires that your driver code sits inside
  `if __name__ == "__main__":` (as in `main.py`).

---

## Quick start

### 1. Simulate one candidate system

```python
import system as sys
from datetime import datetime
import matplotlib.pyplot as plt

# --- Irradiance: normalised production, kW per kWp installed ---------------
irradiance = sys.solar_irradiance()
irradiance.load_csv_file(
    file_path="data/csv_13.31_-16.68_fixed_13_180_PT15M.csv",
    irradiance_name="gti",          # column with W/m^2 (or W per kWp)
    time_series_name="period_start")

# --- Consumption: measured/typical daily profile ---------------------------
consumption = sys.consumption()
consumption.consumption_determined_by_time_from_file(
    file_path="examples/consumption_example.csv", delimiter=";")

# --- System ---------------------------------------------------------------
sy = sys.system(peak_power=6.0,          # kWp
                battery_capacity=20.0,   # kWh (nominal)
                battery_efficiency=92,   # % one-way
                DOD=80,                  # % usable depth of discharge
                price_solar=700,         # currency / kWp
                price_battery=390,       # currency / kWh
                solar_irradiance=irradiance,
                consumption=consumption)

unserved_kWh_yr, blackout_h_yr, blackouts_yr = sy.simulate_battery()
print(f"{unserved_kWh_yr:.1f} kWh/yr unserved, "
      f"{blackout_h_yr:.1f} h/yr in {blackouts_yr:.0f} events/yr")

plt.figure()
sy.plot_battery_profile(datetime.fromisoformat("2005-01-01"),
                        datetime.fromisoformat("2005-02-01"))
plt.savefig("battery_profile_january.pdf")
```

### 2. Build consumption from appliances instead

```python
consumption = sys.consumption()
consumption.consumption_from_consumers({
    sys.consumer("LED"):              12,
    sys.consumer("security-light"):    3,
    sys.consumer("fridge"):            1,
    sys.consumer("desktop_computer2"): 10,
    sys.consumer("printer"):           2,
    sys.consumer("projector"):         1,
    sys.consumer("wifi"):              1,
    sys.consumer("smartphone"):       15,
})
daily_kWh = sum(consumption.consumption) / (60 / consumption.get_period())
print(f"Daily consumption: {daily_kWh:.2f} kWh")
```

> **Keep the periods equal.** If the consumption period equals the irradiance period,
> the simulator uses fast index-based lookup. Mixed periods fall back to
> timestamp interpolation, which is slower and currently fragile
> (see [Known issues](#known-issues-and-limitations)).

### 3. Find the cheapest system for a reliability target

```python
opt = sys.system_optimization(
    sy,
    min_peak_power=3.,  max_peak_power=7.,        # kWp search bounds
    min_battery_capacity=6., max_battery_capacity=20.,   # kWh search bounds
    optimization_objective="energy_from_grid",    # or "black_out_time" / "number_of_black_outs"
    optimization_threshold=daily_kWh * 365.25 * 0.05)    # allow 5 % unserved energy

pv_kWp, battery_kWh, cost, achieved = opt.find_optimal_system(nr_processes=4)
print(pv_kWp, battery_kWh, cost, achieved)
```

### 4. Pareto front around the optimum

```python
opt.set_min_peak_power(0.8 * pv_kWp);      opt.set_max_peak_power(1.2 * pv_kWp)
opt.set_min_battery_capacity(0.8 * battery_kWh)
opt.set_max_battery_capacity(1.2 * battery_kWh)
opt.brute_force_pareto(steps_peak_power=3, steps_battery_capacity=3, nr_processes=4)

fig, ax = plt.subplots()
opt.plot_pareto(fig, ax, currency="€")   # hover/click points to reveal PV & battery sizes
plt.savefig("pareto_boundary.pdf")
```

---

## Input data

Full details, including worked header examples for Solcast, SoDa and Victron VRM
exports, are in **[docs/DATA_FORMATS.md](docs/DATA_FORMATS.md)**. Summary:

### Irradiance / production file

* One row per time step, constant step length (the step is inferred from the **first two
  timestamps**).
* The irradiance column is divided by 1000 and multiplied by `multiplication`, so the
  internal unit is **kW per kWp installed**:
  `generation [kWh] = peak_power [kWp] · value [kW/kWp] · Δt [h]`.
  Feeding plane-of-array irradiance in W/m² therefore implies the standard-test-condition
  assumption 1000 W/m² → 1 kW/kWp, i.e. *no* temperature, soiling, wiring or inverter
  losses. Apply a derating factor via `multiplication` (e.g. `multiplication=0.8`)
  or pre-derate the file.
* Empty cells are read as 0. **Negative** values are interpreted as *missing data*: the
  state of charge is then held constant instead of being propagated.
* `row_nr_start=1e11` (any value ≥ 1e10) switches on **automatic header detection**:
  leading blank lines and lines starting with `#` are skipped, the last of them is used
  as the column-name row, and the data block ends at the first blank line.
* `date_time_sep=True` handles files that split date and time into two columns
  (`date_series_name`, `time_series_name`), and copes with `24:00` (rolled to `00:00`
  of the next day) and `dd/mm/yyyy` dates.

### Consumption file

Two columns, with a header row:

```csv
Time;Power [kW]
0:00;0.35
0:15;0.34
...
23:45;0.41
```

* Column names default to `Time` and `Power [kW]` and are configurable.
* `time_input="time"` (default) → the profile is a **typical day**, repeated for the whole
  simulation. `time_input="datetime"` → absolute timestamps.
* Reading stops at the first row with an empty time cell, so trailing junk is tolerated.
* Values are **kW** and are *not* rescaled.

### Appliance library (`consumer`)

Instantiate with a name to get sensible defaults, or override `power` (W) and
`times` (list of `(on, off)` `datetime.time` tuples); set `randomization=True` with
`random_switch_on_time`, `random_times` and `random_on_proportion` for stochastic loads.

| Name | Power [W] | Default behaviour |
|---|---|---|
| `LED` | 5 | on 17:00 → 24:00 |
| `incandescent` | 50 | on 17:00 → 24:00 |
| `security-light` | 40 | on 00:00–07:00 and 18:00–24:00 |
| `fridge` | 100 | random, 15 min cycles, 33 % duty, all day |
| `freezer` | 100 | random, 15 min cycles, 33 % duty, all day |
| `laptop` | 40 | random, 2 h sessions, ~29 % duty, 08:00–22:00 |
| `desktop_computer` | 200 | random, 2 h sessions, ~29 % duty, 08:00–22:00 |
| `desktop_computer2` | 200 | fixed lesson blocks 08:30–09:15, 10:00–10:45, 14:00–14:45 |
| `projector` | 300 | 08:00–12:00 and 13:00–18:00 |
| `wifi` | 20 | 24 h |
| `smartphone` | 5 | random, 1.5 h charges, 25 % duty, 08:00–18:00 |
| `ventilator` | 80 | 11:00–15:00 |
| `printer` | 100 | random, 30 min bursts, 5 % duty, 08:00–18:00 |
| `oven` | 1500 | 18:30–19:30 |
| `microwave` | 900 | 18:30–19:30 |
| `air-conditioning` | 700 | 10:30–15:30 |
| `TV` | 60 | 17:00–23:00 |
| `washing_machine` | 300 | 12:00–14:30 |

Randomised profiles are re-drawn until their daily energy is within a tolerance band of
`power · duty · available_time`, so totals stay physical while the shape varies.
Default data period is 15 min (`period=timedelta(minutes=15)`).

---

## Simulation model

See **[docs/MODEL.md](docs/MODEL.md)** for the derivation. Per time step *i* of length
Δt (minutes), with state of charge `S` in %:

```
E_gen   = peak_power · g[i] · Δt/60                     [kWh]
E_load  = P_load(i or t) · Δt/60                        [kWh]
E_net   = E_gen − E_load

E_net > 0 :  S ← min(S + E_net · η_charge   / C · 100, 100)
E_net ≤ 0 :  S ←     S + E_net / η_discharge / C · 100
```

`C` is the nominal battery capacity [kWh], `η` the one-way efficiencies [%]. The battery
starts at 100 %.

When `S` falls below the floor `100 − DOD`:

* the missing energy `C/100 · (100 − DOD − S)` is added to **`energy_from_grid`**
  (interpret as unserved load for a stand-alone system, or as grid import for a hybrid one),
* `S` is clamped to the floor,
* `black_out_time` grows by Δt/60 hours,
* a new black-out **event** is counted if the previous step was not already in deficit.

`minimal_battery_profile` (lowest SoC reached) is also stored; the optimiser uses it as a
tie-breaker when a configuration never fails.

All three returned metrics are **annualised**:

```
value_per_year = value / n_samples · (24 · 60 · 365.25 / Δt)
```

Deliberate simplifications: no inverter/charge-controller limits, no C-rate limits, no
temperature or ageing model, no load shedding or dispatch logic, no PV curtailment
accounting (excess energy above SoC = 100 % is simply discarded), and no self-discharge.

---

## Optimisation

`system_optimization` treats the design problem as: *minimise cost subject to
`objective ≤ optimization_threshold`*, where the objective is one of
`"energy_from_grid"`, `"black_out_time"`, `"number_of_black_outs"`.

**Inner problem — `optimal_system_for_cost(cost)`**
For a fixed budget, the battery size is determined by the PV size:
`C = (cost − fixed_cost − price_solar · P) / price_battery`. A bounded scalar
`scipy.optimize.minimize` then searches `P ∈ [0, 0.99 · (cost − fixed_cost)/price_solar]`
for the best reliability at that budget. If the objective hits exactly zero, the
surrogate objective `−minimal_battery_profile` is returned so the search keeps
distinguishing "just barely fine" from "comfortably fine".

**Outer problem — `find_optimal_system()`**
The reliability-vs-budget curve is monotone, so the required budget is the root of
`objective(cost) − threshold`. The root is bracketed by the cost of the min-size and
max-size corners, then located with `gaussian_process.find_zero`, a
GP-regression surrogate refined by bisection. Convergence is on the relative
criterion `|objective − threshold| / threshold ≤ ftol` (default 1 %), with at most
`max_iterations=50` outer steps; inner tolerance is loosened adaptively while far
from the target. The bounds are validated first: if the smallest allowed system already
beats the threshold, or the largest cannot reach it, you are told to widen/narrow the
bounds.

Returns `(peak_power, battery_capacity, cost, achieved_objective)`.

**Pareto front — `brute_force_pareto()`** sweeps a `steps_peak_power × steps_battery_capacity`
grid, simulates each point, and keeps only non-dominated `(cost, objective)` pairs via
`update_pareto()`. `plot_pareto()` draws the front and attaches annotations showing
`PV: … kWp; Bat: … kWh`; annotations appear on hover and can be pinned by clicking
(needs an interactive matplotlib backend).

`gaussian_process.py` is self-contained and reusable: `GP_pred` (posterior mean/variance
with an RBF kernel, output standardisation and `sigma²` jitter), `next_point`
(Bayesian optimisation with an Upper-Confidence-Bound acquisition, coarse-to-fine grid
search in ≥ 4 dimensions, optional MPI fan-out), `find_zero`, `GP_max_mu`.
Kernel hyper-parameters (`sigma` noise, `tau` length scale, `kappa` exploration weight)
are user-supplied — there is no marginal-likelihood fitting.

---

## Parallel execution

```python
results = sy.run_battery_parallel(nr_processes=4, need_battery_profile=False)
```

The year is cut into whole-day chunks, one `multiprocessing.Process` per chunk, and each
chunk restarts the battery at 100 %. Because most profiles re-equilibrate within a day,
the induced error is small for typical systems — but it is *not* zero, and it grows for
systems with multi-day autonomy. Use `nr_processes=1` (i.e. `simulate_battery()`) for
final numbers and the parallel path for optimisation sweeps.

`need_battery_profile=False` skips storing the full SoC trace, which is what the optimiser
wants. Please read the [Known issues](#known-issues-and-limitations) entry on
`run_battery_parallel` before relying on its return values.

---

## Plots and outputs

| Call | Produces |
|---|---|
| `system.plot_battery_profile(start, end)` | SoC [%] vs time with the `100 − DOD` floor as a dash-dot line, x-limited to the window |
| `system_optimization.plot_pareto(fig, ax, currency="€")` | Cost vs objective scatter with interactive size annotations |
| `system.get_total_cost()` | `fixed_cost + price_solar · peak_power + price_battery · battery_capacity` |
| `system.energy_from_grid`, `.black_out_time`, `.black_outs`, `.minimal_battery_profile`, `.battery_profile` | Raw attributes after a simulation |

`plot_battery_profile` requires a previous `simulate_battery()` (not
`simulate_battery_without_profile()`), since it needs the stored trace.

---

## Validating against measured data

`examples/compare_production_reality.py` overlays modelled specific yield on measured
inverter data (a Victron VRM `kWh` export, in the example the sum of *PV to battery* and
*PV to grid/consumer*):

```
measured_specific_yield [kW/kWp] = (PV→battery + PV→load) [kWh] · (60/Δt) / peak_power
```

with an optional `time_zone_difference` shift applied to the modelled series. The script
reuses `solar_irradiance.load_csv_file` to read the energy columns — a pragmatic hack:
the loader will happily divide those values by 1000, which the formula above undoes.
Use it to sanity-check tilt/azimuth, time zone, and the derating factor before sizing.

---

## Repository layout

```
system.py              Core library: solar_irradiance, consumption, consumer,
                       system, system_optimization
gaussian_process.py    GP regression, UCB Bayesian optimisation, GP root finding
main.py                Example driver used for a real project (Benin, A2D/OVO):
                       load data, plot profiles, simulate, optimise, Pareto
examples/              Minimal runnable scripts + sample consumption CSV
docs/                  Data formats, model description, API reference
```

`main.py` contains hard-coded absolute paths and project-specific choices; treat it as a
template and copy it rather than editing it in place.

---

## Known issues and limitations

These are real, reproducible rough edges in the current code. Contributions welcome.

**Blocking / correctness**

1. `system.run_battery_parallel(..., need_battery_profile=True)` raises `NameError`:
   `period` is only defined in the `False` branch but is used in the return statement.
2. `run_battery_parallel` sums **already annualised** per-chunk metrics and then
   annualises again, so its return values are not directly comparable with
   `simulate_battery()`. Cross-check any parallel result against a sequential run.
3. `consumption.__init__` ignores all its arguments and sets every attribute to `None`;
   always populate a `consumption` object through one of the two loader methods.
4. `consumption_from_consumers` sets `time_input = "Time"`, while `search_date`/`get_value`
   test for the lower-case `"time"`. Timestamp-based lookup therefore misbehaves for
   synthesised profiles — keep the consumption period equal to the irradiance period so
   the fast index path (`get_value_by_index`) is used.
5. `self.black_outs.append(datetime)` appends the shadowed `datetime` argument, not the
   event time, so only `len(black_outs)` is meaningful.
6. `simulate_battery_without_profile` falls back to `self.battery_profile[i-1]` when a
   sample is flagged as missing (negative irradiance), which fails if no profile was ever
   stored.
7. `update_pareto` contains `size_pareto -= size_pareto` (intended `-= 1`) and mutates the
   list while scanning it; dominated points can survive on the front.
8. In `gaussian_process.next_point`, the `scale_var=False` branch assigns
   `scale_var = np.ones(...)` instead of `scale`, so `scale` stays undefined.

**Modelling**

9. No inverter or charge-controller power limits, no C-rate limit, no battery ageing,
   self-discharge or temperature dependence; PV losses must be folded into
   `multiplication`.
10. Excess PV beyond a full battery is discarded silently (no curtailment or
    export accounting).
11. Costs are pure CAPEX — no replacement schedule, O&M, discount rate or LCOE.
12. Time-stamp handling assumes a *constant* step inferred from the first two rows; gaps
    and daylight-saving jumps are not detected.
13. `consumer.calculate_power_consumption` labels randomised time stamps with
    `// 3660` instead of `// 3600`, so hour labels of stochastic profiles drift
    (the power array itself is unaffected).
14. GP hyper-parameters are fixed, and the kernel is built with pure-Python double loops
    (`O(N²)` and slow for large designs of experiments).

---

## Roadmap

- [ ] Fix the items above, with regression tests on a synthetic year of data
- [ ] Unit tests (`pytest`) for CSV parsing, battery stepping, annualisation, Pareto logic
- [ ] Optional inverter/charge-controller limits and temperature-dependent PV derating
- [ ] LCOE objective with battery replacement and discounting
- [ ] Replace the bespoke GP with `scikit-learn`/`GPyTorch` (or keep it as a zero-dependency fallback)
- [ ] `pyproject.toml` packaging, a CLI (`solarbatterysim simulate --config my_site.yaml`)
- [ ] Vectorised (NumPy) battery loop; the step recursion can be reformulated per clear-sky day block

---

## References

* J. Wang, "An intuitive tutorial to Gaussian process regression," *Computing in Science &
  Engineering*, **25**(4), 4–11, 2023. doi:10.1109/MCSE.2023.3342149
  (preprint: arXiv:2009.10862) — the basis of `gaussian_process.py`.

```bibtex
@article{wang2023intuitive,
  title={An intuitive tutorial to {Gaussian} process regression},
  author={Wang, Jie},
  journal={Computing in Science \& Engineering},
  volume={25}, number={4}, pages={4--11}, year={2023}, publisher={IEEE}
}
```

Typical data sources: Solcast time series (`period_start`, `gti`), SoDa / HelioClim-3
(`# Date`, `Time`, `Global Inclined`), PVGIS hourly radiation, and Victron VRM exports for
validation.

## Acknowledgements

Developed in the context of [Humasol](https://humasol.be) student-engineering projects for
off-grid solar installations (Tanzania, The Gambia, Benin, …).

## License

MIT — see [`LICENSE`](LICENSE).