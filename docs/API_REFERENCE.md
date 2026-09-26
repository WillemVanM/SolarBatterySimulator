# API reference

Module `system` (import as `import system as sys`) and module `gaussian_process`.

---

## `class solar_irradiance`

Time series of specific PV power in **kW per kWp**.

```python
solar_irradiance(time_series=None, irradiance=None, size_of_data=None, period=None)
```

| Method | Description |
|---|---|
| `load_csv_file(...)` | Read a CSV; see [DATA_FORMATS](DATA_FORMATS.md#1-irradiance--pv-production). Sets `irradiance`, `time_series`, `size_of_data`, `period`. |
| `get_size()` | Number of samples. |
| `get_period()` | Sampling period in minutes (from the first two time stamps). |
| `get_value(index)` | Specific power at `index` [kW/kWp]. |
| `get_irradiance()` | The whole NumPy array. |
| `get_time_series()` | List of `datetime`. |
| `set_size()` | Recompute `size_of_data` from `time_series`. |
| `split_irradiances(nr_processes)` | Split into that many objects along **whole days**, for multiprocessing. |

---

## `class consumption`

Load profile in **kW**, either a repeating typical day/week or absolute time stamps.

| Method | Description |
|---|---|
| `consumption_determined_by_time_from_file(file_path, consumption_name="Power [kW]", time_series_name="Time", delimiter=",", time_input="Time")` | Read a profile from CSV. |
| `consumption_from_consumers(consumers: dict)` | Build a profile by summing `{consumer: count}`. |
| `get_value(datetime)` | Linearly interpolated load at a time stamp. |
| `get_value_by_index(index)` | Load at `index % size_of_data` (fast path, no interpolation). |
| `get_period()` | Period in minutes. |
| `search_date(datetime_obj)` | Binary search → `(lower_index, interpolation_ratio)`. |

Attributes: `consumption` (array/list, kW), `time_series`, `time_input`
(`"time"` or `"datetime"`), `size_of_data`, `period`.

> `__init__` currently ignores its arguments — always use one of the loaders.

---

## `class consumer`

```python
consumer(name="", power=None, times=None, power_profile=None,
         period=timedelta(minutes=15), randomization=False,
         random_switch_on_time=None, random_times=None, random_on_proportion=0.0)
```

| Parameter | Meaning |
|---|---|
| `name` | Selects built-in defaults (see README table). Any other string = fully custom. |
| `power` | Rated power [W]. |
| `times` | `[(on_time, off_time), ...]` for deterministic devices. |
| `randomization` | Use the stochastic duty-cycle model. |
| `random_switch_on_time` | `timedelta` a switch-on lasts. |
| `random_times` | Windows in which the device may run. |
| `random_on_proportion` | Duty cycle inside those windows (0–1). |
| `period` | Resolution of the generated profile. |

| Method | Description |
|---|---|
| `calculate_power_consumption()` | Fills `self.power_profile` (a `consumption` with `time_series` and `consumption` in **W**). Called automatically by `consumption_from_consumers`. |

---

## `class system`

```python
system(peak_power=0, price_solar=800, battery_capacity=0, battery_efficiency=92,
       charging_efficiency=None, discharging_efficiency=None, DOD=80,
       price_battery=300, solar_irradiance=None, consumption=None,
       battery_profile=None, fixed_cost=0, ...)
```

| Parameter | Unit | Default | Notes |
|---|---|---|---|
| `peak_power` | kWp | 0 | PV array size |
| `battery_capacity` | kWh | 0 | Nominal capacity |
| `battery_efficiency` | % | 92 | One-way; used for both directions unless overridden |
| `charging_efficiency` / `discharging_efficiency` | % | = `battery_efficiency` | |
| `DOD` | % | 80 | Usable depth of discharge; floor = `100 − DOD` |
| `price_solar` | cur./kWp | 800 | |
| `price_battery` | cur./kWh | 300 | |
| `fixed_cost` | cur. | 0 | Size-independent CAPEX |

| Method | Returns | Description |
|---|---|---|
| `simulate_battery()` | `[kWh/yr, h/yr, events/yr]` | Full simulation; stores `battery_profile`. |
| `simulate_battery_without_profile(data_range=None)` | same | Memory-light; optional `[start_idx, stop_idx]`. |
| `run_battery_parallel(nr_processes, need_battery_profile=True)` | same | Day-aligned multiprocessing; each chunk restarts at 100 % SoC. See README known issues. |
| `plot_battery_profile(start_datetime, end_datetime)` | – | SoC plot with DOD floor; needs a stored profile. |
| `get_total_cost()` | cur. | CAPEX. |
| `set_peak_power(x)` / `set_battery_capacity(x)` | – | Setters used by the optimiser. |
| `_battery_step(...)`, `_split_systems(...)`, `child_process_battery(...)` | – | Internal. |

Post-simulation attributes: `energy_from_grid`, `black_out_time`, `black_outs`,
`minimal_battery_profile`, `battery_profile`.

---

## `class system_optimization`

```python
system_optimization(system=None, min_peak_power=0, max_peak_power=100,
                    min_battery_capacity=0, max_battery_capacity=100,
                    optimization_objective="energy_from_grid",
                    optimization_threshold=520)
```

`optimization_objective ∈ {"energy_from_grid", "black_out_time", "number_of_black_outs"}`;
`optimization_threshold` is expressed in the objective's own annual unit
(kWh/yr, h/yr, events/yr).

| Method | Description |
|---|---|
| `find_optimal_system(x=None, max_iterations=50, ftol=1e-2, nr_processes=1)` | Cheapest system meeting the threshold → `(peak_power, battery_capacity, cost, objective)`. `x` seeds the first budget guess. |
| `optimal_system_for_cost(cost, obj=0, x=None, max_iterations=10, f_tol=1e-2, nr_processes=1)` | Best reliability for a fixed budget → `(objective, peak_power)`. |
| `brute_force_pareto(steps_peak_power=5, steps_battery_capacity=5, nr_processes=1)` | Grid sweep, keeps the non-dominated front in `self.pareto`. |
| `get_pareto()` | Array of rows `[cost, objective, peak_power, battery_capacity]`. |
| `plot_pareto(fig=None, ax=None, currency="€")` | Interactive scatter (hover = show, click = pin). |
| `update_pareto()` | Insert the latest simulation result into the front. |
| `set_min_peak_power`, `set_max_peak_power`, `set_min_battery_capacity`, `set_max_battery_capacity` | Bound setters (handy for zooming a Pareto sweep around an optimum). |

Errors you may hit by construction:
*"Too high minimal values given to achieve optimal result"* — the smallest allowed system
already beats the threshold (lower your bounds or tighten the threshold);
*"Too low maximal values ..."* — even the largest allowed system fails
(raise the bounds or relax the threshold).

---

## Module `gaussian_process`

Bayesian optimisation / GP regression, implemented after Wang (2023).

| Function | Description |
|---|---|
| `GP_pred(X1, X2, Y, sigma=0.2, tau=0.15, only_mu=False, ground_level=None)` | Posterior mean and variance at `X2` given data `(X1, Y)`. Outputs are centred on `mean(Y)` (or `ground_level`) and scaled by mean absolute deviation. `sigma` = observation noise, `tau` = RBF length scale. |
| `next_point(X1, Y, bounds=None, prec=50, acq="UCB", sigma=0.2, tau=0.15, kappa=30, processes=1, scale_var=True)` | Maximiser of the Upper-Confidence-Bound acquisition `μ + κ·σ`. 1-D: dense grid; 2–3 D: full mesh; ≥ 4 D: coarse mesh → refine around the best candidates, optionally fanned out over MPI ranks. Inputs are min-max scaled onto `[0, 1]^d` using `bounds`. |
| `find_zero(X, Y, sigma=0.05, tau=0.5, tol=0.01)` | Root of a monotone decreasing function, from the GP mean, by bisection on `[min(X), max(X)]`. Used by `find_optimal_system`. |
| `GP_max_mu(X1, Y, bounds, X2, fixed_d, prec=40, sigma=0.2, tau=0.2, ref=None)` | Maximum of the posterior mean over the free dimensions with `fixed_d` held at `X2`. |
| `K(X1, X2, tau=1, same=False)` | RBF Gram matrix `exp(−‖x₁−x₂‖² / 2τ²)`; `same=True` exploits symmetry. |
| `ki(x1, x2, tau2)`, `product(X)` | Scalar kernel and helper. |

Only `acq="UCB"` is implemented. Hyper-parameters are **not** fitted; pick `tau`
relative to your normalised input range and `kappa` for the exploration you want.
`mpi4py` is imported lazily and only when `processes > 1`.