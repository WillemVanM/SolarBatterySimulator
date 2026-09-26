# Input data formats

All loaders are CSV-based and driven by **column names**, not column positions. If a
requested name is not found in the header row, the loader prints the available names and
prompts you on stdin, so an interactive session can recover from a typo.

---

## 1. Irradiance / PV production

```python
irradiance = sys.solar_irradiance()
irradiance.load_csv_file(
    file_path=None,                  # None -> tkinter file dialog
    irradiance_name=None,            # e.g. "gti", "Global Inclined", "G(i)"
    time_series_name="period_start",
    delimiter=",",
    row_nr_names=0,                  # 0-based index of the header row
    row_nr_start=1,                  # 0-based index of the first data row
    date_time_sep=False,             # True if date and time are in separate columns
    date_series_name="# Date",
    multiplication=1)                # extra scaling / derating factor
```

### Units and the meaning of the irradiance column

Values are stored as `value / 1000 · multiplication` and consumed as

```
generation [kWh] = peak_power [kWp] × stored_value × Δt [h]
```

so the stored value is a **specific power in kW per kWp**. Consequences:

| You supply | What to set | Implied assumption |
|---|---|---|
| Plane-of-array irradiance in W/m² | `multiplication=1` | 1000 W/m² ⇒ 1 kW/kWp; no losses |
| Same, with a 20 % system derate | `multiplication=0.8` | flat loss factor |
| 15-min energy in Wh/m² per step | `multiplication=4` | converts Wh/15 min → average W |
| Hourly energy in Wh/m² per step | `multiplication=1` | already average W over the hour |
| Already-normalised kW/kWp | pre-multiply by 1000 | — |

### Special values

* empty cell → `0`
* **negative** value → treated as *missing data*: the state of charge is frozen at its
  previous value (or at the DOD floor) for that step instead of being updated.

### Header auto-detection

Set `row_nr_start` to anything ≥ `1e10` (idiomatically `1e11`) for files with a variable
number of comment lines:

* leading empty lines and lines whose first field starts with `#` are counted as header,
* the **last** header line becomes the column-name row (`row_nr_names` is overwritten),
* the data block ends at the first empty line after data has started.

This is what the SoDa/HelioClim exports need.

### Time stamps

`date_time_sep=False` — one ISO-8601-ish column. The loader repairs:

* `.0000000` fractional-second padding (stripped),
* offsets written as `+01:00` → `+0100`,
* a trailing `Z` (stripped; the series is then naive local time).

`date_time_sep=True` — two columns. The loader repairs:

* `dd/mm/yyyy` → `yyyy-mm-dd` (detected when the third field is longer than the first),
* `H:MM` → `HH:MM`, missing seconds appended,
* `24:00` → `00:00` of the **next** day.

The sampling period is inferred from the **first two rows** and stored in minutes
(`get_period()`); a constant step is assumed throughout.

### Worked header examples

**Solcast**

```csv
period_end,period_start,period,air_temp,dni,ghi,gti
2021-01-01T00:15:00Z,2021-01-01T00:00:00Z,PT15M,22.0,0,0,0
```

```python
irradiance.load_csv_file(file_path="csv_13.31_-16.68_fixed_13_180_PT15M.csv",
                         irradiance_name="gti")      # defaults are fine
```

**SoDa / HelioClim-3 (semicolon-delimited, `#` comment block)**

```csv
# Latitude: 9.321 ...
# Date;Time;Global Horiz;Global Inclined;...
2005-01-01;00:15;0;0;...
```

```python
irradiance.load_csv_file(file_path="SoDa_HC3-METEO_lat9.321_lon2.626.csv",
                         delimiter=";", time_series_name="Time",
                         date_time_sep=True, date_series_name="# Date",
                         irradiance_name="Global Inclined",
                         multiplication=4, row_nr_start=1e11)
```

**Victron VRM energy export (used for validation, not for sizing)**

```csv
timestamp;PV to battery;PV to grid;...
```

```python
measured = sys.solar_irradiance()
measured.load_csv_file(file_path="Kudimba_kwh_....csv", delimiter=";",
                       time_series_name="timestamp",
                       irradiance_name="PV to battery", row_nr_start=2)
```

Remember that the loader divides by 1000; multiply back when plotting kWh data.

---

## 2. Consumption time series

```python
consumption = sys.consumption()
consumption.consumption_determined_by_time_from_file(
    file_path=None,
    consumption_name="Power [kW]",
    time_series_name="Time",
    delimiter=",",
    time_input="Time")               # "time" (typical day) or "datetime" (absolute)
```

```csv
Time;Power [kW]
0:00;0.35
0:15;0.34
0:30;0.34
...
23:45;0.41
```

* **Unit: kW.** No scaling is applied.
* `time_input="time"`/`"Time"` → the profile is a *typical day* and is wrapped around
  periodically for the whole simulation horizon (`get_value_by_index` uses
  `index % size_of_data`). A typical week works too, as long as the file covers exactly
  the repeat period and the horizon is interpreted accordingly.
* Parsing stops at the first row with an empty time field.
* Times like `0:00` are padded to `00:00` automatically.
* The period is inferred from the first two rows, and **should equal the irradiance
  period** — the simulator then uses fast index matching
  (`simplified_consumption_evaluation = True`). Otherwise it interpolates linearly
  between neighbouring samples via `search_date`, which is slower and currently
  unreliable for synthesised profiles (see README known issues).

Daily energy from a loaded profile:

```python
daily_kWh = sum(consumption.consumption) / (60 / consumption.get_period())
```

---

## 3. Appliance-based consumption

```python
consumption.consumption_from_consumers({sys.consumer("LED"): 28, ...})
```

Each key is a `consumer` **template** and each value the **count** of identical devices;
the template is re-evaluated for every unit, so randomised appliances get independent
draws. Power is summed in W and divided by 1000 → kW.

Custom deterministic device:

```python
lamp = sys.consumer(name="my-lamp", power=9,
                    times=[(time(18, 0), time(22, 30))],
                    period=timedelta(minutes=15))
```

Custom stochastic device:

```python
pump = sys.consumer(name="water-pump", power=750,
                    randomization=True,
                    random_switch_on_time=timedelta(minutes=45),
                    random_times=[(time(6, 0), time(9, 0)),
                                  (time(16, 0), time(19, 0))],
                    random_on_proportion=0.4)   # duty cycle inside those windows
```

How randomisation works: at each step inside a `random_times` window the device switches
on with probability `1 − (1 − duty)^(Δt / switch_on_time)` and then stays on for
`random_switch_on_time`. The whole day is re-drawn until the daily energy lands within a
slowly widening tolerance band around `power · duty · Σ(window lengths)`, which keeps
totals physical while preserving shape variability.

Interval end points are matched **exactly** against the step grid, so choose `times`
boundaries that fall on multiples of `period` (e.g. 15-minute marks). To keep a device on
until midnight use `time(23, 59, 59, 999999)` as in the built-ins.