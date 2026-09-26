"""Minimal end-to-end example: load data, simulate one system, plot a month."""
from datetime import datetime
import matplotlib.pyplot as plt
import system as sys

if __name__ == "__main__":
    irradiance = sys.solar_irradiance()
    irradiance.load_csv_file(file_path="data/irradiance.csv", irradiance_name="gti")

    load = sys.consumption()
    load.consumption_determined_by_time_from_file(
        file_path="examples/consumption_example.csv", delimiter=";")

    steps_per_hour = 60 / load.get_period()
    daily_kWh = sum(load.consumption) / steps_per_hour
    print(f"Daily consumption: {daily_kWh:.2f} kWh")
    print(f"Equivalent sun hours: "
          f"{sum(irradiance.get_irradiance()) / irradiance.get_size() * 24:.2f} h/day")

    sy = sys.system(peak_power=6.0, battery_capacity=20.0,
                    price_solar=700, price_battery=390,
                    solar_irradiance=irradiance, consumption=load)

    unserved, blackout_h, blackouts = sy.simulate_battery()
    print(f"Unserved energy : {unserved:.1f} kWh/yr "
          f"({100 * unserved / (daily_kWh * 365.25):.1f} % of demand)")
    print(f"Black-out time  : {blackout_h:.1f} h/yr in {blackouts:.0f} events")
    print(f"Lowest SoC      : {sy.minimal_battery_profile:.1f} %")
    print(f"CAPEX           : {sy.get_total_cost():.0f}")

    plt.figure()
    sy.plot_battery_profile(datetime.fromisoformat("2005-01-01"),
                            datetime.fromisoformat("2005-02-01"))
    plt.savefig("battery_profile_january.pdf")