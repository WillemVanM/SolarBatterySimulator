"""Find the cheapest PV + battery combination for a 5 % unserved-energy target."""
import matplotlib.pyplot as plt
import system as sys

if __name__ == "__main__":
    irradiance = sys.solar_irradiance()
    irradiance.load_csv_file(file_path="data/irradiance.csv", irradiance_name="gti")

    load = sys.consumption()
    load.consumption_determined_by_time_from_file(
        file_path="examples/consumption_example.csv", delimiter=";")
    daily_kWh = sum(load.consumption) / (60 / load.get_period())

    sy = sys.system(peak_power=6.0, battery_capacity=20.0,
                    price_solar=700, price_battery=390, fixed_cost=0,
                    solar_irradiance=irradiance, consumption=load)

    opt = sys.system_optimization(
        sy,
        min_peak_power=3.0, max_peak_power=7.0,
        min_battery_capacity=6.0, max_battery_capacity=20.0,
        optimization_objective="energy_from_grid",
        optimization_threshold=daily_kWh * 365.25 * 0.05)

    pv, bat, cost, obj = opt.find_optimal_system(nr_processes=1)
    print(f"Optimal PV      : {pv:.2f} kWp")
    print(f"Optimal battery : {bat:.2f} kWh")
    print(f"CAPEX           : {cost:.0f}")
    print(f"Objective       : {obj:.1f} kWh/yr unserved")

    # Zoom a Pareto sweep around the optimum
    opt.set_min_peak_power(0.8 * pv);        opt.set_max_peak_power(1.2 * pv)
    opt.set_min_battery_capacity(0.8 * bat); opt.set_max_battery_capacity(1.2 * bat)
    opt.brute_force_pareto(steps_peak_power=3, steps_battery_capacity=3)

    print(opt.get_pareto())
    fig, ax = plt.subplots()
    opt.plot_pareto(fig, ax, currency="€")
    plt.savefig("pareto_boundary.pdf")