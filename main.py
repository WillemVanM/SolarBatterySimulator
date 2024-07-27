import system as sys
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import time
import os

if __name__ == '__main__':
    figure_map = r"/mnt/c/Users/wille/Documents/Humasol/Partner_Relations/2425/A2D - OVO - Benin/"
    # Load irradiance data
    irradiance = sys.solar_irradiance()
    irradiance.load_csv_file(
        # file_path=r"C:\Users\wille\Documents\Humasol\Partner_Relations\2324\Hope_for_girls_and_women-Tanzania/" +
        #           "csv_-1.8416119_34.6686364_fixed_2_0_PT15M.csv",
        file_path=figure_map +
                  "SoDa_HC3-METEO_lat9.321_lon2.626_2004-02-01_2006-12-31_696304395.csv", delimiter=';',
        time_series_name="Time", date_time_sep = True, multiplication=4, row_nr_start=1e11,
        irradiance_name = "Global Inclined")

    consumption = sys.consumption()
    consumption.consumption_determined_by_time_from_file(
        file_path=figure_map + "consumption.csv",
        delimiter=";")
    # consumption.consumption_from_consumers({
    #     sys.consumer("LED"): 28+22,
    #     sys.consumer("ventilator"): 0,
    #     sys.consumer("security-light"): 3,
    #     sys.consumer("microwave"): 1,
    #     sys.consumer("fridge"): 1,
    #     sys.consumer("printer"): 1,
    #     sys.consumer("projector"): 1,
    #     sys.consumer("wifi"): 1,
    #     sys.consumer("desktop_computer2"): 20,
    #     sys.consumer("smartphone"): 15})
    # consumption.consumption_from_consumers({
    #     sys.consumer("LED"): 10,
    #     sys.consumer("ventilator"): 0,
    #     sys.consumer("security-light"): 0,
    #     sys.consumer("microwave"): 0,
    #     sys.consumer("fridge"): 1,
    #     sys.consumer("printer"): 2,
    #     sys.consumer("projector"): 1,
    #     sys.consumer("wifi"): 1,
    #     sys.consumer("desktop_computer2"): 10,
    #     sys.consumer("washing_machine"): 1,
    #     sys.consumer("TV"): 1,
    #     sys.consumer("smartphone"): 10})

    # print(sum(consumption.consumption) / 4)
    # Initialize system object
    system = sys.system(peak_power=6,
                        battery_capacity=20,
                        price_battery=390,
                        price_solar=700,
                        solar_irradiance=irradiance,
                        consumption=consumption)

    plt.figure()
    plt.plot(np.linspace(1/(24*4), 24, 24*4), consumption.consumption[:24*4])
    plt.xlabel("Time [h]")
    plt.ylabel("Consumption [kW]")
    daily_cons = sum(consumption.consumption[:24*4]) / 4
    print("Daily consumption = " + str(daily_cons) + " kWh")
    plt.savefig(figure_map + "consumption_profile.pdf")
    plt.figure()
    plt.plot(np.linspace(1/(24*4), 24, 24*4), irradiance.get_irradiance()[:24*4])
    plt.xlabel("Time [h]")
    plt.ylabel("Production [kW]")
    print("Daily sun hours = " + str(sum(irradiance.get_irradiance()) / len(irradiance.get_irradiance()) * 24))
    plt.savefig(figure_map + "production_profile_day1.pdf")
    # start = time.time()
    # processes = 6
    # system.run_battery_parallel(processes, need_battery_profile=False)
    # endd = time.time()
    # print("Time with " + str(processes) + " processes: " + str(endd - start))
    # start = time.time()
    # processes = 5
    # system.run_battery_parallel(processes, need_battery_profile=False)
    # endd = time.time()
    # print("Time with " + str(processes) + " processes: " + str(endd - start))
    # start = time.time()
    # processes = 4
    # system.run_battery_parallel(processes, need_battery_profile=False)
    # endd = time.time()
    # print("Time with " + str(processes) + " processes: " + str(endd - start))
    # start = time.time()
    # processes = 3
    # system.run_battery_parallel(processes, need_battery_profile=False)
    # endd = time.time()
    # print("Time with " + str(processes) + " processes: " + str(endd - start))
    # start = time.time()
    # processes = 2
    # system.run_battery_parallel(processes, need_battery_profile=False)
    # endd = time.time()
    # print("Time with " + str(processes) + " processes: " + str(endd - start))
    # start = time.time()
    # processes = 1
    # system.run_battery_parallel(processes, need_battery_profile=False)
    # endd = time.time()
    # print("Time with " + str(processes) + " processes: " + str(endd - start))
    # start = time.time()
    # system.simulate_battery_without_profile()
    # endd = time.time()
    # print("Time with for sequential calculation: " + str(endd - start))
    # start = time.time()
    # system.simulate_battery()
    # endd = time.time()
    # print("Time with for sequential calculation with profile: " + str(endd - start))

    energy_from_grid, black_out_time, nr_blackouts = system.simulate_battery()
    print(energy_from_grid, black_out_time, nr_blackouts, daily_cons * 365.25 * 0.2)

    for i in range(1, 13):
        plt.figure()
        print("2005-" + str(i) + "-01" if i > 9 else "2005-0" + str(i) + "-01",
              "2005-" + str(i+1) + "-01" if i > 8 and i < 12 else "2005-0" + str(i+1) + "-01" if i < 12 else "2006-01-01")
        system.plot_battery_profile(datetime.fromisoformat("2005-" + str(i) + "-01" if i > 9 else "2005-0" + str(i) + "-01"),
                                    datetime.fromisoformat("2005-" + str(i+1) + "-01" if i > 8 and i < 12 else "2005-0" + str(i+1) + "-01"
                                                           if i < 12 else "2006-01-01"))
        plt.savefig(figure_map + "battery_profile" + str(i) + ".pdf")
    # plt.figure()
    # system.plot_battery_profile(datetime.fromisoformat("2006-01-01"), datetime.fromisoformat("2006-02-01"))
    # plt.savefig(figure_map + "battery_profile2.pdf")

    # Initialize optimization object
    opt = sys.system_optimization(system,
                                  min_peak_power=3.,
                                  max_peak_power=7.,
                                  min_battery_capacity=6.,
                                  max_battery_capacity=20.,
                                  optimization_threshold=daily_cons * 365.25 * 0.05)

    optimal = opt.find_optimal_system(nr_processes=4)
    print("Optimal solar: " + str(optimal[0]) + " kWp")
    print("Optimal battery: " + str(optimal[1]) + " kWh")
    print("Gives a cost of: " + str(optimal[2]))
    print("And an objective value of: " + str(optimal[3]))

    system.set_peak_power(optimal[0])
    system.set_battery_capacity(optimal[1])

    # Simulate and plot battery profile
    system.simulate_battery()

    plt.figure()
    system.plot_battery_profile(datetime.fromisoformat("2005-01-01"), datetime.fromisoformat("2005-02-01"))
    plt.savefig(figure_map + "battery_profile_jan.pdf")
    system.plot_battery_profile(datetime.fromisoformat("2006-08-01"), datetime.fromisoformat("2006-09-01"))
    plt.savefig(figure_map + "battery_profile_aug.pdf")

    opt.set_min_peak_power(0.8 * optimal[0])
    opt.set_max_peak_power(1.2 * optimal[0])
    opt.set_min_battery_capacity(0.8 * optimal[1])
    opt.set_max_battery_capacity(1.2 * optimal[1])
    opt.brute_force_pareto(steps_battery_capacity=3, steps_peak_power=3, nr_processes=4)

    pareto = opt.get_pareto()
    print(pareto)

    fig, ax = plt.subplots()
    opt.plot_pareto(fig, ax)

    plt.savefig(figure_map + "pareto_boundary.pdf")

