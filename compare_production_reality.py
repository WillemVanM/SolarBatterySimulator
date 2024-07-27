import system as sys
from datetime import timedelta
from matplotlib import pyplot as plt

irradiance = sys.solar_irradiance()
irradiance.load_csv_file(
    file_path="C:/Users/wille/Documents/Humasol/victron_data/csv_13.3147351_-16.6837919_fixed_13_180_PT15M.csv",
    irradiance_name = "gti")

real_PV_to_battery = sys.solar_irradiance()
real_PV_to_battery.load_csv_file(
    file_path="C:/Users/wille/Documents/Humasol/victron_data/Kudimba_kwh_20220801-0000_to_20230803-1726.csv",
    irradiance_name = "PV to battery",
    time_series_name="timestamp",
    delimiter=";",
    row_nr_start=2)

real_PV_to_consumer = sys.solar_irradiance()
real_PV_to_consumer.load_csv_file(
    file_path="C:/Users/wille/Documents/Humasol/victron_data/Kudimba_kwh_20220801-0000_to_20230803-1726.csv",
    irradiance_name = "PV to grid",
    time_series_name="timestamp",
    delimiter=";",
    row_nr_start=2)

peak_power = 3 # kWp
time_zone_difference = 0

plt.figure()
plt.plot(real_PV_to_battery.get_time_series(), (real_PV_to_battery.get_irradiance() +
                                                real_PV_to_consumer.get_irradiance())
         * 60 / real_PV_to_battery.get_period() * 1000 / peak_power, "-.")
plt.plot(list(map(lambda x: x - timedelta(hours=time_zone_difference),
             irradiance.get_time_series())), irradiance.get_irradiance())

plt.show()


