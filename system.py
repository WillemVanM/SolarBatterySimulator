import tkinter
from tkinter import filedialog
import csv
from datetime import datetime
from datetime import time
from datetime import date
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.optimize as sco
import multiprocessing as mpc
import time as tm
# from scipy.optimize import *


class solar_irradiance:
    """
    The solar irradiance controls the properties of the irradiance. It contains the time series and the irradiance data.
    """
    def __init__(self, time_series=None, irradiance=None, size_of_data=None, period=None):
        self.time_series = time_series
        self.irradiance = irradiance
        self.size_of_data = size_of_data
        self.period = period

    def load_csv_file(self,
                      file_path: str = None,
                      irradiance_name: str = None,
                      time_series_name: str = "period_start",
                      delimiter: str = ",",
                      row_nr_names: int = 0,
                      row_nr_start: int = 1,
                      date_time_sep: bool = False,
                      date_series_name: str = "# Date",
                      multiplication: float = 1,
                      ):
        """
        Reads the irradiance and the time from a csv file and converts it to kW.
        :param file_path: path to the file with the irradiance data [W]. If None, the file explorer will be oppened
        to select a file.
        :param irradiance_name: name of the column that gives the irradiance (if known in advance)
        :param time_series_name: name of the column that gives the time series (if known in advance)
        """
        if file_path is None:
            # Open file dialog and select csv to read the data from
            tkinter.Tk().withdraw()  # prevents an empty tkinter window from appearing
            print("Select weather data file")
            file_path = filedialog.askopenfilename()
            print("Selected file path: " + file_path)

        time_series = []

        # Count lines and initiate irrad numpy array with the irradiation values
        with open(file_path) as csv_file:
            if row_nr_start < 1e10:
                size_lines = sum(1 for line in csv.reader(csv_file, delimiter=delimiter))
            else:
                row_nr_start = 0
                found_start = False
                size_lines = 0
                for line in csv.reader(csv_file, delimiter=delimiter):
                    if found_start and len(line) == 0:
                        break
                    elif len(line) == 0 or line[0][0] == "#":
                        row_nr_start += 1
                    elif not found_start:
                        found_start = True
                    size_lines += 1
                row_nr_names = row_nr_start - 1
            irradiance = np.zeros(size_lines - row_nr_start)

        with open(file_path) as csv_file:
            line_count = 0
            csv_reader = csv.reader(csv_file, delimiter=delimiter)

            # Read line by line
            for row in csv_reader:
                # if line_count < row_nr_names:
                #     line_count = row_nr_names
                # print(line_count, row)
                if line_count == row_nr_names:
                    # Check in which columns the correct data is present, from the column names
                    first_try = True
                    while time_series_name not in row:
                        if not first_try:
                            print("Wrong input, try again.")
                        print(f'Column names are {", ".join(row)}')
                        time_series_name = input("Name of the column that gives the time: ")
                        first_try = False

                    first_try = True
                    while irradiance_name not in row:
                        if not first_try:
                            print("Wrong input, try again.")
                        print(f'Column names are {", ".join(row)}')
                        irradiance_name = input("Name of the column that gives the irradiance: ")
                        first_try = False

                    first_try = True
                    while date_series_name not in row:
                        if not first_try:
                            print("Wrong input, try again.")
                        print(f'Column names are {", ".join(row)}')
                        date_series_name = input("Name of the column that gives the date: ")
                        first_try = False

                    # Store the index of the correct columns
                    irradiance_index = row.index(irradiance_name)
                    time_series_index = row.index(time_series_name)
                    if date_time_sep:
                        date_series_index = row.index(date_series_name)


                # elif line_count < row_nr_start:
                #     line_count = row_nr_start

                elif line_count >= row_nr_start and line_count < size_lines:
                    # Read the irradiation data
                    if row[irradiance_index] == "":
                        irradiance[line_count - row_nr_start] = 0
                    else:
                        irradiance[line_count - row_nr_start] = float(row[irradiance_index]) / 1000 * multiplication


                    # Read timestamp
                    add_day = 0
                    if date_time_sep:
                        date_stamp = row[date_series_index]
                        time_stamp = row[time_series_index]
                        if '/' in date_stamp:
                            date_sep = date_stamp.split('/')
                            if len(date_sep[2]) > len(date_sep[0]):
                                date_stamp = date_sep[2] + "-" + [x if len(x) > 1 else "0" + x for x in [date_sep[1]]][0] \
                                             + "-" + [x if len(x) > 1 else "0" + x for x in [date_sep[0]]][0]
                        time_sep = time_stamp.split(':')
                        if len(time_sep) < 3:
                            time_sep.append("00")
                        if len(time_sep[0]) < 2:
                            time_sep[0] = "0" + time_sep[0]
                        elif time_sep[0] == "24":
                            time_sep[0] = "00"
                            add_day = 1
                        time_stamp = ":".join(time_sep)

                        datetime_stamp = date_stamp + "T" + time_stamp
                    else:
                        datetime_stamp = row[time_series_index]
                    # Make sure the format is readable by datetime
                    if ".0000000" in datetime_stamp:
                        datetime_stamp = "".join(datetime_stamp.split(".0000000"))
                    if datetime_stamp[-3] == ":" and (datetime_stamp[-5] == "+" or datetime_stamp[-5] == "-"):
                        datetime_stamp = datetime_stamp[:-4] + "0" + datetime_stamp[-4:]
                    if datetime_stamp[-1] == "Z":
                        datetime_stamp = datetime_stamp[:-1]
                    time_series.append(datetime.fromisoformat(datetime_stamp) + timedelta(days=add_day))

                line_count += 1

        self.irradiance = irradiance
        self.time_series = time_series
        self.size_of_data = size_lines - row_nr_start
        difference = time_series[1] - time_series[0]
        self.period = int(difference.total_seconds() / 60)

    def get_size(self):
        """
        :return: length of the data
        """
        return self.size_of_data

    def get_period(self):
        """
        :return: period between two data elements
        """
        return self.period

    def get_value(self, index):
        """
        :return: value of the irradiation at the given index
        """
        return self.irradiance[index]


    def get_irradiance(self):
        """
        :return: irradiance list
        """
        return self.irradiance


    def get_time_series(self):
        """
        :return: time_series list
        """
        return self.time_series

    def set_size(self):
        """
        :return: calculate size
        """
        self.size_of_data = np.size(self.time_series)
        return


    def split_irradiances(self, nr_processes):
        """
        Splits the solar_irradiance object into an amount of smaller solar_irradiance objects equal to nr_processes.
        The data is split up into equal parts.
        """
        period = self.get_period()
        # Distribute the DAYS as equally as possible
        size = int(np.ceil(self.size_of_data / nr_processes / (60 / period * 24)) * (60 / period * 24))

        irradiances = [solar_irradiance(time_series=self.time_series[i*size:(i + 1)*size],
                                 irradiance=self.irradiance[i*size:(i + 1)*size],
                                 period=period) for i in range(nr_processes)]
        for irradiance in irradiances:
            irradiance.set_size()

        return irradiances


class consumption:
    """
    The consumption object contains a function to calculate the power consumption based on different parameters.
    """
    def __init__(self, consumption=None, time_input=None, size_of_data=None, period=None):
        self.consumption = None
        self.time_input = None
        self.size_of_data = None
        self.period = None

    def consumption_determined_by_time_from_file(self,
                                                 file_path = None,
                                                 consumption_name: str = "Power [kW]",
                                                 time_series_name: str = "Time",
                                                 delimiter: str = ",",
                                                 time_input: str = "Time"
                                                 ):
        """
        Returns a consumption function that only depends on the time.
        The input file should have 1 column, with the consumption data [kW] with the same time difference between every
        element as the irradiation data. The consumption data will be repeated periodically.
        The time input is either a time or datetime.
        """
        if file_path is None:
            # Open file dialog and select csv to read the data from
            print("Select consumption file")
            file_path = filedialog.askopenfilename()
            print("Selected file path: " + file_path)

        consumption_data = []
        time_series = []

        with open(file_path) as csv_file:
            line_count = 0
            csv_reader = csv.reader(csv_file, delimiter=delimiter)
            # Read line by line
            for row in csv_reader:

                if line_count == 0:
                    # Check in which columns the correct data is present, from the column names
                    first_try = True
                    while time_series_name not in row:
                        if not first_try:
                            print("Wrong input, try again.")
                        print(f'Column names are {", ".join(row)}')
                        time_series_name = input("Name of the column that gives the time: ")
                        first_try = False

                    first_try = True
                    while consumption_name not in row:
                        if not first_try:
                            print("Wrong input, try again.")
                        print(f'Column names are {", ".join(row)}')
                        irradiance_name = input("Name of the column that gives the consumption: ")
                        first_try = False

                    # Store the index of the correct columns
                    consumption_index = row.index(consumption_name)
                    time_series_index = row.index(time_series_name)

                    line_count += 1

                else:
                    # Read the irradiation data
                    if len(row[time_series_index]) == 0:
                        break
                    else:
                        consumption_data.append(float(row[consumption_index]))


                    # Read timestamp
                    if (time_input == "Time" or time_input == "time"):
                        time_input = "time"
                        time_stamp = row[time_series_index]
                        if time_stamp[1] == ":":
                            # Make sure the format is readable by time
                            time_stamp = "0" + time_stamp
                        time_series.append(time.fromisoformat(time_stamp))

                    elif (time_input == "Datetime" or time_input == "datetime"):
                        time_input = "datetime"
                        datetime_stamp = row[time_series_index]
                        # Make sure the format is readable by datetime
                        if ".0000000" in datetime_stamp:
                            datetime_stamp = "".join(datetime_stamp.split(".0000000"))
                        if datetime_stamp[-3] == ":" and (datetime_stamp[-5] == "+" or datetime_stamp[-5] == "-"):
                            datetime_stamp = datetime_stamp[:-4] + "0" + datetime_stamp[-4:]
                        time_series.append(datetime.fromisoformat(datetime_stamp))

                    else:
                        ValueError("Incorrect time_input. Should be 'time' or 'datetime'.")
                    line_count += 1

        self.consumption = consumption_data
        self.time_series = time_series
        self.time_input = time_input
        self.size_of_data = line_count - 1
        difference = (datetime.combine(date.today(), time_series[1]) - datetime.combine(date.today(), time_series[0]))
        self.period = int(difference.total_seconds() / 60)


    def get_value(self, datetime):
        """
        :param datetime:
        :return:
        """
        index, ratio = self.search_date(datetime)

        if ratio < 0 or ratio > 1:
            ValueError("datetime should be within time series of consumption object")

        return self.consumption[index] * (1 - ratio) + \
               self.consumption[0 if index+1==self.size_of_data else index+1] * ratio


    def get_period(self):
        """
        :return: period between two data elements
        """
        return self.period


    def get_value_by_index(self, index):
        """
        :param datetime:
        :return:
        """
        index = index % self.size_of_data

        return self.consumption[index]


    def search_date(self, datetime_obj):
        """
        :param datetime_obj: datetime object that needs to be searched in self.time_series
        :return: the index of the index in the largest value of time_series that is smaller than datetime_obj
        and the ratio for linear interpolation between the found index and index + 1
        """
        if self.time_input == "time":
            datetime_obj = datetime_obj.time()

        i_low = 0
        i_high = self.size_of_data
        i = self.size_of_data // 2
        go = True

        while i_high > i_low + 1:
            if datetime_obj < self.time_series[i]:
                i_high = i
                i = (i + i_low) // 2
            elif datetime_obj > self.time_series[i]:
                i_low = i
                i = (i + i_high) // 2
            else:
                i_high = i + 1
                i_low = i

        if i_high == self.size_of_data:
            i_high = 0

        if self.time_input == "time":
            ratio = (datetime.combine(date.today(), datetime_obj) -
                     datetime.combine(date.today(), self.time_series[i_low])) / (
                        datetime.combine(date.today(), self.time_series[i_high])
                        - datetime.combine(date.today(), self.time_series[i_low]))
        else:
            ratio = (datetime_obj - self.time_series[i_low]) / (
                        self.time_series[i_high] - self.time_series[i_low])

        return i_low, ratio


    def consumption_from_consumers(
            self,
            consumers: dict):
        """
        Calculate the consumption profile from all the consumers
        :param consumers: consumer objects
        """
        self.consumption = np.zeros(int(24*3600 / next(iter(consumers)).period.total_seconds()))
        for consumer_type, number in consumers.items():
            for i in range(number):
                consumer_type.calculate_power_consumption()
                self.consumption += consumer_type.power_profile.consumption / 1000

        self.time_series = next(iter(consumers)).power_profile.time_series
        self.time_input = "Time"
        self.size_of_data = np.size(self.consumption)
        difference = (datetime.combine(date.today(), self.time_series[1]) -
                      datetime.combine(date.today(), self.time_series[0]))
        self.period = int(difference.total_seconds() / 60)


class consumer:
    """
    Object that defines an eletrical consumer and
    """

    def __init__(
            self,
            name: str="",
            power = None,
            times = None,
            power_profile = None,
            period: timedelta = timedelta(minutes=15.0),
            randomization: bool = False,
            random_switch_on_time = None,
            random_times = None,
            random_on_proportion: float = 0.0
            ):
        """
        :param name: name of the consumer. Standard values available for LED, incandescent, fridge, freezer, laptop,
        dektop_computer, smartphone, ventilator, printer, oven and air-conditioning.
        :param power: power consumption [W]
        :param times: list of tuples of switch on and switch off times during the day
        :param power_profile: consumption object with times and power profile of the consumer
        :param period: period of data
        :param randomization: True if the consumption is calculated randomly
        :param random_switch_on_time: time that the consumer remains switched on
        :param random_times: list of tuples: start and end times that it is possible that the consumer is turned on,
        with probability equal to random_on_proportion.
        :param random_on_proportion: probability that the consumer is turned on, during the random_times
        """
        self.name = name
        self.power = power
        self.times = times
        self.power_profile = power_profile
        self.period = period
        self.randomization = randomization
        self.random_switch_on_time = random_switch_on_time
        self.random_times = random_times
        self.random_on_proportion = random_on_proportion

        if name == "LED":
            if self.power is None:
                self.power = 5
            if self.times is None and randomization is False:
                self.times = [(time(hour=17), time(hour=23, minute=59, second=59, microsecond=999999))]

        elif name == "incandescent":
            if self.power is None:
                self.power = 50
            if self.times is None and randomization is False:
                self.times = [(time(hour=17), time(hour=23, minute=59, second=59, microsecond=999999))]

        elif name == "security-light":
            if self.power is None:
                self.power = 40
            if self.times is None and randomization is False:
                self.times = [(time(hour=0), time(hour=7,)),
                              (time(hour=18), time(hour=23, minute=59, second=59, microsecond=999999))]

        elif name == "fridge":
            if self.power is None:
                self.power = 100
            if self.times is None and self.power_profile is None:
                self.randomization = True
                self.random_switch_on_time = timedelta(minutes=15)
                self.random_times = [(time(hour=0), time(hour=23, minute=59, second=59, microsecond=999999))]
                self.random_on_proportion = 1/3

        elif name == "freezer":
            if self.power is None:
                self.power = 100
            if self.times is None and self.power_profile is None:
                self.randomization = True
                self.random_switch_on_time = timedelta(minutes=15)
                self.random_times = [(time(hour=0), time(hour=23, minute=59, second=59, microsecond=999999))]
                self.random_on_proportion = 1/3

        elif name == "laptop":
            if self.power is None:
                self.power = 40
            if self.times is None and self.power_profile is None:
                self.randomization = True
                self.random_switch_on_time = timedelta(hours=2)
                self.random_times = [(time(hour=8), time(hour=22))]
                self.random_on_proportion = 1/3.5

        elif name == "desktop_computer":
            if self.power is None:
                self.power = 200
            if self.times is None and self.power_profile is None:
                self.randomization = True
                self.random_switch_on_time = timedelta(hours=2)
                self.random_times = [(time(hour=8), time(hour=22))]
                self.random_on_proportion = 1/3.5

        elif name == "desktop_computer2":
            if self.power is None:
                self.power = 200
            if self.times is None and randomization is False:
                self.times = [(time(hour=8, minute=30), time(hour=9, minute=15)),
                              (time(hour=10), time(hour=10, minute=45)),
                              (time(hour=14), time(hour=14, minute=45))]

        elif name == "projector":
            if self.power is None:
                self.power = 300
            if self.times is None and randomization is False:
                self.times = [(time(hour=8), time(hour=12, )),
                              (time(hour=13), time(hour=18))]

        elif name == "wifi":
            if self.power is None:
                self.power = 20
            if self.times is None and randomization is False:
                self.times = [(time(hour=0), time(hour=23, minute=59, second=59))]

        elif name == "smartphone":
            if self.power is None:
                self.power = 5
            if self.times is None and self.power_profile is None:
                self.randomization = True
                self.random_switch_on_time = timedelta(hours=1, minutes=30)
                # self.random_times = [(time(hour=20), time(hour=23, minute=59, second=59, microsecond=999999)),
                #                      (time(hour=0), time(hour=2))]
                self.random_times = [(time(hour=8), time(hour=18))]
                self.random_on_proportion = 0.25

        if name == "ventilator":
            if self.power is None:
                self.power = 80
            if self.times is None and randomization is False:
                self.times = [(time(hour=11), time(hour=15))]

        elif name == "printer":
            if self.power is None:
                self.power = 100
            if self.times is None and self.power_profile is None:
                self.randomization = True
                self.random_switch_on_time = timedelta(minutes=30)
                self.random_times = [(time(hour=8), time(hour=18))]
                self.random_on_proportion = 1/20

        elif name == "oven":
            if self.power is None:
                self.power = 1500
            if self.times is None and randomization is False:
                self.times = [(time(hour=18, minute=30), time(hour=19, minute=30))]

        elif name == "microwave":
            if self.power is None:
                self.power = 900
            if self.times is None and randomization is False:
                self.times = [(time(hour=18, minute=30), time(hour=19, minute=30))]

        elif name == "air-conditioning":
            if self.power is None:
                self.power = 700
            if self.times is None and randomization is False:
                self.times = [(time(hour=10, minute=30), time(hour=15, minute=30))]

        elif name == "TV":
            if self.power is None:
                self.power = 60
            if self.times is None and randomization is False:
                self.times = [(time(hour=17, minute=0), time(hour=23, minute=0))]

        elif name == "washing_machine":
            if self.power is None:
                self.power = 300
            if self.times is None and randomization is False:
                self.times = [(time(hour=12, minute=0), time(hour=14, minute=30))]


    def calculate_power_consumption(self):
        """
        Calculates self.power_profile based on the other characteristics
        """
        self.power_profile = consumption()

        if self.randomization:
            size_data = int(timedelta(hours=24).total_seconds() / self.period.total_seconds())

            total_energy = 0
            today = date.today()

            total_time_available = sum(map(lambda x: (datetime.combine(today, x[1])
                             - datetime.combine(today, x[0])).total_seconds(), self.random_times))

            required_total_energy = self.power * self.random_on_proportion * total_time_available


            # Keep trying until the total energy consumption is acceptable
            attempts = 0
            while (total_energy < 0.9975**(attempts/5)*required_total_energy or
                   total_energy > 1.005**(attempts/5) * required_total_energy):
                time_series = list([0] * size_data)
                power_consumption = np.zeros(size_data)
                possibly_on = False

                i = 0
                while i < size_data:
                    time_series[i] = time(hour=int(self.period.total_seconds() * i // 3660),
                                      minute=int(self.period.total_seconds() * i // 60 % 60))

                    # Check if the consumer can be turned on at this time of the day.
                    if time_series[i] in map(lambda x: x[0], self.random_times):
                        possibly_on = True
                    elif time_series[i] in map(lambda x: x[1], self.random_times):
                        possibly_on = False

                    if possibly_on:
                        # randomly select whether the consumer is switched on
                        x = random.random()
                        if x < 1 - (1 - self.random_on_proportion)\
                                **(self.period.total_seconds() / self.random_switch_on_time.total_seconds()):
                            j = i
                            while i < j + int(self.random_switch_on_time.total_seconds() / self.period.total_seconds())\
                                    and i < size_data:
                                time_series[i] = time(hour=int(self.period.total_seconds() * i // 3660),
                                      minute=int(self.period.total_seconds() * i // 60 % 60))
                                power_consumption[i] = self.power
                                i += 1
                        else:
                            # consumer is switched off
                            power_consumption[i] = 0.0
                            i += 1

                    else:
                        # consumer cannot be switched on at this time
                        power_consumption[i] = 0.0
                        i += 1

                total_energy = sum(power_consumption) * self.period.total_seconds()
                attempts += 1

        else:
            size_data = int(timedelta(hours=24).total_seconds() / self.period.total_seconds())
            time_series = list([0] * size_data)
            power_consumption = np.zeros(size_data)
            switched_on = False

            for i in range(size_data):
                time_series[i] = time(hour=int(self.period.total_seconds() * i // 3600),
                                      minute=int(self.period.total_seconds() * i // 60 % 60))
                # check if consumer is switched on at this time
                if time_series[i] in map(lambda x: x[0], self.times):
                    switched_on = True
                elif time_series[i] in map(lambda x: x[1], self.times):
                    switched_on = False

                if switched_on:
                    power_consumption[i] = self.power
                else:
                    power_consumption[i] = 0

        self.power_profile.time_series = time_series
        self.power_profile.consumption = power_consumption


class system:
    """
    The `system` [class](#classes) contains all the attributes that you can set to
    control various parameters of the solar and battery computation.
    """

    def __init__(
        self,
        peak_power: float = 0,
        price_solar: float = 800,
        battery_capacity: float = 0,
        battery_efficiency: float = 92,
        charging_efficiency: float = None,
        discharging_efficiency: float = None,
        DOD: float = 80,
        price_battery: float = 300,
        solar_irradiance = None,
        consumption = None,
        battery_profile = None,
        fixed_cost = 0,
        energy_from_grid = None,
        black_out_time = None,
        black_outs = None,
        go = True
    ):
        """
        :param peak_power: the peak power of the solar system [kWp]
        :param price_solar: price of solar panels [€/kWp] (or any other currency)
        :param battery_capacity: the energetic capacity of the battery pack [kWh]
        :param battery_efficiency: one-way efficiency of the battery for both charging and discharging [%]
        :param charging_efficiency: charging efficiency of the battery; 100 % - losses during charging [%]
        :param discharging_efficiency: discharging efficiency of the battery; 100 % - losses during discharging [%]
        :param DOD: depth of discharge = 100 % - minimal battery charge
        :param price_battery: price of batteries [€/kWh] (or any other currency)
        :param solar_irradiance: solar irradiance object
        :param consumption: consumption object
        """
        self.peak_power = peak_power
        self.battery_capacity = battery_capacity
        self.price_solar = price_solar
        if charging_efficiency is None:
            self.charging_efficiency = battery_efficiency
        else:
            self.charging_efficiency = charging_efficiency
        if discharging_efficiency is None:
            self.discharging_efficiency = battery_efficiency
        else:
            self.discharging_efficiency = discharging_efficiency
        self.DOD = DOD
        self.price_battery = price_battery
        self.solar_irradiance = solar_irradiance
        self.consumption = consumption
        self.battery_profile = battery_profile
        self.fixed_cost = fixed_cost
        self.energy_from_grid = energy_from_grid
        self.black_out_time = black_out_time
        self.black_outs = black_outs
        self.go = go


    def set_peak_power(self, peak_power):
        """
        :param peak_power: sets the peak power of the system to peak_power [kWp]
        """
        self.peak_power = peak_power


    def set_battery_capacity(self, battery_capacity):
        """
        :param battery_capacity: sets the battery capacity of the system to battery_capacity [kWh]
        """
        self.battery_capacity = battery_capacity


    def simulate_battery(self):
        """
        Calculates the battery profile over time.
        """
        # print("Simulate battery started")
        size_of_data = self.solar_irradiance.get_size()
        period = self.solar_irradiance.get_period() # in minutes
        consumption_period = self.consumption.get_period() # in minutes
        current_datetime = self.solar_irradiance.time_series[0]

        if period == consumption_period:
            simplified_consumption_evaluation = True
        else:
            simplified_consumption_evaluation = False

        self.battery_profile = np.zeros(size_of_data)

        # Initializing values
        self.black_outs = []
        self.black_out_time = 0
        self.energy_from_grid = 0
        self.minimal_battery_profile = 100
        blacked_out = False
        next_profile = self._battery_step(100, 0, current_datetime, period, simplified_consumption_evaluation)
        if next_profile is None:  # No data
            self.battery_profile[0] = 100
        else:
            self.battery_profile[0] = next_profile

        for i in range(1, size_of_data):
            current_datetime = current_datetime + timedelta(minutes=period)
            next_profile = self._battery_step(self.battery_profile[i-1], i, current_datetime, period,
                                                         simplified_consumption_evaluation)
            if next_profile is None:  # No data
                self.battery_profile[i] = max(self.battery_profile[i-1], 100 - self.DOD + 0.0000001)
            else:
                self.battery_profile[i] = next_profile

            if self.battery_profile[i] < 100 - self.DOD:
                self.energy_from_grid += self.battery_capacity / 100 * (100 - self.DOD - self.battery_profile[i])
                self.battery_profile[i] = 100 - self.DOD
                self.black_out_time = self.black_out_time + period / 60
                if not blacked_out:
                    blacked_out = True
                    self.black_outs.append(datetime)

            else:
                blacked_out = False

            if self.battery_profile[i] < self.minimal_battery_profile:
                self.minimal_battery_profile = self.battery_profile[i]

        # print("Simulate battery ended")
        return [self.energy_from_grid / size_of_data * (24 * 60 * 365.25 / period),
                self.black_out_time / size_of_data * (24 * 60 * 365.25 / period),
                len(self.black_outs) / size_of_data * (24 * 60 * 365.25 / period)]


    def simulate_battery_without_profile(self, data_range=None):
        """
        Calculates the energy needed from the grid (if hybrid connected), time of blackouts (if standalone) and
        the amount of blackouts that have occured (if standalone) in the given data range
        :param data_range: Sequence with start index and stop index that should be used.
        """
        # print("Simulate battery started")
        if data_range is None:
            data_range = [0, self.solar_irradiance.get_size()]

        size_of_data = data_range[1] - data_range[0]
        period = self.solar_irradiance.get_period() # in minutes
        consumption_period = self.consumption.get_period() # in minutes
        current_datetime = self.solar_irradiance.time_series[data_range[0]]

        if period == consumption_period:
            simplified_consumption_evaluation = True
        else:
            simplified_consumption_evaluation = False

        # Initializing values
        nr_black_outs = 0
        self.black_out_time = 0
        self.energy_from_grid = 0
        self.minimal_battery_profile = 100
        blacked_out = False
        battery_charge = self._battery_step(100, 0, current_datetime, period, simplified_consumption_evaluation)
        if battery_charge is None:  # No data
            battery_charge = 100

        for i in range(1, size_of_data):
            current_datetime = current_datetime + timedelta(minutes=period)
            battery_charge = self._battery_step(battery_charge, i, current_datetime, period,
                                                         simplified_consumption_evaluation)
            if battery_charge is None:  # No data
                battery_charge = max(self.battery_profile[i-1], 100 - self.DOD + 0.0000001)

            if battery_charge < 100 - self.DOD:
                self.energy_from_grid += self.battery_capacity / 100 * (100 - self.DOD - battery_charge)
                battery_charge = 100 - self.DOD
                self.black_out_time = self.black_out_time + period / 60
                if not blacked_out:
                    blacked_out = True
                    nr_black_outs += 1

            else:
                blacked_out = False

            if battery_charge < self.minimal_battery_profile:
                self.minimal_battery_profile = battery_charge

        # print("Simulate battery ended")
        return [self.energy_from_grid / size_of_data * (24 * 60 * 365.25 / period),
                self.black_out_time / size_of_data * (24 * 60 * 365.25 / period),
                nr_black_outs / size_of_data * (24 * 60 * 365.25 / period)]


    def _battery_step(self, current_value, index, datetime, period, simplified_consumption_evaluation):
        """
        :param current_value: value of battery percentage at datetime
        :param datetime: current datetime
        :param period: period of a battery step [minutes]
        :return: value of battery percentage at datetime + period
        """
        generation = self.peak_power * self.solar_irradiance.get_value(index) * period / 60
        if generation < 0:  # no data
            return None
        if simplified_consumption_evaluation:
            consumption = self.consumption.get_value_by_index(index) * period / 60
        else:
            consumption = self.consumption.get_value(datetime) * period / 60

        net_generation = generation - consumption

        if net_generation > 0:
            return min(current_value + net_generation * (self.charging_efficiency/100) / self.battery_capacity * 100, 100)
        else:
            return current_value + net_generation / (self.discharging_efficiency/100) / self.battery_capacity * 100


    def child_process_battery(self, results, battery_profile, data_range=None, need_battery_profile=True):
        """
        Used for parallelization. Runs the given part of the battery profile.
        :param results: Empty Mulitprocessing Array to save results
        :param battery_profile: Empty Mulitprocessing Array to save battery profile
        :param need_battery_profile: Indicates whether we need to calculate the entire battery profile or merely the
            results
        :return: None
        """
        # time1 = tm.time()
        if need_battery_profile:
            results_array = self.simulate_battery()
        else:
            results_array = self.simulate_battery_without_profile(data_range=data_range)
        # print("Simulation time: " + str(tm.time() - time1))

        for i in range(len(results) - 1):
            results[i] = results_array[i]

        results[-1] = self.minimal_battery_profile

        if need_battery_profile:
            battery_profile_array = self.battery_profile
            for i in range(len(battery_profile_array)):
                battery_profile[i] = battery_profile_array[i]

        return


    def run_battery_parallel(self, nr_processes, need_battery_profile=True):
        """
        Splits the system in a number of subsystems and simulates those in parallel. The results can change very
        slightly compared to the results of a sequential process. Namely, at the start of every subsystem the charge is
        set equal to 100 %. As most profiles equilibrate after a day, the introduced error is very small in most
        practical systems.
        :param nr_processes: Determines the amount of subsystems that are executed in parallel
        :param need_battery_profile: Indicates whether we need to calculate the entire battery profile or merely the
            results
        :return: Results: energy required from the grid (if hybrid connected), time of blackouts (if standalone) and
            the amount of blackouts that have occured (if standalone)
        """
        # time0 = tm.time()
        if need_battery_profile:
            systems = self._split_systems(nr_processes)
            sizes = [systems[i].solar_irradiance.get_size() for i in range(nr_processes)]
        else:
            period = self.solar_irradiance.get_period()
            # Distribute the DAYS as equally as possible
            size = int(np.ceil(self.solar_irradiance.size_of_data / nr_processes / (60 / period * 24)) * (60 / period * 24))
            sizes = [size if i < nr_processes - 1 else self.solar_irradiance.size_of_data - size * (nr_processes - 1)
                     for i in range(nr_processes)]

        results = [mpc.Array('f', 4) for j in range(nr_processes)]
        if need_battery_profile:
            battery_profiles = [mpc.Array('f', sizes[i]) for i in range(nr_processes)]

        # print("Prepare simulations: " + str(tm.time() - time0))

        if need_battery_profile:
            processes = [mpc.Process(target=system.child_process_battery,
                                     args=(systems[i], results[i], battery_profiles[i], None, need_battery_profile))
                         for i in range(nr_processes)]
        else:
            processes = [mpc.Process(target=system.child_process_battery,
                                     args=(self, results[i], None, [sum(sizes[:i]), sum(sizes[:i+1])],
                                           need_battery_profile))
                         for i in range(nr_processes)]

        # time1 = tm.time()
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # print("Process execution: " + str(tm.time() - time1))

        self.energy_from_grid = sum(map(lambda result: result[0], results))
        self.black_out_time = sum(map(lambda result: result[1], results))
        len_black_outs = sum(map(lambda result: result[2], results))
        self.minimal_battery_profile = min(map(lambda result: result[3], results))

        if need_battery_profile:
            self.battery_profile = []
            for i in range(nr_processes):
                self.battery_profile.extend(battery_profiles[i])

        return [self.energy_from_grid / self.solar_irradiance.get_size() * (24 * 60 * 365.25 / period),
                self.black_out_time / self.solar_irradiance.get_size() * (24 * 60 * 365.25 / period),
                len_black_outs / self.solar_irradiance.get_size() * (24 * 60 * 365.25 / period)]





    def _split_systems(self, nr_processes):
        size_of_data = self.solar_irradiance.get_size()
        time1 = tm.time()
        solar_irradiances = self.solar_irradiance.split_irradiances(nr_processes)
        print("Split irradiances: " + str(tm.time() - time1))
        return [system(peak_power=self.peak_power,
                            price_solar = self.price_solar,
                            battery_capacity=self.battery_capacity,
                            charging_efficiency = self.charging_efficiency,
                            discharging_efficiency = self.discharging_efficiency,
                            DOD = self.DOD,
                            price_battery = self.price_battery,
                            solar_irradiance = solar_irradiances[i],
                            consumption = self.consumption,
                            battery_profile = None if self.battery_profile is None else \
                                self.battery_profile[i*int(np.ceil(size_of_data / nr_processes)):
                                                    (i + 1)*int(np.ceil(size_of_data / nr_processes))],
                            fixed_cost = self.fixed_cost)
                   for i in range(nr_processes)]


    def plot_battery_profile(self, start_datetime, end_datetime):
        """
        Plots the battery profile between the given start and end datetime
        :param start_datetime: datetime object from which the battery profile is plotted
        :param end_datetime: datetime object until which the battery profile is plotted
        """
        plt.plot(self.solar_irradiance.time_series, self.battery_profile)
        plt.plot([self.solar_irradiance.time_series[0],
                  self.solar_irradiance.time_series[self.solar_irradiance.size_of_data - 1]],
                 [100 - self.DOD, 100 - self.DOD], '-.')
        plt.xlim([start_datetime, end_datetime])

        plt.xlabel("Time")
        plt.ylabel("Battery percentage [%]")


    def get_total_cost(self):
        """
        :return: system cost
        """
        return self.fixed_cost + self.price_solar * self.peak_power + \
               self.price_battery * self.battery_capacity


class system_optimization:
    """
    Object to optimize the amount of solar panels and batteries for the given system.
    """
    def __init__(self,
                 system=None,
                 min_peak_power: float=0,
                 max_peak_power: float=100,
                 min_battery_capacity: float=0,
                 max_battery_capacity: float=100,
                 optimization_objective: str = "energy_from_grid",
                 optimization_threshold: float = 520):
        """
        :param system: system object that needs to be optimized
        :param min_peak_power: minimum restriction of the peak power that can be installed [kWp]
        :param max_peak_power: maximum restriction of the peak power that can be installed [kWp]
        :param min_battery_capacity: minimum restriction of the battery capacity that can be installed [kWh]
        :param max_battery_capacity: maximum restriction of the battery capacity that can be installed [kWh]
        """
        self.system = system
        self.min_peak_power = min_peak_power
        self.max_peak_power = max_peak_power
        self.min_battery_capacity = min_battery_capacity
        self.max_battery_capacity = max_battery_capacity
        self.optimization_threshold = optimization_threshold
        self.pareto = None
        self.visible_annotation = set()

        if optimization_objective != "energy_from_grid" and optimization_objective != "black_out_time" and \
            optimization_objective != "number_of_black_outs":
            ValueError("Invalid optimization_objective")
        else:
            self.optimization_objective = optimization_objective


    def set_min_peak_power(self, min_peak_power):
        """
        :param min_peak_power: sets the minimal peak power of the optimization object to min_peak_power [kWp]
        """
        self.min_peak_power = min_peak_power


    def set_max_peak_power(self, max_peak_power):
        """
        :param min_peak_power: sets the maximal peak power of the optimization object to max_peak_power [kWp]
        """
        self.max_peak_power = max_peak_power


    def set_min_battery_capacity(self, min_battery_capacity):
        """
        :param min_peak_power: sets the minimal battery capacity of the optimization object to min_battery_capacity
        [kWh]
        """
        self.min_battery_capacity = min_battery_capacity


    def set_max_battery_capacity(self, max_battery_capacity):
        """
        :param min_peak_power: sets the maximal battery capacity of the optimization object to max_battery_capacity
        [kWp]
        """
        self.max_battery_capacity = max_battery_capacity


    def brute_force_pareto(self, steps_peak_power=5, steps_battery_capacity=5, nr_processes=1):
        """
        :param steps:
        :return:
        """
        peak_powers = np.linspace(self.min_peak_power, self.max_peak_power, steps_peak_power)
        battery_capacities = np.linspace(self.min_battery_capacity, self.max_battery_capacity, steps_battery_capacity)
        self.pareto = []

        for peak_power in peak_powers:
            self.system.set_peak_power(peak_power)
            print(peak_power)
            for battery_capacity in battery_capacities:
                self.system.set_battery_capacity(battery_capacity)
                if nr_processes == 1:
                    self.system.simulate_battery()
                else:
                    self.system.run_battery_parallel(nr_processes=nr_processes, need_battery_profile=False)
                self.update_pareto()

        self.pareto = np.array(self.pareto)


    def find_optimal_system(self, x=None, max_iterations=50, ftol=1e-2, nr_processes=1):
        """
        Finds optimal system based on the given optimization objective and threshold.
        """
        from gaussian_process import find_zero
        if self.optimization_objective == "energy_from_grid":
            obj = 0
        elif self.optimization_objective == "black_out_time":
            obj = 1
        elif self.optimization_objective == "number_of_black_outs":
            obj = 2
        else:
            ValueError("Invalid optimization_objective")

        lb = self.min_peak_power * self.system.price_solar + self.min_battery_capacity * self.system.price_battery \
            + self.system.fixed_cost
        ub = self.max_peak_power * self.system.price_solar + self.max_battery_capacity * self.system.price_battery \
             + self.system.fixed_cost

        upper_limit = self.optimal_system_for_cost(ub, obj, f_tol=0.1, nr_processes=nr_processes)[0]
        lower_limt = self.optimal_system_for_cost(lb, obj, f_tol=0.1, nr_processes=nr_processes)[0]

        print(lower_limt, upper_limit, self.optimization_threshold)

        if lower_limt < self.optimization_threshold:
            print("Too high minimal values given to achieve optimal result")
            ValueError("Too high minimal values given to achieve optimal result")
            exit()
        if upper_limit > self.optimization_threshold:
            print("Too low maximal values given to achieve optimal result")
            ValueError("Too low maximal values given to achieve optimal result")
            exit()

        iteration = 0
        if x is None:
            # cost = (lb + ub) / 2
            # new_cost = lb + (lower_limt - self.optimization_threshold) / (lower_limt - upper_limit) * (ub - lb)
            new_cost = min(lb * 1.1 + 10, lb + (lower_limt - self.optimization_threshold) / (lower_limt - upper_limit) * (ub - lb))
        else:
            new_cost = x

        last_cost = lb
        last_value = lower_limt
        optimization_value = self.optimal_system_for_cost(new_cost, obj, nr_processes=nr_processes)
        new_value = optimization_value[0]

        X = [lb, ub, new_cost]
        Y = [lower_limt, upper_limit, new_value]

        plt.figure()
        costs = [last_cost, new_cost]
        values = [last_value, new_value]
        closeness = abs(new_value - self.optimization_threshold) / self.optimization_threshold

        while closeness > ftol and iteration < max_iterations:
            print(new_cost, lb, ub, new_value, (new_value - self.optimization_threshold) / self.optimization_threshold)

            cost = new_cost
            # if True: #iteration == 0:
            # secant method
            # print(new_value)
            if new_value - last_value == 0:
                print(new_value, lb, self.optimization_threshold)

            new_cost = find_zero(X, np.array(Y) - self.optimization_threshold)
            X.append(new_cost)
            # if (new_cost - last_cost) * (new_value - last_value) > 0:
            #     new_cost = (new_cost + last_cost) / 2
            # else:
            #     new_cost = max(new_cost + (new_cost - last_cost) / (new_value - last_value) *
            #                    (self.optimization_threshold - new_value), lb)


            # else:
            #     # Tiruneh method
            #     new_cost = last_last_cost - last_last_value * (new_value - last_value) / \
            #                ((new_value - last_last_value) / (new_cost - last_last_cost) * (new_value - last_value) -
            #                  new_value * (new_value - last_last_value) / (new_cost - last_last_cost) -
            #                 (last_value - last_last_value) / (last_cost - last_last_cost))

            # last_last_cost = last_cost
            last_cost = cost
            print("Tolerance = " + str(min(ftol*(1 + closeness), 0.1)))
            optimization_value = self.optimal_system_for_cost(new_cost, obj, f_tol=min(ftol*(1 + closeness), 0.1),
                                                              nr_processes=nr_processes)
            # last_last_value = last_value
            last_value = new_value
            new_value = optimization_value[0]
            Y.append(new_value)

            costs.append(new_cost)
            values.append(new_value)
            # plt.scatter(costs, values)

            # plt.pause(0.001)

            iteration += 1

            closeness = abs(new_value - self.optimization_threshold) / self.optimization_threshold

        solar = optimization_value[1]
        batteries = (new_cost - self.system.fixed_cost - self.system.price_solar * optimization_value[1]) \
                    / self.system.price_battery

        return (solar, batteries, new_cost, optimization_value[0])



    def optimal_system_for_cost(self, cost, obj=0, x=None, max_iterations=10, f_tol=1e-2, nr_processes=1):

        def objective_function(self, x, cost, obj):#, g):
            self.system.set_peak_power(x[0])
            battery_capacity = (cost - self.system.price_solar * x[0] - self.system.fixed_cost) \
                               / self.system.price_battery
            self.system.set_battery_capacity(battery_capacity)
            if nr_processes == 1:
                objective = self.system.simulate_battery()
            else:
                objective = self.system.run_battery_parallel(nr_processes=nr_processes, need_battery_profile=False)

            if objective[obj] == 0:
                new_obj = -self.system.minimal_battery_profile
                return new_obj
            else:
                return objective[obj]

        x_max = (cost - self.system.fixed_cost) / self.system.price_solar * 0.99
        if x is None:
            x0 = x_max / 3
        else:
            x0 = min(x, x_max)

        bounds = sco.Bounds([0], [x_max])
        res = sco.minimize(lambda xi: objective_function(self, xi, cost, obj), x0=x0, tol=f_tol,
                           bounds=bounds,
                           options={'maxiter': max_iterations})

        print("optimal x-ratio: " + str(float(res.x) / x_max))

        return (res.fun, float(res.x))


    def update_pareto(self):
        """
        Updates self.pareto with the results for the latest simulation of self.system
        """
        cost = self.system.get_total_cost()

        if self.optimization_objective == "energy_from_grid":
            result = [cost, self.system.energy_from_grid, self.system.peak_power, self.system.battery_capacity]
        elif self.optimization_objective == "black_out_time":
            result = [cost, self.system.black_out_time, self.system.peak_power, self.system.battery_capacity]
        elif self.optimization_objective == "number_of_black_outs":
            result = [cost, len(self.system.black_outs), self.system.peak_power, self.system.battery_capacity]
        else:
            ValueError("Invalid optimization_objective")

        size_pareto = len(self.pareto)
        add = True

        i = 0
        while add and i < size_pareto:
            # checks if any element in self.pareto is pareto dominated by result
            if result[0] <= self.pareto[i][0] and result[1] <= self.pareto[i][1]:
                self.pareto[i] = result
                add = False
                j = i + 1
                while j < size_pareto:
                    if result[0] <= self.pareto[j][0] and result[1] <= self.pareto[j][1]:
                        self.pareto.pop(j)
                        size_pareto -= size_pareto
                    j += 1

            # check if result is pareto dominated by any element in self.pareto
            elif result[0] >= self.pareto[i][0] and result[1] >= self.pareto[i][1]:
                add = False

            i += 1

        if add: # result is not pareto dominated by and does not pareto dominate any element in self.pareto
            self.pareto.append(result)


    def get_pareto(self):
        """
        :return: the pareto boundary
        """
        return self.pareto


    def plot_pareto(self, fig=None, ax=None, currency = "€"):
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        sc = plt.scatter(self.pareto[:, 0], self.pareto[:, 1])
        plt.xlabel("Cost [" + currency + "]")

        if self.optimization_objective == "energy_from_grid":
            plt.ylabel("Energy from grid [kWh]")
        elif self.optimization_objective == "black_out_time":
            plt.ylabel("Black-out time [hours]")
        elif self.optimization_objective == "number_of_black_outs":
            plt.ylabel("Number of black-outs")
        else:
            ValueError("Invalid optimization_objective")

        # Add annotations showing the peak power and the battery capacity of each data point
        annotations = [ax.annotate("PV: " + str((self.pareto[i, 2])) + " kWp; Bat: " + str(self.pareto[i, 3]) + " kWh",
                            xy=sc.get_offsets()[i], xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"), visible=False,
                            arrowprops=dict(arrowstyle="->")) for i in range(np.size(self.pareto, 0))]


        def update_annot(ind):
            """
            Sets the clicked and the hoovered annotations visible
            :param ind: contains the indices that are hovered over
            """
            indices = set(ind["ind"])
            indices = indices.union(self.visible_annotation)

            for index in range(np.size(self.pareto, 0)):
                if index in indices:
                    annotations[index].set_visible(True)
                else:
                    annotations[index].set_visible(False)


        def hover(event):
            """
            By moving the cursor, the annotations are updated, based on which are hovered over and which clicked or
            un-clicked.
            """
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                update_annot(ind)
                fig.canvas.draw_idle()


        def click(event):
            """
            Keeps track of clicked and un-clicked annotations.
            """
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    indices = ind["ind"]
                    for index in indices:
                        if index in self.visible_annotation:
                            self.visible_annotation.remove(index)
                        else:
                            self.visible_annotation.add(index)


        fig.canvas.mpl_connect('motion_notify_event', hover) # 'key_press_event'
        fig.canvas.mpl_connect('button_press_event', click)



