# C:\Users\jason\PycharmProjects\Transit-Time\venv\Scripts\python.exe C:\Users\jason\PycharmProjects\Transit-Time\main.py "East River" "C:\users\jason\Developer Workspace\GPX\East River West to East.gpx" 2023

import logging
from time import sleep, perf_counter
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from sympy import Point
from pathlib import Path

from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By

from project_globals import WDW, DF_FILE_TYPE, file_exists
from project_globals import mins_secs, date_to_index, index_to_date
import multiprocess as mpm
from ChromeDriver import ChromeDriver as cd
from ReadWrite import ReadWrite as rw
from Navigation import Navigation as nav
from VelocityInterpolation import Interpolator as VI
from FileTools import FileTools as FT

#  VELOCITIES ARE DOWNLOADED, CALCULATED AND SAVE AS NAUTICAL MILES PER HOUR!

logging.getLogger('WDM').setLevel(logging.NOTSET)

def dash_to_zero(value): return 0.0 if str(value).strip() == '-' else value

# noinspection PyProtectedMember
class VelocityJob:

    @staticmethod
    def velocity_download(folder, wdw):
        newest_before = newest_after = FT.newest_file(folder)
        wdw.until(ec.element_to_be_clickable((By.ID, 'generatePDF'))).click()
        while newest_before == newest_after:
            sleep(0.1)
            newest_after = FT.newest_file(folder)
        return newest_after

    def velocity_page(self, y, driver, wdw):
        code_string = 'Annual?id=' + self.wp.code
        wdw.until(ec.element_to_be_clickable((By.CSS_SELECTOR, "a[href*='" + code_string + "']"))).click()
        Select(driver.find_element(By.ID, 'fmt')).select_by_index(3)  # select format
        Select(driver.find_element(By.ID, 'timeunits')).select_by_index(1)  # select 24 hour time
        dropdown = Select(driver.find_element(By.ID, 'year'))
        options = [int(o.text) for o in dropdown.options]
        dropdown.select_by_index(options.index(y))

    def velocity_aggregate(self):
        download_df = pd.DataFrame()
        driver = cd.get_driver(self.wp.folder)
        for y in range(self.year - 1, self.year + 2):  # + 2 because of range behavior
            driver.get(self.wp.noaa_url)
            wdw = WebDriverWait(driver, WDW)
            self.velocity_page(y, driver, wdw)
            downloaded_file = VelocityJob.velocity_download(self.wp.folder, wdw)
            file_df = pd.read_csv(downloaded_file, parse_dates=['Date_Time (LST/LDT)'])
            download_df = pd.concat([download_df, file_df])
        download_df['date_index'] = pd.to_numeric(download_df['Date_Time (LST/LDT)'].apply(date_to_index))
        download_df['velocity'] = download_df[' Speed (knots)'].apply(dash_to_zero)
        driver.quit()
        return download_df

    def __init__(self, year, start_index, end_index, waypoint):
        self.wp = waypoint
        self.year = year.value
        self.start = start_index.value
        self.end = end_index.value

class CurrentStationJob(VelocityJob):

    def execute(self):
        init_time = perf_counter()
        print(f'+     {self.wp.unique_name}', flush=True)

        if file_exists(self.wp.output_data_file):
            return tuple([self.result_key, rw.read_arr(self.wp.output_data_file), init_time])
        else:
            if file_exists(self.wp.interpolation_data_file):
                download_df = rw.read_df_csv(self.wp.interpolation_data_file)
            else:
                download_df = self.velocity_aggregate()
                download_df = download_df[(self.start <= download_df['date_index']) & (download_df['date_index'] <= self.end)]
                rw.write_df_csv(download_df, self.wp.interpolation_data_file)

            temp = list(download_df['date_index'])
            for i, value in enumerate(temp[:-1]):
                if value > temp[i+1]:
                    print(self.wp.unique_name, i, value)

            # create cubic spline
            cs = CubicSpline(download_df['date_index'], download_df['velocity'])

            output_df = pd.DataFrame()
            output_df['date_index'] = self.v_range
            output_df['date_time'] = output_df['date_index'].apply(index_to_date)
            output_df['velocity'] = output_df['date_index'].apply(cs)
            rw.write_df_csv(output_df, self.wp.folder.joinpath(self.wp.unique_name + '_output_as_csv'))

            velo_array = np.array(output_df['velocity'].to_list(), dtype=np.half)
            rw.write_arr(velo_array, self.wp.output_data_file)
            return tuple([self.result_key, velo_array, init_time])

    def execute_callback(self, result):
        print(f'-     {self.wp.unique_name} {mins_secs(perf_counter() - result[2])} minutes', flush=True)

    def error_callback(self, result):
        print(f'!     {self.wp.unique_name} process has raised an error: {result}', flush=True)

    def __init__(self, year, start_index, end_index, waypoint, timestep):
        super().__init__(year, start_index, end_index, waypoint)
        self.result_key = id(waypoint)
        self.v_range = range(self.start, self.end, timestep)

class InterpolationDataJob(CurrentStationJob):

    range_value = 10800  # three hour timestep

    def dataframe(self, velocities):
        download_df = pd.DataFrame()
        download_df['date_index'] = VelocityJob.velocity_range(InterpolationDataJob.range_value)
        download_df['velocity'] = velocities
        rw.write_df_csv(download_df, self.wp.interpolation_data_file)

    def __init__(self, year, start_index, end_index, waypoint):
        super().__init__(year, start_index, end_index, waypoint, InterpolationDataJob.range_value)

class InterpolationJob:

    def execute(self):
        init_time = perf_counter()
        print(f'+     {self.wp.unique_name} {self.index}', flush=True)
        if file_exists(self.wp.output_data_file):
            return tuple([self.result_key, rw.read_df(self.output_data_file), init_time])
        else:
            interpolator = VI(self.surface_points)
            interpolator.set_interpolation_point(self.input_point)
            output = interpolator.get_interpolated_point()
            if self.display:
                interpolator.set_interpolation_point(self.input_point)
                interpolator.show_axes()
            return tuple([self.result_key, output, init_time])

    def execute_callback(self, result):
        print(f'-     {self.wp.unique_name} {self.index} {mins_secs(perf_counter() - result[2])} minutes', flush=True)

    def error_callback(self, result):
        print(f'!     {self.wp.unique_name} process has raised an error: {result}', flush=True)

    def __init__(self, waypoints, index: int, display=False):
        self.display = display
        interpolation_point = waypoints[0]
        self.wp = interpolation_point
        self.index = index
        self.input_point = Point(interpolation_point.lat, interpolation_point.lon, 0)
        self.result_key = str(id(interpolation_point))+'_'+str(index)
        self.surface_points = [Point(wp.lat, wp.lon, wp.output_data[index]) for wp in waypoints[1:]]
