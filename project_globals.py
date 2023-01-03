from pathlib import Path
from os import environ, makedirs, umask, remove
import shutil
import pandas as pd
import numpy as np
import dateparser as dp
from datetime import timedelta as td
import warnings
from pickle import HIGHEST_PROTOCOL

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TIMESTEP = 15  # seconds
TIME_RESOLUTION = 15  # rounded to minutes
WINDOW_MARGIN = 10  # minutes
TIMESTEP_MARGIN = WINDOW_MARGIN * 60 / TIMESTEP

boat_speeds = [v for v in range(-9, -1, 2)]+[v for v in range(3, 10, 2)]  # knots
# boat_speeds = [v for v in range(-3, -1, 2)]+[v for v in range(3, 4, 2)]  # knots
shared_columns = ['departure_index', 'departure_time']

def semaphore_on(name): open(Path(environ['TEMP']).joinpath(name).with_suffix('.tmp'), 'w').close()
def semaphore_off(name): remove(Path(environ['TEMP']).joinpath(name).with_suffix('.tmp'))
def is_semaphore_set(name): return True if Path(environ['TEMP']).joinpath(name).with_suffix('.tmp').exists() else False

def sign(value): return value/abs(value)
def seconds(start, end): return int((end-start).total_seconds())
def dash_to_zero(value): return 0.0 if str(value).strip() == '-' else value
def rounded_to_minutes(time):
    basis = dp.parse('1/1/2020')
    total_minutes = int((time - basis).total_seconds())/60
    rounded_seconds = round(total_minutes/TIME_RESOLUTION)*TIME_RESOLUTION*60
    return basis + td(seconds=rounded_seconds)
def write_df_csv(df, path): df.to_csv(path.with_suffix('.csv'), index=False)
def write_df_pkl(df, path): df.to_pickle(path.with_suffix('.pkl'), protocol=HIGHEST_PROTOCOL)
def write_df_hdf(df, path): df.to_hdf(path.with_suffix('.hdf'), key='gonzo', mode='w', index=False)
def read_df_csv(path): return pd.read_csv(path.with_suffix('.csv'), header='infer')
def read_df_pkl(path): return pd.read_pickle(path.with_suffix('.pkl'))
def read_df_hdf(path): return pd.read_hdf(path.with_suffix('.hdf'))
def read_df(path):
    if path.with_suffix('.csv').exists(): return read_df_csv(path)
    elif path.with_suffix('.pkl').exists(): return read_df_pkl(path)
    elif path.with_suffix('.hdf').exists(): return read_df_hdf(path)
    else: print('Unrecognizable extension')
def write_df(df, path, extension):
    if extension == 'csv': write_df_csv(df, path)
    elif extension == 'pkl': write_df_pkl(df, path)
    elif extension == 'hdf': write_df_hdf(df, path)
    else: print('Unrecognizable extension')
def read_arr(path):
    if path.with_suffix('.npy').exists(): return np.load(path.with_suffix('.npy'))
def write_arr(arr, path): np.save(path, arr, allow_pickle=False)
def read_list(path):
    if path.with_suffix('.npy').exists(): return list(np.load(path.with_suffix('.npy')))
def write_list(lst, path): write_arr(lst, path)
def output_file_exists(path): return True if path.with_suffix('.csv').exists() or path.with_suffix('.pkl').exists() or path.with_suffix('.hdf').exists() or path.with_suffix('.npy').exists() else False
def hours_min(time):
    if isinstance(time, td): return "%d:%02d" % (time.seconds // 3600, time.seconds % 3600)
    elif isinstance(time, float): return "%d:%02d" % (time // 3600, time % 3600)
    else: print('Unrecognizable time unit')
def min_sec(time):
    if isinstance(time, td): return "%d:%02d" % (time.seconds // 60, time.seconds % 60)
    elif isinstance(time, float): return "%d:%02d" % (time // 60, time % 60)
    else: print('Unrecognizable time unit')

class Environment:

    def create_node_folder(self, name):
        node_folder = self.velocity_folder().joinpath(name)
        makedirs(node_folder, exist_ok=True)
        return node_folder
    def create_edge_folder(self, name):
        edge_folder = self.elapsed_time_folder().joinpath(name)
        makedirs(edge_folder, exist_ok=True)
        return edge_folder
    def project_folder(self, args=None):
        if args:
            self.__project_folder = Path(self.__user_profile + '/Downloads/' + args['project_name']+'/')
            if args['delete_data']:
                shutil.rmtree(self.__project_folder, ignore_errors=True)
                makedirs(self.__project_folder, exist_ok=True)
                return self.__project_folder
        return self.__project_folder
    def velocity_folder(self):
        if not self.__velocity_folder:
            self.__velocity_folder = self.__project_folder.joinpath('Velocity')
            makedirs(self.__velocity_folder, exist_ok=True)
        return self.__velocity_folder
    def elapsed_time_folder(self):
        if not self.__elapsed_time_folder:
            self.__elapsed_time_folder = self.__project_folder.joinpath('Elapsed Time')
            makedirs(self.__elapsed_time_folder, exist_ok=True)
        return self.__elapsed_time_folder
    def transit_time_folder(self):
        if not self.__transit_time_folder:
            self.__transit_time_folder = self.__project_folder.joinpath('Transit Time')
            makedirs(self.__transit_time_folder, exist_ok=True)
        return self.__transit_time_folder
    def user_profile(self): return self.__user_profile

    def __init__(self):
        self.__project_folder = None
        self.__velocity_folder = None
        self.__elapsed_time_folder = None
        self.__transit_time_folder = None
        self.__user_profile = environ['USERPROFILE']
        umask(0)

class ChartYear:

    def set_year(self, args):
        if not self.__year:
            self.__year = args['year']
            self.__first_date = '1/1/'+str(self.__year)
            self.__last_date = '12/31/' + str(self.__year)
            self.__first_day = dp.parse(self.__first_date)
            self.__first_day_minus_one = self.__first_day - td(days=1)
            self.__last_day = dp.parse(self.__last_date)
            self.__last_day_plus_one = self.__last_day + td(days=1)
            self.__last_day_plus_two = self.__last_day + td(days=2)
            self.__last_day_plus_three = self.__last_day + td(days=3)
            self.__index_basis = self.__first_day_minus_one

    def year(self): return self.__year
    def first_day_minus_one(self): return self.__first_day_minus_one
    def first_day(self): return self.__first_day
    def last_day(self): return self.__last_day
    def last_day_plus_one(self): return self.__last_day_plus_one
    def last_day_plus_two(self): return self.__last_day_plus_two
    def last_day_plus_three(self): return self.__last_day_plus_three
    def time_to_index(self, time): return seconds(self.__index_basis, time)
    def index_to_time(self, index): return self.__index_basis + td(seconds=index)
    def index_basis(self): return self.__index_basis

    def __init__(self):
        self.__year = None
        self.__first_date = None
        self.__last_date = None
        self.__first_day_minus_one = None
        self.__first_day = None
        self.__last_day = None
        self.__last_day_plus_one = None
        self.__last_day_plus_two = None
        self.__last_day_plus_three = None
        self.__index_basis = None
