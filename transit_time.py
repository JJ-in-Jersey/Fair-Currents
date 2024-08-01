import pandas as pd
from scipy.signal import savgol_filter
from num2words import num2words
from pathlib import Path
import datetime

from tt_file_tools.file_tools import read_df, write_df, print_file_exists
from tt_geometry.geometry import Arc
from tt_job_manager.job_manager import Job
from tt_date_time_tools.date_time_tools import index_to_date, round_datetime
from tt_globals.globals import Globals

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def none_row(row, df):
    for c in range(len(df.columns)):
        df.iloc[row, c] = None


def total_transit_time(init_row, d_frame, cols):
    row = init_row
    tt = 0
    for col in cols:
        val = d_frame.at[row, col]
        tt += val
        row += val
    return tt


def minima_table(transit_array, template_df, savgol_path):
    noise_size = 100
    min_df = template_df.copy(deep=True)
    min_df = min_df.assign(tts=transit_array)
    if savgol_path.exists():
        min_df['midline'] = read_df(savgol_path)['midline']
    else:
        min_df['midline'] = savgol_filter(transit_array, 50000, 1).round()
        write_df(min_df, savgol_path, True)
    print_file_exists(savgol_path)

    min_df['TF'] = min_df['tts'].lt(min_df['midline'])  # Above midline = False,  below midline = True
    min_df = min_df.drop(min_df[min_df['tts'] == min_df['midline']].index).reset_index(drop=True)  # remove values that equal midline
    min_df['block'] = (min_df['TF'] != min_df['TF'].shift(1)).cumsum()  # index the blocks of True and False
    clump_lookup = {index: df for index, df in min_df.groupby('block') if df['TF'].any()}  # select only the True blocks
    clump_lookup = {index: df.drop(['TF', 'block', 'midline'], axis=1).reset_index() for index, df in clump_lookup.items() if len(df) > noise_size}  # remove the tiny blocks caused by noise at the inflections

    for index, clump in clump_lookup.items():
        # for this clump get the row with the minimum transit time
        median_departure_index = clump[clump['tts'] == clump.min()['tts']]['departure_index'].median()  # median of the departure indices among the tts minimum values
        abs_diff = clump['departure_index'].sub(median_departure_index).abs()  # series of abs differences
        minimum_index = abs_diff[abs_diff == abs_diff.min()].index[0]  # row closest to median departure index
        minimum_row = clump.iloc[minimum_index]

        # find the upper and lower limits of the time window
        offset = int(minimum_row['tts']*Globals.TIME_WINDOW_SCALE_FACTOR)  # offset is transit time steps (tts)

        sr_range = clump[clump['departure_index'].lt(minimum_row['departure_index'])]  # range of valid starting points less than minimum
        if len(sr_range[sr_range['tts'].gt(offset)]):  # there is a valid starting point between the beginning and minimum
            sr = sr_range[sr_range['tts'].gt(offset)].iloc[-1]  # pick the one closest to minimum
        else:
            sr = clump.iloc[0]

        er_range = clump[clump['departure_index'].gt(minimum_row['departure_index'])]  # range of valid ending points greater than minimum
        if len(er_range[er_range['tts'].gt(offset)]):  # there is a valid starting point between the minimum and end
            er = er_range[er_range['tts'].gt(offset)].iloc[0]  # pick the one closest to minimum
        else:
            er = clump.iloc[-1]

        start_row = sr
        end_row = er

        # start_row = clump.iloc[0] if clump.iloc[0]['tts'] < offset else clump[clump['departure_index'].le(minimum_row['departure_index']) & clump['tts'].ge(offset)].iloc[-1]
        # end_row = clump.iloc[-1] if clump.iloc[-1]['tts'] < offset else clump[clump['departure_index'].ge(minimum_row['departure_index']) & clump['tts'].ge(offset)].iloc[0]

        min_df.at[minimum_row['index'], 'start_index'] = start_row['departure_index']
        min_df.at[minimum_row['index'], 'start_datetime'] = index_to_date(start_row['departure_index'])
        min_df.at[minimum_row['index'], 'start_et'] = (datetime.datetime.min + datetime.timedelta(seconds=int(start_row['tts'])*Globals.TIMESTEP)).time()
        min_df.at[minimum_row['index'], 'min_index'] = minimum_row['departure_index']
        min_df.at[minimum_row['index'], 'min_datetime'] = index_to_date(minimum_row['departure_index'])
        min_df.at[minimum_row['index'], 'min_et'] = (datetime.datetime.min + datetime.timedelta(seconds=int(minimum_row['tts'])*Globals.TIMESTEP)).time()
        min_df.at[minimum_row['index'], 'end_index'] = end_row['departure_index']
        min_df.at[minimum_row['index'], 'end_datetime'] = index_to_date(end_row['departure_index'])
        min_df.at[minimum_row['index'], 'end_et'] = (datetime.datetime.min + datetime.timedelta(seconds=int(end_row['tts'])*Globals.TIMESTEP)).time()

    min_df = min_df.dropna(axis=0).sort_index().reset_index(drop=True)  # remove lines with NA

    min_df['start_round_datetime'] = min_df['start_datetime'].apply(round_datetime)
    min_df['min_round_datetime'] = min_df['min_datetime'].apply(round_datetime)
    min_df['end_round_datetime'] = min_df['end_datetime'].apply(round_datetime)
    min_df['start_round_time'] = min_df['start_round_datetime'].dt.time
    min_df['min_round_time'] = min_df['min_round_datetime'].dt.time
    min_df['end_round_time'] = min_df['end_round_datetime'].dt.time
    min_df.drop(['start_round_datetime', 'min_round_datetime', 'end_round_datetime'], axis=1, inplace=True)

    min_df['start_time'] = min_df['start_datetime'].dt.time
    min_df['start_date'] = min_df['start_datetime'].dt.date
    min_df['min_time'] = min_df['min_datetime'].dt.time
    min_df['min_date'] = min_df['min_datetime'].dt.date
    min_df['end_time'] = min_df['end_datetime'].dt.time
    min_df['end_date'] = min_df['end_datetime'].dt.date
    min_df.drop(['start_datetime', 'min_datetime', 'end_datetime'], axis=1, inplace=True)

    min_df.drop(['tts', 'departure_index', 'midline', 'block', 'TF'], axis=1, inplace=True)
    return min_df


def index_arc_df(frame):

    frame_columns = frame.columns.to_list()

    date_keys = [key for key in sorted(list(set(frame['start_date'])))]

    start_time_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    start_round_time_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    start_angle_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    start_round_angle_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    start_et_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}

    min_time_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    min_round_time_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    min_angle_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    min_round_angle_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    min_et_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}

    end_time_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    end_round_time_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    end_angle_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    end_round_angle_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}
    end_et_dict = {key: [] for key in sorted(list(set(frame['start_date'])))}

    for i, row in frame.iterrows():
        start_time_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('start_time')])
        start_round_time_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('start_round_time')])
        start_angle_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('start_angle')])
        start_round_angle_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('start_round_angle')])
        start_et_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('start_et')])

        min_time_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('min_time')])
        min_round_time_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('min_round_time')])
        min_angle_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('min_angle')])
        min_round_angle_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('min_round_angle')])
        min_et_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('min_et')])

        end_time_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('end_time')])
        end_round_time_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('end_round_time')])
        end_angle_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('end_angle')])
        end_round_angle_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('end_round_angle')])
        end_et_dict[row.iloc[frame_columns.index('start_date')]].append(row.iloc[frame_columns.index('end_et')])

    output_columns = [
        'index', 'start_date',
        'start_time', 'start_round_time', 'start_angle', 'start_round_angle', 'start_et',
        'end_time', 'end_round_time', 'end_angle', 'end_round_angle', 'end_et',
        'min_time', 'min_round_time', 'min_angle', 'min_round_angle', 'min_et'
    ]
    output_frame = pd.DataFrame(columns=output_columns)

    for date in date_keys:
        start_times = start_time_dict[date]
        start_round_times = start_round_time_dict[date]
        start_angles = start_angle_dict[date]
        start_round_angles = start_round_angle_dict[date]
        start_ets = start_et_dict[date]

        min_times = min_time_dict[date]
        min_round_times = min_round_time_dict[date]
        min_angles = min_angle_dict[date]
        min_round_angles = min_round_angle_dict[date]
        min_ets = min_et_dict[date]

        end_times = end_time_dict[date]
        end_round_times = end_round_time_dict[date]
        end_angels = end_angle_dict[date]
        end_round_angles = end_round_angle_dict[date]
        end_ets = end_et_dict[date]

        for i in range(len(start_times)):
            output_frame.loc[len(output_frame)] = [
                i+1, date,
                start_times[i], start_round_times[i], start_angles[i], start_round_angles[i], start_ets[i],
                min_times[i], min_round_times[i], min_angles[i], min_round_angles[i],
                min_ets[i], end_times[i], end_round_times[i], end_angels[i], end_round_angles[i], end_ets[i]
            ]

    return output_frame


def create_arcs(f_day, l_day, minima_frame):

    arcs = [Arc(row.to_dict()) for i, row in minima_frame.iterrows()]
    dicts = [a.arc_dict for a in filter(lambda a: not (a.arc_dict['arc_angle'] == 0 or a.arc_dict['arc_round_angle'] == 0), arcs)]

    arcs_df = pd.DataFrame(columns=Arc.columns)
    for d in dicts:
        arcs_df.loc[len(arcs_df)] = d

    arcs_df.sort_values(by=['start_date', 'start_time'], inplace=True)
    arcs_df = index_arc_df(arcs_df)
    arcs_df = arcs_df[arcs_df['start_date'] <= l_day.date()]
    arcs_df = arcs_df[arcs_df['start_date'] >= f_day.date()]

    return arcs_df


class ArcsDataframe:

    def __init__(self, speed, template_df: pd.DataFrame, et_file, tt_folder, f_day, l_day):

        et_df = read_df(et_file)
        row_range = range(len(template_df))

        self.frame = None
        speed_folder = tt_folder.joinpath(num2words(speed))
        transit_timesteps_path = speed_folder.joinpath('timesteps.csv')
        savgol_path = speed_folder.joinpath('savgol.csv')
        minima_path = speed_folder.joinpath('minima.csv')
        self.filepath = speed_folder.joinpath('arcs.csv')

        if not self.filepath.exists():
            if transit_timesteps_path.exists():
                transit_timesteps_arr = list(read_df(transit_timesteps_path)['0'].to_numpy())
            else:
                col_list = et_df.columns.to_list()
                col_list.remove('departure_index')
                col_list.remove('date_time')

                transit_timesteps_arr = [total_transit_time(row, et_df, col_list) for row in row_range]
                write_df(pd.concat([template_df, pd.DataFrame(transit_timesteps_arr)], axis=1), transit_timesteps_path, True)
            print_file_exists(transit_timesteps_path)

            if minima_path.exists():
                minima_df = read_df(minima_path)
            else:
                minima_df = minima_table(transit_timesteps_arr, template_df, savgol_path)
                write_df(minima_df, minima_path, True)
            print_file_exists(minima_path)

            frame = create_arcs(f_day, l_day, minima_df)
            frame['speed'] = speed
            write_df(frame, self.filepath, True)
            print_file_exists(self.filepath)


class TransitTimeJob(Job):  # super -> job name, result key, function/object, arguments

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, speed, et_file: Path, tt_folder: Path):

        job_name = 'transit_time' + ' ' + str(speed)
        result_key = speed
        arguments = tuple([speed, Globals.TEMPLATE_TRANSIT_TIME_DATAFRAME, et_file, tt_folder, Globals.FIRST_DAY, Globals.LAST_DAY])
        super().__init__(job_name, result_key, ArcsDataframe, arguments)
