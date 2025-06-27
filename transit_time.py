from scipy.signal import savgol_filter
from datetime import timedelta
from dateparser import parse as parsedate
from pathlib import Path
from pandas import concat, to_datetime, Series
from numpy import ndarray

from tt_dataframe.dataframe import DataFrame
from tt_file_tools.file_tools import print_file_exists
from tt_geometry.geometry import Arc
from tt_job_manager.job_manager import Job
from tt_date_time_tools.date_time_tools import hours_mins
from tt_gpx.gpx import Route, Segment
from tt_globals.globals import PresetGlobals as pg
from tt_exceptions.exceptions import IndexGreaterThanThree


class TimeStepsFrame(DataFrame):

    @staticmethod
    def total_transit_timesteps(init_row: int, ets_values: ndarray, array_indices: list):
        max_row = len(ets_values) - 1
        transit_time = 0
        for idx in array_indices:
            row = int(transit_time) + init_row
            if row > max_row:
                break
            transit_time += ets_values[row, idx]
        return transit_time

    def __init__(self, speed: int, route: Route):

        tts_path = route.filepath('transit timesteps', speed)
        ets_path = route.filepath('elapsed timesteps', speed)

        if tts_path.exists():
            frame = DataFrame(csv_source=tts_path)
        else:
            if not ets_path.exists():
                raise FileExistsError(ets_path)

            frame = DataFrame(csv_source=ets_path)
            frame.Time = to_datetime(frame.Time, utc=True)

            # pass values and indices to improve performance
            indices = [frame.columns.get_loc(c) for c in frame.columns.to_list() if Segment.prefix in c]
            values = frame.values
            transit_timesteps_arr = [self.total_transit_timesteps(row, values, indices) for row in range(len(frame))]

            frame.drop(frame.columns[indices], axis=1, inplace=True)
            frame['t_time'] = Series(transit_timesteps_arr)
            frame.write(tts_path)

        super().__init__(data=frame)


class TimeStepsJob(Job):

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, speed: int, route: Route):
        job_name = 'transit times' + ' ' + str(speed)
        result_key = speed
        arguments = tuple([speed, route])
        super().__init__(job_name, result_key, TimeStepsFrame, arguments)


class SavGolFrame(DataFrame):

    savgol_size = 1100
    savgol_order = 1

    def __init__(self, timesteps_frame: DataFrame, savgol_path: Path):

        if savgol_path.exists():
            super().__init__(data=DataFrame(csv_source=savgol_path))
        else:
            timesteps_frame['midline'] = savgol_filter(timesteps_frame.t_time, self.savgol_size, self.savgol_order).round().astype('int')
            timesteps_frame = timesteps_frame[timesteps_frame.t_time.ne(timesteps_frame.midline)].copy()  # remove values that equal the midline
            timesteps_frame.loc[timesteps_frame.t_time.lt(timesteps_frame.midline), 'GL'] = True  # less than midline = false
            timesteps_frame.loc[timesteps_frame.t_time.gt(timesteps_frame.midline), 'GL'] = False  # greater than midline => true
            timesteps_frame['block'] = (timesteps_frame['GL'] != timesteps_frame['GL'].shift(1)).cumsum()  # index the blocks of True and False
            timesteps_frame.write(savgol_path)
            super().__init__(data=timesteps_frame)


class SavGolJob(Job):

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, timesteps_frame: DataFrame, speed: int, route: Route):
        job_name = 'savitzky-golay midlines' + ' ' + str(speed)
        result_key = speed
        arguments = tuple([timesteps_frame, route.filepath('savgol', speed)])
        super().__init__(job_name, result_key, SavGolFrame, arguments)


class MinimaFrame(DataFrame):

    noise_threshold = 100

    # def __init__(self, transit_timesteps_path: Path, savgol_path: Path):
    def __init__(self, savgol_frame: DataFrame, minima_path: Path):

        if minima_path.exists():
            super().__init__(data=DataFrame(csv_source=minima_path))
        else:
            # create a list of minima frames (TF = True, below midline) and larger than the noise threshold
            blocks = [df.reset_index(drop=True).drop(labels=['GL', 'block', 'midline'], axis=1)
                  for index, df in savgol_frame.groupby('block') if df['GL'].any() and len(df) > MinimaFrame.noise_threshold]

            frame = DataFrame(columns=['start_eastern_round', 'min_eastern_round', 'end_eastern_round'])
            for i, df in enumerate(blocks):
                median_stamp = df[df.t_time == df.min().t_time]['stamp'].median().astype(int)
                frame.at[i, 'start_utc'] = df.iloc[0].Time
                frame.at[i, 'end_utc'] = df.iloc[-1].Time
                frame.at[i, 'min_utc'] = df.iloc[abs(df.stamp - median_stamp).idxmin()].Time
                frame.at[i, 'start_tt'] = hours_mins(df.iloc[0].t_time * pg.timestep)
                frame.at[i, 'end_tt'] = hours_mins(df.iloc[-1].t_time * pg.timestep)
                frame.at[i, 'min_tt'] = hours_mins(df.iloc[abs(df.stamp - median_stamp).idxmin()].t_time * pg.timestep)

            frame['start_eastern_round'] =  to_datetime(frame.start_utc, utc=True).dt.tz_convert('US/Eastern').round('15min')
            frame['min_eastern_round'] = to_datetime(frame.min_utc, utc=True).dt.tz_convert('US/Eastern').round('15min')
            frame['end_eastern_round'] = to_datetime(frame.end_utc, utc=True).dt.tz_convert('US/Eastern').round('15min')

            frame.drop(['start_utc', 'min_utc', 'end_utc'], axis=1, inplace=True)

            frame.write(minima_path)
            super().__init__(data=frame)


class MinimaJob(Job):  # super -> job name, result key, function/object, arguments

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, savgol_frame: DataFrame, speed: int, route: Route):
        job_name = 'minima' + ' ' + str(speed)
        result_key = speed
        arguments = tuple([speed, route])
        arguments = tuple([savgol_frame, route.filepath('minima', speed)])
        super().__init__(job_name, result_key, MinimaFrame, arguments)


def index_arc_df(frame):
    output_frame = DataFrame(columns=['idx'] + frame.columns.to_list())

    date_arr_dict = {key: [] for key in sorted(list(set(frame['date'])))}

    for i, row in frame.iterrows():
        date_arr_dict[row['date']].append(row)

    for key in date_arr_dict.keys():
        for i in range(len(date_arr_dict[key])):
            output_frame.loc[len(output_frame)] = [i + 1] + date_arr_dict[key][i].tolist()
        if len(date_arr_dict[key]) == 2:
            last_row_dict = output_frame.loc[len(output_frame) - 1].to_dict()
            new_dict = {key: None for key in last_row_dict}
            new_dict.update({'idx': 3, 'date': last_row_dict['date'], 'speed': last_row_dict['speed']})
            output_frame.loc[len(output_frame)] = new_dict

    if len(output_frame[output_frame.idx > 3]):
        raise IndexGreaterThanThree('Index greater than 3 for speed ' + str(frame.speed[0]))

    return output_frame


def create_arcs(minima_path: Path, speed: int):
    minima_frame = read_df(minima_path)
    for col in MinimaFrame.frame_time_columns:
        minima_frame[col] = to_datetime(minima_frame[col])
    start_year = to_datetime(minima_frame.start_eastern[0]).year
    first_day = parsedate(Route.template_dict['first_day'].substitute({'year': start_year}))
    last_day = parsedate(Route.template_dict['last_day'].substitute({'year': start_year+2}))

    arcs = [Arc(row.to_dict()) for i, row in minima_frame.iterrows()]
    next_day_arcs = [arc.next_day_arc for arc in arcs if arc.next_day_arc]
    all_arcs = arcs + next_day_arcs
    all_good_arcs = [arc for arc in all_arcs if not arc.zero_angle]

    arcs_df = DataFrame(columns=Arc.columns)
    for d in [a.arc_dict for a in all_good_arcs]:
        arcs_df.loc[len(arcs_df)] = d

    # noinspection PyTypeChecker
    arcs_df.insert(loc=0, column='date', value=None)
    arcs_df['date'] = arcs_df.start_eastern.apply(lambda x: x.date())
    arcs_df['speed'] = speed

    arcs_df.sort_values(by=['start_eastern'], inplace=True)
    arcs_df = index_arc_df(arcs_df)
    arcs_df = arcs_df[arcs_df.date <= last_day.date()]
    arcs_df = arcs_df[arcs_df.date >= first_day.date()]
    arcs_df.reset_index(drop=True, inplace=True)

    for col in ['start_round', 'min_round', 'end_round']:
        arcs_df['str_' + col] = None
        for row in range(len(arcs_df)):
            arcs_df.loc[row, 'str_' + col] = arcs_df.loc[row, col].strftime("%I:%M %p") if arcs_df.loc[row, col] is not None else None

    return arcs_df


class TransitTimeArcsFrame:

    def __init__(self, speed: int, route: Route):

        minima_path = route.filepath('minima', speed)
        if not print_file_exists(minima_path):
            raise FileExistsError(minima_path)

        arcs_path = route.filepath('arcs', speed)
        if not print_file_exists(arcs_path):
            self.frame = create_arcs(minima_path, speed)
            print_file_exists(write_df(self.frame, arcs_path))





class TransitTimeArcsJob(Job):  # super -> job name, result key, function/object, arguments

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, speed: int, route: Route):
        job_name = 'arcs' + ' ' + str(speed)
        result_key = speed
        arguments = tuple([speed, route])
        super().__init__(job_name, result_key, TransitTimeArcsFrame, arguments)


def transit_time_processing(job_manager, route: Route):

    print(f'\nCalculating transit times')
    keys = [job_manager.submit_job(TimeStepsJob(speed, route)) for speed in pg.speeds]
    job_manager.wait()

    print(f'\nGenerating savgol frames')
    results = {speed: job_manager.get_result(speed) for speed in keys}  # {'speed': frame}
    keys = [job_manager.submit_job(SavGolJob(frame, speed, route)) for speed, frame in results.items()]
    job_manager.wait()

    print(f'\nGenerating minima frames')
    results = {speed: job_manager.get_result(speed) for speed in keys}  # {'speed': frame}
    keys = [job_manager.submit_job(MinimaJob(frame, speed, route)) for speed, frame in results.items()]
    job_manager.wait()

    print(f'\nCalculating transit arcs')
    for speed in pg.speeds:
        job_manager.submit_job(TransitTimeArcsJob(speed, route))
    job_manager.wait()

    print(f'\nAggregating transit times', flush=True)
    frames = [DataFrame(csv_source=route.filepath('arcs', speed)) for speed in pg.speeds]
    transit_times_df = concat(frames).reset_index(drop=True)
    transit_times_df = transit_times_df.fillna("-")

    transit_times_path = route.folder.joinpath(route.template_dict['transit times'].substitute({'loc': route.code}))
    print_file_exists(transit_times_df.write(transit_times_path))
