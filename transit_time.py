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


class MinimaFrame:

    frame_time_columns = ['start_utc', 'start_eastern', 'start_round',
                          'min_utc', 'min_eastern', 'min_round',
                          'end_utc', 'end_eastern', 'end_round']
    frame_tts_columns = ['start_tt', 'min_tt', 'end_tt']

    noise_threshold = 100

    def __init__(self, transit_timesteps_path: Path, savgol_path: Path):

        self.frame = DataFrame(columns=MinimaFrame.frame_time_columns + MinimaFrame.frame_tts_columns)

        if not print_file_exists(savgol_path):
            frame = DataFrame(csv_source=transit_timesteps_path)
            frame['test'] = savgol_filter(frame.t_time, pg.savgol_size, pg.savgol_order).round().astype('int')
            frame['midline'] = savgol_filter(frame.t_time, pg.savgol_size, pg.savgol_order).round()
            frame.midline = frame.midline.astype('int')
            frame['TF'] = frame.t_time.lt(frame['midline'])  # Above midline = False,  below midline = True
            frame = frame.drop(frame[frame.t_time == frame['midline']].index).reset_index(drop=True)  # remove values = midline
            frame['block'] = (frame['TF'] != frame['TF'].shift(1)).cumsum()  # index the blocks of True and False
            print_file_exists(frame.write(savgol_path))
        else:
            frame = DataFrame(csv_source=savgol_path)

        # create a list of minima frames (TF = True, below midline) and larger than the noise threshold
        blocks = [df.reset_index(drop=True).drop(labels=['TF', 'block', 'midline'], axis=1)
                  for index, df in frame.groupby('block') if df['TF'].any() and len(df) > MinimaFrame.noise_threshold]

        for i, df in enumerate(blocks):
            median_stamp = df[df.t_time == df.min().t_time]['stamp'].median().astype(int)
            self.frame.at[i, 'start_utc'] = df.iloc[0].Time
            self.frame.at[i, 'end_utc'] = df.iloc[-1].Time
            self.frame.at[i, 'min_utc'] = df.iloc[abs(df.stamp - median_stamp).idxmin()].Time
            self.frame.at[i, 'start_tt'] = hours_mins(df.iloc[0].t_time * pg.timestep)
            self.frame.at[i, 'end_tt'] = hours_mins(df.iloc[-1].t_time * pg.timestep)
            self.frame.at[i, 'min_tt'] = hours_mins(df.iloc[abs(df.stamp - median_stamp).idxmin()].t_time * pg.timestep)

        self.frame.start_utc =  to_datetime(self.frame.start_utc, utc=True)
        self.frame.min_utc = to_datetime(self.frame.min_utc, utc=True)
        self.frame.end_utc = to_datetime(self.frame.end_utc, utc=True)

        self.frame.start_eastern = self.frame.start_utc.dt.tz_convert(tz='US/Eastern')
        self.frame.min_eastern = self.frame.min_utc.dt.tz_convert(tz='US/Eastern')
        self.frame.end_eastern = self.frame.end_utc.dt.tz_convert(tz='US/Eastern')

        self.frame.start_round = self.frame.start_utc.dt.round('15min')
        self.frame.min_round = self.frame.min_utc.dt.round('15min')
        self.frame.end_round = self.frame.end_utc.dt.round('15min')

        self.frame.start_round = self.frame.start_round.dt.tz_convert(tz='US/Eastern')
        self.frame['srutc'] = self.frame.start_utc.dt.tz_convert(tz='US/Eastern').round('15min')
        self.frame.min_round = self.frame.min_round.dt.tz_convert(tz='US/Eastern')
        self.frame.end_round = self.frame.end_round.dt.tz_convert(tz='US/Eastern')


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

        ets_path = route.filepath('elapsed timesteps', speed)
        tts_path = route.filepath('transit timesteps', speed)
        if not ets_path.exists():
            raise FileExistsError(ets_path)

        if not tts_path.exists():
            frame = DataFrame(csv_source=ets_path)
            frame.Time = to_datetime(frame.Time, utc=True)

            # pass values and indices to improve performance
            indices = [frame.columns.get_loc(c) for c in frame.columns.to_list() if Segment.prefix in c]
            values = frame.values
            transit_timesteps_arr = [self.total_transit_timesteps(row, values, indices) for row in range(len(frame))]

            frame.drop(frame.columns[indices], axis=1, inplace=True)
            frame['t_time'] = Series(transit_timesteps_arr)
            frame.write(tts_path)
        else:
            frame = DataFrame(csv_source=tts_path)

        super().__init__(data=frame)


class TransitTimeMinimaFrame:

    def __init__(self, speed: int, route: Route):

        tts_path = route.filepath('transit timesteps', speed)
        if not print_file_exists(tts_path):
            raise FileExistsError(tts_path)

        minima_path = route.filepath('minima', speed)
        if not print_file_exists(minima_path):
            self.frame = MinimaFrame(tts_path, route.filepath('savgol', speed)).frame
            print_file_exists(write_df(self.frame, minima_path))


class TransitTimeArcsFrame:

    def __init__(self, speed: int, route: Route):

        minima_path = route.filepath('minima', speed)
        if not print_file_exists(minima_path):
            raise FileExistsError(minima_path)

        arcs_path = route.filepath('arcs', speed)
        if not print_file_exists(arcs_path):
            self.frame = create_arcs(minima_path, speed)
            print_file_exists(write_df(self.frame, arcs_path))

# class TransitTimeJob(Job):  # super -> job name, result key, function/object, arguments
#
#     def execute(self): return super().execute()
#     def execute_callback(self, result): return super().execute_callback(result)
#     def error_callback(self, result): return super().error_callback(result)
#
#     def __init__(self, speed: int, route: Route):
#         job_name = 'transit_time' + ' ' + str(speed)
#         result_key = speed
#         arguments = tuple([speed, route])
#         super().__init__(job_name, result_key, TransitTimeDataframe, arguments)

class TimeStepsJob(Job):  # super -> job name, result key, function/object, arguments

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, speed: int, route: Route):
        job_name = 'transit times' + ' ' + str(speed)
        result_key = speed
        arguments = tuple([speed, route])
        super().__init__(job_name, result_key, TimeStepsFrame, arguments)


class TransitTimeMinimaJob(Job):  # super -> job name, result key, function/object, arguments

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, speed: int, route: Route):
        job_name = 'minima' + ' ' + str(speed)
        result_key = speed
        arguments = tuple([speed, route])
        super().__init__(job_name, result_key, TransitTimeMinimaFrame, arguments)


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

    # print(f'\nCalculating transit timesteps')
    # # keys = [job_manager.submit_job(TransitTimeJob(speed, route)) for speed in PresetGlobals.speeds]
    # for speed in pg.speeds:
    #     job = TransitTimeJob(speed, route)
    #     result = job.execute()
    # # job_manager.wait()

    print(f'\nCalculating transit times')
    for speed in pg.speeds:
        job_manager.submit_job(TimeStepsJob(speed, route))
    job_manager.wait()

    print(f'\nCalculating transit minima')
    for speed in pg.speeds:
        job_manager.submit_job(TransitTimeMinimaJob(speed, route))
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
