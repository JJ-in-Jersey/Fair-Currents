from scipy.signal import savgol_filter
from datetime import timedelta, datetime
from pathlib import Path
from pandas import concat, to_datetime, Series, Timestamp
from numpy import ndarray
from typing import Hashable

from tt_dataframe.dataframe import DataFrame
from tt_file_tools.file_tools import print_file_exists
from tt_geometry.geometry import Arc, StartArc, EndArc
from tt_job_manager.job_manager import Job
from tt_date_time_tools.date_time_tools import hours_mins
from tt_gpx.gpx import Route, Segment
from tt_globals.globals import PresetGlobals as pg
from tt_exceptions.exceptions import DateMismatch, LengthMismatch


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
            super().__init__(data=DataFrame(csv_source=tts_path))
            self.Time = to_datetime(self.Time, utc=True)
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
            self.Time = to_datetime(self.Time, utc=True)
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
            for col in [c for c in self.columns if 'time' in c]:
                self[col] = self[col].apply(to_datetime)
        else:
            # create a list of minima frames (TF = True, below midline) and larger than the noise threshold
            blocks = [df.reset_index(drop=True).drop(labels=['GL', 'block', 'midline'], axis=1)
                  for index, df in savgol_frame.groupby('block') if df['GL'].any() and len(df) > MinimaFrame.noise_threshold]

            frame = DataFrame(columns=['start_time', 'min_time', 'end_time', 'start_duration', 'min_duration', 'end_duration'])
            for i, df in enumerate(blocks):
                median_stamp = df[df.t_time == df.min().t_time]['stamp'].median().astype(int)
                frame.at[i, 'start_utc'] = df.iloc[0].Time
                frame.at[i, 'min_utc'] = df.iloc[abs(df.stamp - median_stamp).idxmin()].Time
                frame.at[i, 'end_utc'] = df.iloc[-1].Time
                frame.at[i, 'start_duration'] = hours_mins(df.iloc[0].t_time * pg.timestep)
                frame.at[i, 'min_duration'] = hours_mins(df.iloc[abs(df.stamp - median_stamp).idxmin()].t_time * pg.timestep)
                frame.at[i, 'end_duration'] = hours_mins(df.iloc[-1].t_time * pg.timestep)

            # all eastern timezone rounded to 15 minutes, for now!
            frame.start_time =  to_datetime(frame.start_utc, utc=True).dt.tz_convert('US/Eastern').round('15min')
            frame.min_time = to_datetime(frame.min_utc, utc=True).dt.tz_convert('US/Eastern').round('15min')
            frame.end_time = to_datetime(frame.end_utc, utc=True).dt.tz_convert('US/Eastern').round('15min')

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
        arguments = tuple([savgol_frame, route.filepath('minima', speed)])
        super().__init__(job_name, result_key, MinimaFrame, arguments)


class ArcsFrame(DataFrame):

    @staticmethod
    def create_subordinate_arc(arc_class: type[StartArc] | type[EndArc], row_dict: dict, index: Hashable):
        try:
            return arc_class(**row_dict)
        except LengthMismatch as lmm:
            print(index, lmm)
            return None
        except Exception as e:
            print(index, e)
            return None

    def __init__(self, minima_frame: DataFrame, arc_path: Path, speed: int, first_day: datetime.date, last_day: datetime.date):

        if arc_path.exists():
            super().__init__(DataFrame(csv_source=arc_path))
            for col in [c for c in self.columns if 'time' in c]:
                self[col] = self[col].apply(to_datetime)
        else:
            arcs = []
            for i, row in minima_frame.iterrows():
                row_dict = row.to_dict()
                try:
                    arcs.append(Arc(**row_dict))
                except DateMismatch:
                    start_arc = self.create_subordinate_arc(StartArc, row_dict, i)
                    if start_arc is not None:
                        arcs.append(start_arc)

                    end_arc = self.create_subordinate_arc(EndArc, row_dict, i)
                    if end_arc is not None:
                        arcs.append(end_arc)
                except LengthMismatch as lmm:
                    print(i, lmm)
                except Exception as e:
                    print(i, e)

            frame = DataFrame([arc.arc_dict for arc in arcs])
            frame.insert(0, 'date', frame.start_time.apply(lambda timestamp: timestamp.date()))
            frame = frame.sort_values(by=['date', 'start_time']).reset_index(drop=True)
            frame.insert(0, 'idx', frame.groupby('date').cumcount() + 1)
            frame.insert(1, 'speed', speed)

            frame = frame[(frame['date'] >= first_day) & (frame['date'] <= last_day)]
            frame.reset_index(drop=True, inplace=True)

            frame.write(arc_path)
            super().__init__(frame)

    # for col in ['start_round', 'min_round', 'end_round']:
    #     arcs_df['str_' + col] = None
    #     for row in range(len(arcs_df)):
    #         arcs_df.loc[row, 'str_' + col] = arcs_df.loc[row, col].strftime("%I:%M %p") if arcs_df.loc[row, col] is not None else None


class ArcsJob(Job):  # super -> job name, result key, function/object, arguments

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, minima_frame: DataFrame, route: Route, speed: int):
        job_name = 'arcs' + ' ' + str(speed)
        result_key = speed
        year = minima_frame.loc[0]['start_time'].year
        first_day = to_datetime(Route.template_dict['first_day'].substitute({'year': year})).date()
        last_day = to_datetime(Route.template_dict['last_day'].substitute({'year': year+2})).date()
        arguments = tuple([minima_frame, route.filepath('arcs', speed), speed, first_day, last_day])
        super().__init__(job_name, result_key, ArcsFrame, arguments)


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

    print(f'\nGenerating arcs frame')
    results = {speed: job_manager.get_result(speed) for speed in keys}  # {'speed': frame}
    keys = [job_manager.submit_job(ArcsJob(frame, route, speed)) for speed, frame in results.items()]
    job_manager.wait()

    print(f'\nAggregating transit times', flush=True)
    frames = [DataFrame(csv_source=route.filepath('arcs', speed)) for speed in pg.speeds]
    transit_times_df = concat(frames).reset_index(drop=True)
    transit_times_df = transit_times_df.fillna("-")

    transit_times_path = route.folder.joinpath(route.template_dict['transit times'].substitute({'loc': route.code}))
    print_file_exists(transit_times_df.write(transit_times_path))
