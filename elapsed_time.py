import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from functools import reduce

from tt_gpx.gpx import Route, Waypoint, Segment
from tt_job_manager.job_manager import Job
from tt_globals.globals import PresetGlobals
from tt_file_tools.file_tools import print_file_exists, read_df, write_df


#  Elapsed times are reported in number of timesteps
def elapsed_time(distance_start_index, distances, length):  # returns number of timesteps
    distance_index = distance_start_index + 1  # distance at departure time is 0
    count = total = 0
    not_there_yet = True
    while not_there_yet:
        total += abs(distances[distance_index])
        count += 1
        distance_index += 1
        not_there_yet = True if length > 0 and total < length else False
    return count  # count = number of time steps


class ElapsedTimeFrame:

    range_offset = 7

    @staticmethod
    def distance(water_vf, water_vi, boat_speed, ts_in_hr):
        return ((water_vf + water_vi) / 2 + boat_speed) * ts_in_hr  # distance is nm

    def __init__(self, start_path: Path, end_path: Path, length: float, speed: int, name: str):

        if not start_path.exists() or not end_path.exists():
            raise FileExistsError

        start_frame = read_df(start_path)
        end_frame = read_df(end_path)

        if not len(start_frame) == len(end_frame):
            raise ValueError
        if not (start_frame.stamp == end_frame.stamp).all():
            raise ValueError
        if not (start_frame.Time == end_frame.Time).all():
            raise ValueError
        if not (start_frame.stamp == end_frame.stamp).all():
            raise ValueError

        start_frame.Time = pd.to_datetime(start_frame.Time, utc=True)
        end_frame.Time = pd.to_datetime(end_frame.Time, utc=True)
        et_index = start_frame[start_frame.Time == start_frame.Time.iloc[-1] - timedelta(days=ElapsedTimeFrame.range_offset)].index[0]

        init_velos = start_frame.Velocity_Major.to_numpy()
        final_velos = end_frame.Velocity_Major.to_numpy()

        self.frame = pd.DataFrame(data={'stamp': start_frame.stamp, 'Time': start_frame.Time})
        self.frame.drop(self.frame.index[et_index:], inplace=True)

        dist = ElapsedTimeFrame.distance(final_velos[1:], init_velos[:-1], speed, PresetGlobals.timestep / 3600)
        dist = np.insert(dist, 0, 0.0)  # distance uses an offset calculation VIx, VFx+1, need a zero at the beginning
        self.frame[name] = [elapsed_time(i, dist, length) for i in range(et_index)]


class ElapsedTimeJob(Job):  # super -> job name, result key, function/object, arguments

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, seg: Segment, speed: int):
        # job_name = seg.name + '  ' + str(round(seg.length, 3)) + '  ' + str(speed)
        job_name = f'{speed} {seg.name}'
        # result_key = seg.name + ' ' + str(speed)
        result_key = job_name
        start_path = seg.start.folder.joinpath(Waypoint.velocity_csv_name)
        end_path = seg.end.folder.joinpath(Waypoint.velocity_csv_name)
        arguments = tuple([start_path, end_path, seg.length, speed, seg.name])
        super().__init__(job_name, result_key, ElapsedTimeFrame, arguments)


def edge_processing(route: Route, job_manager):

    for s in PresetGlobals.speeds:
        print(f'\nCalculating elapsed timesteps for edges at {s} kts')
        speeds_file = route.filepath('elapsed timesteps', s)

        if not print_file_exists(speeds_file):
            keys = [job_manager.submit_job(ElapsedTimeJob(seg, s)) for seg in route.segments]
            job_manager.wait()

            print(f'\nAggregating elapsed timesteps at {s} kts into a dataframe', flush=True)
            results = [job_manager.get_result(key) for key in keys]
            elapsed_time_df = reduce(lambda left, right: pd.merge(left, right, on=['stamp', 'Time']), [r.frame for r in results])
            print_file_exists(write_df(elapsed_time_df, speeds_file))
            del results
