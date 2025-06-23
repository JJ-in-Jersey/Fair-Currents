import numpy as np
from pandas import merge, to_datetime
from pathlib import Path
from datetime import timedelta
from functools import reduce

from tt_dataframe.dataframe import DataFrame
from tt_gpx.gpx import Route, Waypoint, Segment
from tt_job_manager.job_manager import Job
from tt_globals.globals import PresetGlobals as pg
from tt_file_tools.file_tools import print_file_exists


class ElapsedTimeFrame(DataFrame):

    @staticmethod
    def distance(water_vf, water_vi, boat_speed, ts_in_hr):
            dist = ((water_vf + water_vi) / 2 + boat_speed) * ts_in_hr  # distance is nm
            # if np.all(np.sign(dist) == np.sign(boat_speed)):
            #     return abs(dist)
            # else:
            #     raise ValueError
            return dist

    #  Elapsed times are reported in number of timesteps
    @staticmethod
    def elapsed_time(distances, length):
        cumsum = 0
        index = 0
        while cumsum < length:
            cumsum += distances[index]
            index += 1
        return index

    def __init__(self, start_path: Path, end_path: Path, length: float, speed: int, name: str):

        if not start_path.exists() or not end_path.exists():
            raise FileExistsError

        start_frame = DataFrame(csv_source=start_path)
        end_frame = DataFrame(csv_source=end_path)

        if not len(start_frame) == len(end_frame):
            raise ValueError
        if not (start_frame.stamp == end_frame.stamp).all():
            raise ValueError
        if not (start_frame.Time == end_frame.Time).all():
            raise ValueError

        dist = self.distance(end_frame.Velocity_Major.to_numpy()[1:], start_frame.Velocity_Major.to_numpy()[:-1], speed, pg.timestep / 3600)
        dist = dist * np.sign(speed)  # make sure distances are positive in the direction of the current
        dist_limit = len(dist) -1 - np.where(np.cumsum(dist[::-1]) > length)[0][0]  # don't run off the end of the array
        timesteps = [self.elapsed_time(dist[i:], length) for i in range(dist_limit)]
        timesteps.insert(0,0)  # initial time 0 has no displacement

        frame = DataFrame(data={'stamp': start_frame.stamp, 'Time': to_datetime(start_frame.Time, utc=True)})
        frame = frame.loc[:dist_limit]
        frame[name] = timesteps
        super().__init__(data=frame)


class ElapsedTimeJob(Job):  # super -> job name, result key, function/object, arguments

    def execute(self): return super().execute()
    def execute_callback(self, result): return super().execute_callback(result)
    def error_callback(self, result): return super().error_callback(result)

    def __init__(self, seg: Segment, speed: int):

        node = seg.start
        node_path = node.folder.joinpath(Waypoint.velocity_csv_name)
        while not node_path.exists():
            node = node.next_edge.end
            node_path = node.folder.joinpath(Waypoint.velocity_csv_name)
        start_path = node_path

        node = seg.end
        node_path = node.folder.joinpath(Waypoint.velocity_csv_name)
        while not node_path.exists():
            node = node.prev_edge.start
            node_path = node.folder.joinpath(Waypoint.velocity_csv_name)
        end_path = node_path

        job_name = f'{speed} {seg.name}'
        result_key = job_name
        arguments = tuple([start_path, end_path, seg.length, speed, seg.name])
        super().__init__(job_name, result_key, ElapsedTimeFrame, arguments)


def edge_processing(route: Route, job_manager):

    for s in pg.speeds:
        print(f'\nCalculating elapsed timesteps for edges at {s} kts')

        if not print_file_exists(route.filepath('elapsed timesteps', s)):
            keys = [job_manager.submit_job(ElapsedTimeJob(seg, s)) for seg in route.segments]
            # for seg in route.segments:
            #     job = ElapsedTimeJob(seg, s)
            #     result = job.execute()
            job_manager.wait()

            print(f'\nAggregating elapsed timesteps at {s} kts into a dataframe', flush=True)
            result_frames = [job_manager.get_result(key) for key in keys]
            elapsed_time_df = reduce(lambda left, right: merge(left, right, on=['stamp', 'Time']), result_frames)
            print_file_exists(elapsed_time_df.write(route.filepath('elapsed timesteps', s)))