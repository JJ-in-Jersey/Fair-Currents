from argparse import ArgumentParser as argParser
from pathlib import Path
from functools import reduce
import numpy as np
from pandas import merge, to_datetime, concat

from tt_dataframe.dataframe import DataFrame
from tt_globals.globals import PresetGlobals
from tt_gpx.gpx import Route, EdgeNode, GpxFile
from tt_job_manager.job_manager import JobManager
from tt_noaa_data.noaa_data import StationDict
from tt_file_tools.file_tools import print_file_exists

from tt_jobs.jobs import ElapsedTimeJob, SavGolFrame, SavGolJob, TimeStepsJob, MinimaJob, ArcsJob


if __name__ == '__main__':

    ap = argParser()
    ap.add_argument('filepath', type=Path, help='path to gpx file')
    args = vars(ap.parse_args())

    # ---------- ROUTE OBJECT ----------
    if PresetGlobals.stations_file.exists():
        station_dict = StationDict(json_source=PresetGlobals.stations_file)
    else:
        station_dict = StationDict()
    gpx_file = GpxFile(args['filepath'])
    route = Route(station_dict, gpx_file.tree)

    print(f'\nParameters')
    print(f'savgol window size {SavGolFrame.savgol_size}')
    print(f'savgol polynomial order {SavGolFrame.savgol_order}')

    print(f'\nCalculating route {route.name}')
    print(f'code {route.code}')
    print(f'total waypoints: {len(route.waypoints)}')
    print(f'total edge nodes: {len(list(filter(lambda w: isinstance(w, EdgeNode), route.waypoints)))}')
    print(f'total edges: {len(route.edges)}')
    print(f'total segments: {len(route.segments)}')
    for segment in route.segments:
        node_list = []
        for node in segment.node_list[:-1]:
            node_list.append(node.name)
            node_list.append(node.next_edge.length)
        node_list.append(segment.node_list[-1].name)
        print(f'{segment.name}: {node_list} {int(segment.length*100)/100} nm')
    print(f'boat speeds: {PresetGlobals.speeds}')
    print(f'length {int(route.length*100)/100} nm')
    print(f'direction: {route.directions[0]}, abbrev: {route.dir_abbrevs[0]},  heading: {route.heading}\n')


    heading_file = route.folder.joinpath(str(int(np.round(route.heading))) + '.heading')
    heading_file.parent.mkdir(parents=True, exist_ok=True)
    heading_file.touch()
    length_file = route.folder.joinpath(str(int(np.round(route.length))) + '.length')
    length_file.parent.mkdir(parents=True, exist_ok=True)
    length_file.touch()
    pos_dir_file = route.folder.joinpath(route.dir_abbrevs[0] + '.dir')
    pos_dir_file.parent.mkdir(parents=True, exist_ok=True)
    pos_dir_file.touch()
    neg_dir_file = route.folder.joinpath(route.dir_abbrevs[1] + '.-dir')
    neg_dir_file.parent.mkdir(parents=True, exist_ok=True)
    neg_dir_file.touch()

    # ---------- CHECK CHROME ----------
    # chrome_driver.check_driver()
    # if chrome_driver.installed_driver_version is None or chrome_driver.latest_stable_version > chrome_driver.installed_driver_version:
    #     chrome_driver.install_stable_driver()

    # ---------- START MULTIPROCESSING ----------
    job_manager = JobManager()
    # job_manager = None

    print(f'\nCalculating elapsed times')
    results = {}
    for speed in PresetGlobals.speeds:
        print(f'Edges at {speed} kts')

        ets_path = route.filepath('elapsed timesteps', speed)
        if ets_path.exists():
            elapsed_time_df = DataFrame(csv_source=ets_path)
            elapsed_time_df.Time = to_datetime(elapsed_time_df.Time, utc=True)
        else:
            keys = [job_manager.submit_job(ElapsedTimeJob(seg, speed)) for seg in route.segments]
            # for seg in route.segments:
            #     job = ElapsedTimeJob(seg, speed)
            #     result = job.execute()
            job_manager.wait()

            print(f'      Aggregating elapsed timesteps at {speed} kts into a dataframe', flush=True)
            edge_frames = [job_manager.get_result(key) for key in keys]
            elapsed_time_df = reduce(lambda left, right: merge(left, right, on=['stamp', 'Time']), edge_frames)
            elapsed_time_df.write(ets_path)
        results[speed] = elapsed_time_df

    print(f'\nCalculating transit times')
    keys = [job_manager.submit_job(TimeStepsJob(frame, speed, route)) for speed, frame in results.items()]
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
    frames = [DataFrame(csv_source=route.filepath('arcs', speed)) for speed in PresetGlobals.speeds]
    transit_times_df = concat(frames).reset_index(drop=True)
    transit_times_df = transit_times_df.fillna("-")
    transit_times_df.drop(['start_datetime', 'min_datetime', 'end_datetime'], axis=1, inplace=True)

    transit_times_path = route.folder.joinpath(route.template_dict['transit times'].substitute({'loc': route.code}))
    print_file_exists(transit_times_df.write(transit_times_path))


    # # if args['east_river']:
    # #     print(f'\nEast River validation')
    # #
    # #     validation_frame = pd.DataFrame()
    # #
    # #     frame = validations.hell_gate_validation()
    # #     validation_frame = pd.concat([validation_frame, frame])
    # #
    # #     frame = validations.battery_validation()
    # #     validation_frame = pd.concat([validation_frame, frame])
    # #
    # #     validation_frame.sort_values(['start_date'], ignore_index=True, inplace=True)
    # #     write_df(validation_frame, Globals.TRANSIT_TIMES_FOLDER.joinpath('east_river_validation.csv'))
    # #
    # # if args['chesapeake_delaware_canal']:
    # #     print(f'\nChesapeake Delaware Canal validation')
    # #
    # #     validation_frame = pd.DataFrame()
    # #
    # #     frame = validations.chesapeake_city_validation()
    # #     validation_frame = pd.concat([validation_frame, frame])
    # #
    # #     frame = validations.reedy_point_tower_validation()
    # #     validation_frame = pd.concat([validation_frame, frame])
    # #
    # #     validation_frame.sort_values(['start_date'], ignore_index=True, inplace=True)
    # #     write_df(validation_frame, Globals.TRANSIT_TIMES_FOLDER.joinpath('chesapeake_delaware_validation.csv'))

    print(f'\nProcess Complete')

    job_manager.stop_queue()
