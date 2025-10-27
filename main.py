from argparse import ArgumentParser as argParser
from pathlib import Path
from functools import reduce
import numpy as np
from pandas import merge, to_datetime, concat

from tt_dataframe.dataframe import DataFrame
import tt_globals.globals as fc_globals
from tt_gpx.gpx import Route, EdgeNode, GpxFile
from tt_job_manager.job_manager import JobManager
import tt_jobs.jobs as jobs
from tt_jobs.jobs import ElapsedTimeFrame
from tt_noaa_data.noaa_data import StationDict
from tt_file_tools.file_tools import print_file_exists

if __name__ == '__main__':

    ap = argParser()
    ap.add_argument('filepath', type=Path, help='path to gpx file')
    args = vars(ap.parse_args())

    # ---------- ROUTE OBJECT ----------
    if fc_globals.STATIONS_FILE.exists():
        station_dict = StationDict(json_source=fc_globals.STATIONS_FILE)
    else:
        station_dict = StationDict()
    gpx_file = GpxFile(args['filepath'])
    route = Route(station_dict, gpx_file.tree)

    print(f'\nParameters')
    print(f'savgol window size {jobs.SavGolFrame.savgol_size}')
    print(f'savgol polynomial order {jobs.SavGolFrame.savgol_order}')

    print(f'\nCalculating route {Route.name}')
    print(f'code {Route.code}')
    print(f'total waypoints: {len(Route.waypoints)}')
    print(f'total edge nodes: {len(list(filter(lambda w: isinstance(w, EdgeNode), Route.waypoints)))}')
    print(f'total edges: {len(Route.edges)}')
    print(f'total segments: {len(Route.segments)}')
    for segment in Route.segments:
        node_list = []
        for node in segment.node_list[:-1]:
            node_list.append(node.name)
            node_list.append(node.next_edge.length)
        node_list.append(segment.node_list[-1].name)
        print(f'{segment.name}: {node_list} {int(segment.length*100)/100} nm')
    print(f'boat speeds: {fc_globals.SPEEDS}')
    print(f'length {int(Route.length*100)/100} nm')
    print(f'direction: {Route.directions[0]}, abbrev: {Route.dir_abbrevs[0]},  heading: {Route.heading}\n')


    heading_file = Route.folder.joinpath(str(int(np.round(Route.heading))) + '.heading')
    heading_file.parent.mkdir(parents=True, exist_ok=True)
    heading_file.touch()
    length_file = Route.folder.joinpath(str(int(np.round(Route.length))) + '.length')
    length_file.parent.mkdir(parents=True, exist_ok=True)
    length_file.touch()
    pos_dir_file = Route.folder.joinpath(Route.dir_abbrevs[0] + '.dir')
    pos_dir_file.parent.mkdir(parents=True, exist_ok=True)
    pos_dir_file.touch()
    neg_dir_file = Route.folder.joinpath(Route.dir_abbrevs[1] + '.-dir')
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
    for speed in fc_globals.SPEEDS:
        print(f'Edges at {speed} kts')
        path = Route.filepath(ElapsedTimeFrame, speed)
        if path.exists():
            print(f'      Reading existing csv file for {speed}     ', end='', flush=True)
            results[speed] = ElapsedTimeFrame(csv_source=path)
            print(results[speed].message, flush=True)
        else:
            print(f'      Calculating elapsed times for {speed}', flush=True)
            keys = [job_manager.submit_job(jobs.ElapsedTimeJob(seg, speed)) for seg in route.segments]
            # for seg in route.segments:
            #     job = jobs.ElapsedTimeJob(seg, speed)
            #     results = job.execute()
            job_manager.wait()
            print(f'      Aggregating elapsed timesteps at {speed} kts into a dataframe', flush=True)
            print(f'           collecting results')
            edge_frames = [job_manager.get_result(key) for key in keys]
            print(f'           aggregating results')
            elapsed_time_df = DataFrame(reduce(lambda left, right: merge(left, right, on=['stamp', 'Time', 'date']), edge_frames))
            print(f'           writing results')
            elapsed_time_df.write(route.filepath(jobs.ElapsedTimeFrame, speed))
            results[speed] = elapsed_time_df

    print(f'\nCalculating transit times')
    keys = [job_manager.submit_job(jobs.TimeStepsJob(frame, speed)) for speed, frame in results.items()]
    # for speed, frame in results.items():
    #     job = jobs.TimeStepsJob(frame, speed)
    #     result = job.execute()
    job_manager.wait()
    results = {speed: job_manager.get_result(speed) for speed in keys}  # {'speed': frame}
    for s, f in results.items():
        path = Route.filepath(jobs.TimeStepsFrame, s)
        if not path.exists():
            print(f'      Writing transit times for {s}', flush=True)
            f.write(path)

    print(f'\nGenerating fair current frames')
    keys = [job_manager.submit_job(jobs.FairCurrentJob(frame, speed)) for speed, frame in results.items()]
    # for speed, frame in results.items():
    #     job = jobs.FairCurrentJob(frame, speed)
    #     result = job.execute()
    job_manager.wait()
    for s, f in results.items():
        path = Route.filepath(jobs.FairCurrentFrame, s)
        if not path.exists():
            print(f'      Writing fair current frames for {s}', flush=True)
            f.write(path)
            
    print(f'\nGenerating savgol frames')
    keys = [job_manager.submit_job(jobs.SavGolJob(frame, speed, route)) for speed, frame in results_transit_times_frame.items()]
    job_manager.wait()
    results_savgol_frame = {speed: job_manager.get_result(speed) for speed in keys}  # {'speed': frame}

    print(f'\nGenerating fair current minima frames')
    keys = [job_manager.submit_job(jobs.FairCurrentMinimaJob(frame, speed, route)) for speed, frame in results_fair_currents_frame.items()]
    # for speed, frame in results_fair_currents_frame.items():
    #     job = jobs.FairCurrentMinimaJob(frame, speed, route)
    #     result = job.execute()
    job_manager.wait()
    results_fair_current_minima_frame = {speed: job_manager.get_result(speed) for speed in keys}  # {'speed': frame}

    print(f'\nGenerating savgol minima frames')
    keys = [job_manager.submit_job(jobs.SavGolMinimaJob(frame, speed, route)) for speed, frame in results_savgol_frame.items()]
    job_manager.wait()
    results_savgol_minima_frame = {speed: job_manager.get_result(speed) for speed in keys}  # {'speed': frame}

    minima_frames = {}
    print(f'\nAggregating fair current and savgol minima frames')
    for speed in fc_globals.SPEEDS:
        print(f'Minima frame for {speed}')
        minima_frame_path = route.filepath('MinimaFrame', speed)
        if minima_frame_path.exists():
            frame = DataFrame(csv_source=minima_frame_path)
        else:
            fair_currents_frame = results_fair_current_minima_frame.get(speed)
            savgol_frame = results_savgol_minima_frame.get(speed)

            if len(fair_currents_frame) > 0:
                savgol_frame.drop(columns=['min_datetime', 'min_duration'], inplace=True)
                frame = DataFrame(concat([fair_currents_frame, savgol_frame], axis=0, ignore_index=True))
            else:
                frame = savgol_frame

            frame.sort_values(by=['start_datetime', 'type'], inplace=True, ignore_index=True)

        for date_col in [c for c in frame.columns.tolist() if '_datetime' in c]:
            frame[date_col] = to_datetime(frame[date_col], utc=True).dt.tz_convert('America/New_York')
        frame.write(minima_frame_path)
        minima_frames[speed] = frame

    # minima_frames = {}
    # for speed in fc_globals.SPEEDS:
    #     path = route.filepath('MinimaFrame', speed)
    #     if path.exists():
    #         frame = DataFrame(csv_source=path)
    #         for date_col in [c for c in frame.columns.tolist() if '_datetime' in c]:
    #             frame[date_col] = to_datetime(frame[date_col], utc=True).dt.tz_convert('America/New_York')
    #     minima_frames[speed] = frame

    print(f'\nGenerating arcs frame')
    keys = [job_manager.submit_job(jobs.ArcsJob(frame, route, speed)) for speed, frame in minima_frames.items()]
    # for speed, frame in minima_frames.items():
    #     job = jobs.ArcsJob(frame, route, speed)
    #     result = job.execute()
    job_manager.wait()

    print(f'\nAggregating transit times', flush=True)
    frames = [DataFrame(csv_source=route.filepath(jobs.ArcsFrame.__name__, speed)) for speed in fc_globals.SPEEDS]
    transit_times_df = DataFrame(concat(frames).reset_index(drop=True))
    transit_times_df = transit_times_df.fillna("-")
    transit_times_df.drop(['start_datetime', 'min_datetime', 'end_datetime'], axis=1, inplace=True)

    transit_times_path = route.folder.joinpath(fc_globals.TEMPLATES['savitsky golay'].substitute({'loc': route.code}))
    print_file_exists(transit_times_df.write(transit_times_path))

    transit_times_path = route.folder.joinpath(fc_globals.TEMPLATES['fair currents'].substitute({'loc': route.code}))
    transit_times_df = transit_times_df[transit_times_df['type'] != 'sg']
    transit_times_df['start_duration_display'] = False
    transit_times_df['end_duration_display'] = False
    print_file_exists(transit_times_df.write(transit_times_path))

    print(f'\nProcess Complete')

    job_manager.stop_queue()
