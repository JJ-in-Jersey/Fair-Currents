from argparse import ArgumentParser as argParser
from pathlib import Path

import pandas as pd
from time import perf_counter

# noinspection PyPep8Naming
from pandas import DataFrame, concat as Concat

from tt_gpx.gpx import Route, Waypoint, Edge, TideWP, InterpolationWP, DataWP, CurrentStationWP
# from tt_semaphore import simple_semaphore as semaphore
from tt_file_tools import file_tools as ft
from tt_chrome_driver import chrome_driver
from tt_date_time_tools import date_time_tools as dt
from tt_job_manager.job_manager import JobManager

# import multiprocess as mpm
from velocity import CurrentStationJob, InterpolationJob, InterpolationDataJob
from battery_validation import TideStationJob
from elapsed_time import ElapsedTimeJob
from dataframe_merge import elapsed_time_reduce
from transit_time import TransitTimeMinimaJob
from project_globals import TIMESTEP, TIME_RESOLUTION, WINDOW_MARGIN, boat_speeds, Environment, ChartYear

from hell_gate_validation import HellGateSlackTimes

checkmark = u'\N{check mark}'

if __name__ == '__main__':

    ap = argParser()
    ap.add_argument('project_name', type=str, help='name of transit window project')
    ap.add_argument('filepath', type=Path, help='path to gpx file')
    ap.add_argument('year', type=int, help='calendar year for analysis')
    ap.add_argument('-dd', '--delete_data', action='store_true')
    args = vars(ap.parse_args())
    env = Environment(args)
    cy = ChartYear(args)

    Waypoint.velocity_folder = env.velocity_folder
    Waypoint.current_folder = env.current_folder
    Edge.elapsed_time_folder = env.elapsed_time_folder

    # Assemble route
    route = Route(args['filepath'], cy.waypoint_start_index(), cy.waypoint_end_index(), cy.edge_range())

    print(f'\nCalculating route "{route.filepath.stem}"')
    print(f'total waypoints: {len(route.whole_path.edges)+1}')
    print(f'elapsed time waypoints: {len(route.elapsed_time_path.edges)+1}')
    print(f'timestep: {TIMESTEP}')
    print(f'chart resolution: {TIME_RESOLUTION}')
    print(f'transit time window: {WINDOW_MARGIN}')
    print(f'boat speeds: {boat_speeds}')
    print(f'length {round(route.elapsed_time_path.length, 1)} nm')
    print(f'direction {route.elapsed_time_path.direction}')
    print(f'heading {route.elapsed_time_path.heading}\n')

    env.transit_time_folder.joinpath(str(route.elapsed_time_path.heading) + '.heading').touch()

    job_manager = JobManager()

    chrome_driver.check_driver()
    if chrome_driver.latest_stable_version > chrome_driver.installed_driver_version:
        chrome_driver.install_stable_driver()

    # Download noaa data and create velocity arrays for each CURRENT waypoint
    print(f'\nDownloading and processing currents at CURRENT and INTERPOLATION DATA waypoints (1st day-1 to last day+4)', flush=True)
    init_time = perf_counter()
    for wp in route.waypoints:
        if isinstance(wp, DataWP):  # DataWP must come before CurrentStationWP because DataWP IS A CurrentStationWP
            job_manager.put(InterpolationDataJob(args['year'], wp))
            # idj = InterpolationDataJob(args['year'], wp)
            # idj.execute()
            # pass
        elif isinstance(wp, CurrentStationWP):
            job_manager.put(CurrentStationJob(args['year'], wp, TIMESTEP, False))
            # csj = CurrentStationJob(args['year'], wp, TIMESTEP)
            # csj.execute()
            # pass
    job_manager.wait()
    print(f' Multi-process time {dt.mins_secs(perf_counter()-init_time)}')

    print(f'\nAdding results to waypoints', flush=True)
    for wp in route.waypoints:
        if isinstance(wp, CurrentStationWP) or isinstance(wp, DataWP):
            wp.current_data = job_manager.get(id(wp))
            if isinstance(wp.current_data, pd.DataFrame): print(f'{checkmark}     {wp.unique_name}', flush=True)
            else: print(f'X     {wp.unique_name}', flush=True)

    # del job_manager
    # job_manager = JobManager()

    # Calculate the approximation of the velocity at interpolation points
    if len(route.interpolation_groups):
        print(f'\nApproximating the velocity at INTERPOLATION waypoints (1st day-1 to last day+4)', flush=True)
        init_time = perf_counter()
        for group in route.interpolation_groups:
            interpolation_pt = group[0]
            data_waypoints = group[1:]
            if not ft.csv_npy_file_exists(interpolation_pt.final_data_filepath):
                group_range = range(len(data_waypoints[0].current_data))
                print(f'adding {len(group_range)} jobs to queue')
                for i in group_range:
                    job_manager.put(InterpolationJob(interpolation_pt, data_waypoints, len(group_range), i))  # (group, i, True) to display results
                    # ij = InterpolationJob(group, i)
                    # ij.execute()
                job_manager.wait()

                df = pd.DataFrame([job_manager.get(str(id(interpolation_pt)) + '_' + str(i)) for i in group_range], columns=['date_index', 'velocity'])
                interpolation_pt.set_current_data(df)

            job_manager.put(CurrentStationJob(args['year'], interpolation_pt, TIMESTEP))
            job_manager.wait()

            if isinstance(interpolation_pt, InterpolationWP):
                interpolation_pt.current_data = job_manager.get(id(interpolation_pt))
                if isinstance(interpolation_pt.current_data, pd.DataFrame): print(f'{checkmark}     {interpolation_pt.unique_name}', flush=True)
                else: print(f'X     {interpolation_pt.unique_name}', flush=True)
        print(f'Multi-process time {dt.mins_secs(perf_counter()-init_time)}')

    # Calculate the number of timesteps to get from the start of the edge to the end of the edge
    print(f'\nCalculating elapsed times for edges (1st day-1 to last day+3)')
    init_time = perf_counter()
    for edge in route.elapsed_time_path.edges:
        job_manager.put(ElapsedTimeJob(edge))
        # etj = ElapsedTimeJob(edge)
        # etj.execute()
    job_manager.wait()
    print(f'Multi-process time {dt.mins_secs(perf_counter() - init_time)}')

    print(f'\nAdding results to edges')
    for edge in route.elapsed_time_path.edges:
        edge.output_data = job_manager.get(id(edge))
        if isinstance(edge.output_data, DataFrame): print(f'{checkmark}     {edge.unique_name}', flush=True)
        else: print(f'X     {edge.unique_name}', flush=True)

    # combine elapsed times by speed
    print(f'\nSorting elapsed times by speed', flush=True)
    elapsed_time_reduce(env.elapsed_time_folder, route)

    # calculate the number of timesteps from first node to last node
    print(f'\nCalculating transit times (1st day-1 to last day+2)')
    init_time = perf_counter()
    for speed in boat_speeds:
        job_manager.put(TransitTimeMinimaJob(env, cy, route, speed))
    # tt = TransitTimeMinimaJob(env, cy, route, speed)
    # tt.execute()
    job_manager.wait()
    print(f'Multi-process time {dt.mins_secs(perf_counter() - init_time)}')

    print(f'\nAdding transit time speed results to route')
    for speed in boat_speeds:
        route.transit_time_lookup[speed] = job_manager.get(speed)
        if isinstance(route.transit_time_lookup[speed], DataFrame): print(f'{checkmark}     tt {speed}', flush=True)
        else: print(f'X     tt {speed}', flush=True)

    if not ft.csv_npy_file_exists(env.transit_time_folder.joinpath('text_rotation')):
        # text_rotation_df = DataFrame(columns=['date', 'angle'])
        text_rotation_df = pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'), 'angle': pd.Series(dtype='float')})
        text_arcs_df = Concat([route.transit_time_lookup[boat_speeds[-1]], route.transit_time_lookup[boat_speeds[-1]]])
        text_arcs_df.sort_values(['date', 'start'], ignore_index=True, inplace=True)
        for date in text_arcs_df['date'].drop_duplicates(ignore_index=True):
            date_df = text_arcs_df[text_arcs_df['date'] == date].sort_index().reset_index(drop=True)
            angle = (date_df.loc[1, 'start'] + date_df.loc[0, 'end'])/2
            text_rotation_df = Concat([text_rotation_df, DataFrame.from_dict({'date': [date], 'angle': [angle]})])
            text_arcs_df = text_arcs_df[text_arcs_df['date'] != date]
        ft.write_df(text_rotation_df, env.transit_time_folder.joinpath('legend'))

    arcs_df = Concat([route.transit_time_lookup[key] for key in route.transit_time_lookup])
    arcs_df.sort_values(['date'], ignore_index=True, inplace=True)
    min_rotation_df = arcs_df[arcs_df['min'].notna()]
    min_rotation_df = min_rotation_df.rename(columns={'min': 'angle'})
    min_rotation_df['name'] = min_rotation_df['name'].apply(lambda s: s.replace('Arc', 'Min Arrow'))
    min_rotation_df = min_rotation_df.drop(['date_time', 'start', 'end'], axis=1)
    ft.write_df(min_rotation_df, env.transit_time_folder.joinpath('minima'))
    arcs_df.drop(['date_time', 'min'], axis=1, inplace=True)
    ft.write_df(arcs_df, env.transit_time_folder.joinpath('arcs'))

    erv = HellGateSlackTimes(cy, route.waypoints)
    ft.write_df(erv.hell_gate_slack, env.transit_time_folder.joinpath('hell_gate_slack'))

    print(f'\nCreating Battery tide arcs')
    battery_wp = TideWP(args['filepath'].parent.joinpath('NOAA Tide Stations/8518750.gpx'))
    battery_job = TideStationJob(cy, battery_wp, TIMESTEP, False)
    battery_job.execute()
    ft.write_df(battery_job.battery_lines, env.transit_time_folder.joinpath('battery_tide'))

    del job_manager
