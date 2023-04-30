# C:\Users\jason\PycharmProjects\Transit-Time\venv\Scripts\python.exe C:\Users\jason\PycharmProjects\Transit-Time\main.py ER "C:\users\jason\Developer Workspace\GPX\East River West to East.gpx" 2023 -dd
from argparse import ArgumentParser as argParser
from pathlib import Path
from multiprocessing import Manager
from numpy import ndarray as array
from pandas import DataFrame as dataframe

import multiprocess as mpm
from GPX import Route, CurrentStationWP, InterpolationWP
from velocity import CurrentStationJob, InterpolationJob
from elapsed_time import ElapsedTimeJob
from elapsed_time_reduce import elapsed_time_reduce
from transit_time import TransitTimeMinimaJob
from project_globals import TIMESTEP, boat_speeds, Environment, ChartYear

from Semaphore import SimpleSemaphore as Semaphore
from ChromeDriver import ChromeDriver as cd
from VelocityInterpolation import Interpolator as vi

checkmark = u'\N{check mark}'

if __name__ == '__main__':

    ap = argParser()
    ap.add_argument('project_name', type=str, help='name of transit window project')
    ap.add_argument('filepath', type=Path, help='path to gpx file')
    ap.add_argument('year', type=int, help='calendar year for analysis')
    ap.add_argument('-dd', '--delete_data', action='store_true')
    args = vars(ap.parse_args())

    # Assemble route and route objects
    route = Route(args['filepath'])

    print(f'\nCalculating route "{route.name}"')
    print(f'total waypoints: {len(route.whole_path.edges)+1}')
    print(f'elapsed time waypoints: {len(route.velo_path.edges)+1}')
    print(f'timestep: {TIMESTEP}')
    print(f'boat speeds: {boat_speeds}')
    print(f'length {round(route.velo_path.length,1)} nm')
    print(f'direction {route.velo_path.direction}\n')

    envr = Environment(args)
    cyr = ChartYear(args)

    mgr = Manager()
    mpm.result_lookup = mgr.dict()
    jm = mpm.WaitForProcess(target=mpm.JobManager, args=(mpm.job_queue, mpm.result_lookup))
    jm.start()

    cd.update_driver()  # update chrome driver before launching process that use it

    # Download noaa data and create velocity arrays for each waypoint (node)
    print(f'\nCalculating currents at waypoints (1st day-1 to last day+3)')
    for wp in route.current_stations: mpm.job_queue.put(CurrentStationJob(envr, cyr, wp))
    mpm.job_queue.join()

    print(f'\nAdding results to waypoints')
    for wp in route.current_stations:
        wp.velo_arr = mpm.result_lookup[id(wp)]
        if isinstance(wp.velo_arr, array): print(f'{checkmark}     {wp.short_name}', flush=True)
        else: print(f'X     {wp.short_name}', flush=True)

    # Calculate the number of timesteps to get from the start of the edge to the end of the edge
    print(f'\nCalculating elapsed times for segments (1st day-1 to last day+2)')
    for edge in route.velo_path.edges: mpm.job_queue.put(ElapsedTimeJob(envr, cyr, edge))
    mpm.job_queue.join()

    print(f'\nAdding results to edges')
    for edge in route.velo_path.edges:
        edge.dataframe = mpm.result_lookup[id(edge)]
        if isinstance(edge.dataframe, dataframe): print(f'{checkmark}     {edge.name}', flush=True)
        else: print(f'X     {edge.name}', flush=True)

    # combine elapsed times by speed
    print(f'\nSorting elapsed times by speed', flush=True)
    elapsed_time_reduce(envr, route)

    # calculate the number of timesteps from first node to last node
    print(f'\nCalculating transit times (1st day-1 to last day+1)')
    for speed in boat_speeds: mpm.job_queue.put(TransitTimeMinimaJob(envr, cyr, route, speed))
    mpm.job_queue.join()

    Semaphore.off(mpm.job_manager_semaphore)
