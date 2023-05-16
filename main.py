# C:\Users\jason\PycharmProjects\Transit-Time\venv\Scripts\python.exe C:\Users\jason\PycharmProjects\Transit-Time\main.py "East River" "C:\users\jason\Developer Workspace\GPX\East River West to East.gpx" 2023 -dd

from argparse import ArgumentParser as argParser
from pathlib import Path
from multiprocessing import Manager
from numpy import ndarray as array
from pandas import DataFrame as dataframe
from sympy import Point

import multiprocess as mpm
from GPX import Route, Waypoint, Edge, CurrentStationWP, InterpolationWP, DataWP
from velocity import CurrentStationJob, InterpolationJob, InterpolationDataJob, VelocityJob
from elapsed_time import ElapsedTimeJob
from elapsed_time_reduce import elapsed_time_reduce
from transit_time import TransitTimeMinimaJob
from project_globals import TIMESTEP, boat_speeds, Environment, ChartYear

from Semaphore import SimpleSemaphore as Semaphore
from ChromeDriver import ChromeDriver as cd
from FileTools import FileTools as ft

from VelocityInterpolation import Interpolator as vi

checkmark = u'\N{check mark}'

def assign_verify_output_data(entity, ent_type):
    entity.output_data = mpm.result_lookup[id(entity)]
    if isinstance(entity.output_data, ent_type): print(f'{checkmark}     {entity.unique_name}', flush=True)
    else: print(f'X     {entity.unique_name}', flush=True)

if __name__ == '__main__':

    ap = argParser()
    ap.add_argument('project_name', type=str, help='name of transit window project')
    ap.add_argument('filepath', type=Path, help='path to gpx file')
    ap.add_argument('year', type=int, help='calendar year for analysis')
    ap.add_argument('-dd', '--delete_data', action='store_true')
    args = vars(ap.parse_args())
    env = Environment(args)
    cy = ChartYear(args)

    Waypoint.velocity_folder = env.velocity_folder()
    Edge.elapsed_time_folder = env.elapsed_time_folder()

    # Assemble route and route objects
    route = Route(args['filepath'], cy.waypoint_start_index(), cy.waypoint_end_index(), cy.edge_range())

    print(f'\nCalculating route "{route.name}"')
    print(f'total waypoints: {len(route.whole_path.edges)+1}')
    print(f'elapsed time waypoints: {len(route.elapsed_time_path.edges)+1}')
    print(f'timestep: {TIMESTEP}')
    print(f'boat speeds: {boat_speeds}')
    print(f'length {round(route.elapsed_time_path.length,1)} nm')
    print(f'direction {route.elapsed_time_path.direction}')
    print(f'heading {route.elapsed_time_path.heading}\n')

    mgr = Manager()
    mpm.result_lookup = mgr.dict()
    jm = mpm.WaitForProcess(target=mpm.JobManager, args=(mpm.job_queue, mpm.result_lookup))
    jm.start()

    # cd.update_driver()  # update chrome driver before launching process that use it

    # Download noaa data and create velocity arrays for each CURRENT waypoint
    print(f'\nDownloading and processing currents at CURRENT and INTERPOLATION DATA waypoints (1st day-1 to last day+3)', flush=True)
    for wp in route.waypoints:
        if isinstance(wp, DataWP):  # DataWP must come before CurrentStationWP because DataWP IS A CurrentStationWP
            mpm.job_queue.put(InterpolationDataJob(args['year'], wp))
            # idj = InterpolationDataJob(args['year'], wp)
            # idj.execute()
            # pass
        elif isinstance(wp, CurrentStationWP):
            mpm.job_queue.put(CurrentStationJob(args['year'], wp, TIMESTEP))
            # csj = CurrentStationJob(args['year'], wp, TIMESTEP)
            # csj.execute()
            # pass
    mpm.job_queue.join()

    print(f'\nAdding results to waypoints', flush=True)
    for wp in route.waypoints:
        if isinstance(wp, CurrentStationWP) or isinstance(wp, DataWP):
            assign_verify_output_data(wp, array)

    # Calculate the approximation of the velocity at interpolation points
    print(f'\nApproximating the velocity at INTERPOLATION waypoints (1st day-1 to last day+3)', flush=True)
    for group in route.interpolation_groups:
        interpolation_pt = group[0]

        if not ft.file_exists(interpolation_pt.interpolation_data_file):
            group_range = range(len(group[1].output_data))
            for i in group_range: mpm.job_queue.put(InterpolationJob(group, i))  # (group, i, True) to display results
            mpm.job_queue.join()

            wp_data = [mpm.result_lookup[str(id(interpolation_pt)) + '_' + str(i)][2].evalf() for i in group_range]
            InterpolationJob.write_dataframe(interpolation_pt, wp_data)

        mpm.job_queue.put(CurrentStationJob(args['year'], interpolation_pt, TIMESTEP))
        mpm.job_queue.join()

        if isinstance(interpolation_pt, InterpolationWP):
            assign_verify_output_data(interpolation_pt, array)

    # Calculate the number of timesteps to get from the start of the edge to the end of the edge
    print(f'\nCalculating elapsed times for edges (1st day-1 to last day+2)')
    for edge in route.elapsed_time_path.edges:
        mpm.job_queue.put(ElapsedTimeJob(edge))
        # etj = ElapsedTimeJob(edge)
        # etj.execute()
    mpm.job_queue.join()

    print(f'\nAdding results to edges')
    for edge in route.elapsed_time_path.edges:
        assign_verify_output_data(edge, dataframe)

    # combine elapsed times by speed
    print(f'\nSorting elapsed times by speed', flush=True)
    elapsed_time_reduce(env.elapsed_time_folder(), route)

    # calculate the number of timesteps from first node to last node
    print(f'\nCalculating transit times (1st day-1 to last day+1)')
    for speed in boat_speeds:
        mpm.job_queue.put(TransitTimeMinimaJob(env, cy, route, speed))
        # tt = TransitTimeMinimaJob(env, cy, route, speed)
        # tt.execute()
    mpm.job_queue.join()

    Semaphore.off(mpm.job_manager_semaphore)
