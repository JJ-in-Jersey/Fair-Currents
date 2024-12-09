from argparse import ArgumentParser as argParser
from pathlib import Path

from tt_gpx.gpx import Route, EdgeNode, GpxFile
from tt_chrome_driver import chrome_driver
from tt_job_manager.job_manager import JobManager
from tt_globals.globals import PresetGlobals
from tt_noaa_data.noaa_data import StationDict

from elapsed_time import edge_processing
from transit_time import transit_time_processing

if __name__ == '__main__':

    # ---------- PARSE ARGUMENTS ----------

    ap = argParser()
    ap.add_argument('filepath', type=Path, help='path to gpx file')
    # ap.add_argument('-dd', '--delete_data', action='store_true')
    # ap.add_argument('-er', '--east_river', action='store_true')
    # ap.add_argument('-cdc', '--chesapeake_delaware_canal', action='store_true')
    args = vars(ap.parse_args())


    # ---------- ROUTE OBJECT ----------

    station_dict = StationDict()
    gpx_file = GpxFile(args['filepath'])
    route = Route(station_dict.dict, gpx_file.tree)

    print(f'\nCalculating route {route.name}')
    print(f'code {route.code}')
    print(f'total waypoints: {len(route.waypoints)}')
    print(f'total edge nodes: {len(list(filter(lambda w: isinstance(w, EdgeNode), route.waypoints)))}')
    print(f'total edges: {len(route.edges)}')
    print(f'boat speeds: {PresetGlobals.speeds}')
    print(f'length {route.length} nm')
    print(f'direction {route.direction}')
    print(f'heading {route.heading}\n')

    # ---------- CHECK CHROME ----------
    # chrome_driver.check_driver()
    # if chrome_driver.installed_driver_version is None or chrome_driver.latest_stable_version > chrome_driver.installed_driver_version:
    #     chrome_driver.install_stable_driver()

    # ---------- START MULTIPROCESSING ----------
    job_manager = JobManager()

    # ---------- EDGE PROCESSING ----------
    edge_processing(route, job_manager)

    # ---------- TRANSIT TIMES ----------
    transit_time_processing(job_manager, route)

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
