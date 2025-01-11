# import pandas as pd
# from scipy.interpolate import CubicSpline
# from pathlib import Path
#
# from tt_noaa_data.noaa_data import noaa_current_dataframe
# from tt_file_tools.file_tools import read_df, write_df, print_file_exists
# from tt_date_time_tools.date_time_tools import date_to_index
# from tt_job_manager.job_manager import Job
# from tt_gpx.gpx import Waypoint
# from tt_globals.globals import Globals
#
#
# def dash_to_zero(value): return 0.0 if str(value).strip() == '-' else value
#
#
# class DownloadedVelocityCSV:
#
#     def __init__(self, fdd, ldd, ndd, ndi, folder: Path, code: str):
#
#         self.filepath = None
#         filepath = folder.joinpath(Globals.WAYPOINT_DATAFILE_NAME)
#
#         if filepath.exists():
#             self.filepath = filepath
#         else:
#             downloaded_frame = noaa_current_dataframe(fdd, ldd, code)
#             downloaded_frame['date_index'] = downloaded_frame['Time'].apply(date_to_index)
#             write_df(downloaded_frame, folder.joinpath('orig_velocity_download.csv'))
#
#             frame = pd.DataFrame()
#             frame['date_time'] = ndd
#             frame['date_index'] = ndi
#
#             cs = CubicSpline(downloaded_frame['date_index'], downloaded_frame[' Velocity_Major'])
#             frame['velocity'] = frame['date_index'].apply(cs)
#             frame['velocity'] = frame['velocity'].round(decimals=3)
#
#             self.filepath = write_df(frame, filepath)
#
#
# class DownloadVelocityJob(Job):  # super -> job name, result key, function/object, arguments
#
#     def execute(self): return super().execute()
#     def execute_callback(self, result): return super().execute_callback(result)
#     def error_callback(self, result): return super().error_callback(result)
#
#     def __init__(self, fdd, ldd, ndd, ndi, wp):
#         result_key = id(wp)
#         # arguments = tuple([global_class.FIRST_DOWNLOAD_DAY, global_class.LAST_DOWNLOAD_DAY, global_class.NORMALIZED_DOWNLOAD_DATES, global_class.NORMALIZED_DOWNLOAD_INDICES, wp.folder, wp.code])
#         arguments = tuple([fdd, ldd, ndd, ndi, wp.folder, wp.code])
#         super().__init__(wp.unique_name, result_key, DownloadedVelocityCSV, arguments)
#
#
# class SplineFitNormalizedVelocityCSV:
#
#     def __init__(self, index_range, velocity_file):
#
#         self.filepath = None
#         filepath = velocity_file.parent.joinpath(Globals.EDGE_DATAFILE_NAME)
#
#         if print_file_exists(filepath):
#             self.filepath = filepath
#         else:
#             velocity_frame = read_df(velocity_file)
#             cs = CubicSpline(velocity_frame['date_index'], velocity_frame['velocity'])
#             frame = pd.DataFrame()
#             frame['date_index'] = index_range
#             frame['date_time'] = pd.to_datetime(frame['date_index'], unit='s').round('min')
#             frame['velocity'] = frame['date_index'].apply(cs)
#             self.filepath = write_df(frame, filepath)
#
#
# class SplineFitNormalizedVelocityJob(Job):  # super -> job name, result key, function/object, arguments
#
#     def execute(self): return super().execute()
#     def execute_callback(self, result): return super().execute_callback(result)
#     def error_callback(self, result): return super().error_callback(result)
#
#     def __init__(self, index_range, waypoint: Waypoint):
#         result_key = id(waypoint)
#         filepath = waypoint.folder.joinpath(Globals.WAYPOINT_DATAFILE_NAME)
#         arguments = tuple([index_range, filepath])
#         super().__init__(waypoint.unique_name, result_key, SplineFitNormalizedVelocityCSV, arguments)
