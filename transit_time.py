import pandas as pd
from scipy.signal import savgol_filter
from time import perf_counter
from num2words import num2words
from project_globals import TIMESTEP, TIMESTEP_MARGIN, FIVE_HOURS_OF_TIMESTEPS
from FileTools import FileTools as ft
from MemoryHelper import MemoryHelper as mh
from DateTimeTools import DateTimeTools as dtt
from Geometry import Arc, RoundedArc, FractionalArcStartDay, FractionalArcEndDay

def none_row(row, df):
    for c in range(len(df.columns)):
        df.iloc[row, c] = None
def total_transit_time(init_row, d_frame, cols):
    row = init_row
    tt = 0
    for col in cols:
        val = d_frame.at[row, col]
        tt += val
        row += val
    return tt
class TransitTimeMinimaJob:

    def __init__(self, env, cy, route, speed):
        self.boat_direction = 'P' if speed/abs(speed) > 0 else 'N'
        self.boat_speed = abs(speed)
        self.shape_base_name = self.boat_direction + str(self.boat_speed) + 'A'
        file_header = str(cy.year()) + '_' + self.boat_direction + '_' + str(self.boat_speed)
        self.speed = speed
        self.first_day_index = cy.first_day_index()
        self.last_day_index = cy.last_day_index()
        self._start_index = cy.transit_start_index()
        self._end_index = cy.transit_end_index()
        self.transit_range = cy.transit_range()
        self._elapsed_times_df = route.elapsed_time_lookup[speed]
        self._elapsed_time_table = env.transit_time_folder().joinpath('et_' + str(speed))  # elapsed times in table, sorted by speed
        self.speed_folder = env.speed_folder(num2words(speed))
        self.transit_timesteps = self.speed_folder.joinpath(file_header + '_timesteps')  # transit times column
        self.savgol_data = self.speed_folder.joinpath(file_header + '_savgol_data')  # savgol column
        self.plot_data = self.speed_folder.joinpath(file_header + '_plot_data')  # full data calculated for transit time, can be plotted
        self.debug_data = self.speed_folder.joinpath(file_header + '_debug_data')  # full data with rows for plotting removed and s, m, e added
        self.transit_time_values = env.transit_time_folder().joinpath(file_header + '_')  # final results table

    def execute(self):
        init_time = perf_counter()
        if ft.file_exists(self.transit_time_values):  # results exists
            print(f'+     Transit time ({self.speed}) reading data file', flush=True)
            transit_time_values_df = ft.read_df(self.transit_time_values)
            return tuple([self.speed, transit_time_values_df, init_time])

        print(f'+     Transit time ({self.speed})', flush=True)
        if ft.file_exists(self.transit_timesteps):
            transit_timesteps_arr = ft.read_arr(self.transit_timesteps)
        else:
            row_range = range(len(self.transit_range))
            transit_timesteps_arr = [total_transit_time(row, self._elapsed_times_df, self._elapsed_times_df.columns.to_list()) for row in row_range]
            ft.write_arr(transit_timesteps_arr, self.transit_timesteps)

        if ft.file_exists(self.plot_data):
            plot_data_df = ft.read_df(self.plot_data)
        else:
            plot_data_df = self.minima_table(transit_timesteps_arr)  # call minima_table
            ft.write_df(plot_data_df, self.plot_data)

        plot_data_df = plot_data_df.dropna(axis=0).sort_index().reset_index(drop=True)
        plot_data_df = plot_data_df.astype({'departure_index': int, 'start_index': int, 'min_index': int, 'end_index': int})
        ft.write_df(plot_data_df, self.debug_data)

        arc_df = self.create_arcs(plot_data_df)
        ft.write_df(arc_df, self.speed_folder.joinpath('arc_df'))
        return tuple([self.speed, arc_df, init_time])

    def execute_callback(self, result):
        print(f'-     Transit time ({self.speed}) {dtt.mins_secs(perf_counter() - result[2])} minutes', flush=True)
    def error_callback(self, result):
        print(f'!     Transit time ({self.speed}) process has raised an error: {result}', flush=True)

    def minima_table(self, transit_array):
        tt_df = pd.DataFrame()
        tt_df['departure_index'] = self.transit_range
        tt_df['plot'] = 0
        tt_df = tt_df.assign(tts=transit_array)
        if ft.file_exists(self.savgol_data):
            tt_df['midline'] = ft.read_arr(self.savgol_data)
        else:
            tt_df['midline'] = savgol_filter(transit_array, 50000, 1)
            ft.write_arr(tt_df['midline'], self.savgol_data)
        min_segs = tt_df['tts'].lt(tt_df['midline']).to_list()  # list of True or False the same length as tt_df
        clump = []
        for row, val in enumerate(min_segs):  # rows in tt_df, not the departure index
            if val:
                clump.append(row)  # list of the rows within the clump of True min_segs
            elif len(clump) > FIVE_HOURS_OF_TIMESTEPS:  # ignore clumps caused by small fluctuations in midline or end conditions ( ~ 5-hour tide window )
                segment_df = tt_df[clump[0]:clump[-1]]  # subset of tt_df from first True to last True in clump
                seg_min_df = segment_df[segment_df['tts'] == segment_df.min()['tts']]  # segment rows equal to the minimum
                median_index = seg_min_df['departure_index'].median()  # median departure index of segment minima rows
                abs_diff = segment_df['departure_index'].sub(median_index).abs()  # absolute value of difference between actual index and median index
                min_index = segment_df.at[abs_diff[abs_diff == abs_diff.min()].index[0], 'departure_index']  # actual index closest to median index, may not be minimum tts
                offset = segment_df['tts'].min() + TIMESTEP_MARGIN  # minimum tss + margin for window
                if min_index != clump[0] and min_index != clump[-1]:  # ignore minima at edges
                    start_segment = segment_df[segment_df['departure_index'].le(min_index)]  # portion of segment from start to minimum
                    end_segment = segment_df[segment_df['departure_index'].ge(min_index)]  # portion of segment from minimum to end
                    start_row = start_segment[start_segment['tts'].le(offset)].index[0]
                    end_row = end_segment[end_segment['tts'].ge(offset)].index[0]
                    start_index = start_segment.at[start_row, 'departure_index']
                    end_index = end_segment.at[end_row, 'departure_index']
                    tt_df_start_row = tt_df[tt_df['departure_index'] == start_index].index[0]
                    tt_df_min_row = tt_df[tt_df['departure_index'] == min_index].index[0]
                    tt_df_end_row = tt_df[tt_df['departure_index'] == end_index].index[0]
                    tt_df.at[tt_df_min_row, 'start_index'] = int(start_index)
                    tt_df.at[tt_df_min_row, 'min_index'] = int(min_index)
                    tt_df.at[tt_df_min_row, 'end_index'] = int(end_index)
                    tt_df.at[tt_df_start_row, 'plot'] = max(transit_array)
                    tt_df.at[tt_df_min_row, 'plot'] = min(transit_array)
                    tt_df.at[tt_df_end_row, 'plot'] = max(transit_array)
                clump = []
        tt_df['transit_time'] = pd.to_timedelta(tt_df['tts']*TIMESTEP, unit='s').round('min')
        tt_df = mh.shrink_dataframe(tt_df)
        return tt_df

    def final_output(self, input_frame):

        output_frame = input_frame[['start_index', 'end_index', 'start_rounded', 'start_degrees', 'end_rounded', 'end_degrees', 'min_rounded', 'min_degrees', 'fraction']].copy()
        output_frame.rename({'start_rounded': 'start_date', 'end_rounded': 'end_date', 'min_rounded': 'min_date', 'start_degrees': 'arc_start', 'end_degrees': 'arc_end'}, axis=1, inplace=True)

        fraction_list = output_frame[output_frame['fraction'] == True].index.tolist()

        for row in fraction_list:
            if output_frame.loc[row, 'arc_end'] == 0:
                # special case where arc ends exactly at 00:00
                output_frame.loc[row, 'end_date'] = pd.to_datetime(output_frame.loc[row, 'start_date'].date()) + pd.Timedelta('23:59:59')  # resetting end to ensure correct trim at end of year
                output_frame.loc[row, 'arc_end'] = 360
                output_frame.loc[row, 'fraction'] = 'ADJ'
            else:
                # create new row for fraction - from 0 to end_degrees
                output_frame.loc[row + 0.5] = output_frame.loc[row].to_list()
                if output_frame.loc[row, 'min_date'].date() == output_frame.loc[row, 'start_date'].date(): output_frame.loc[row + 0.5, 'min_degrees'] = 'None'
                output_frame.loc[row + 0.5, 'start_date'] = output_frame.loc[row, 'end_date'].date()
                output_frame.loc[row + 0.5, 'arc_start'] = 0
                output_frame.loc[row + 0.5, 'arc_end'] = output_frame.loc[row, 'arc_end']
                output_frame.loc[row + 0.5, 'fraction'] = 'NEW'
                # fix old row - from start to zero
                if output_frame.loc[row, 'min_date'].date() == output_frame.loc[row, 'end_date'].date():
                    if output_frame.loc[row, 'min_degrees'] == 0:
                        output_frame.loc[row, 'min_degrees'] = 360
                    else:
                        output_frame.loc[row, 'min_degrees'] = 'None'
                output_frame.loc[row, 'end_date'] = pd.to_datetime(output_frame.loc[row, 'start_date'].date()) + pd.Timedelta('23:59:59')  # resetting end to ensure correct trim at end of year
                output_frame.loc[row, 'arc_end'] = 360
                output_frame.loc[row, 'fraction'] = 'ADJ'

        output_frame = output_frame.sort_index().reset_index(drop=True)

        # add clock time columns
        datetime_list = output_frame[output_frame['start_date'].apply(lambda x: not isinstance(x, pd.Timestamp))].index.to_list()
        for r in datetime_list: output_frame.loc[r, 'start_date'] = pd.Timestamp(output_frame.loc[r, 'start_date'])
        output_frame['start_time'] = output_frame['start_date'].apply(lambda x: x.time())
        output_frame['start_date'] = output_frame['start_date'].apply(lambda x: x.date())

        datetime_list = output_frame[output_frame['end_date'].apply(lambda x: not isinstance(x, pd.Timestamp))].index.to_list()
        for r in datetime_list: output_frame.loc[r, 'end_date'] = pd.Timestamp(output_frame.loc[r, 'end_date'])
        output_frame['end_time'] = output_frame['end_date'].apply(lambda x: x.time())

        datetime_list = output_frame[output_frame['min_date'].apply(lambda x: not isinstance(x, pd.Timestamp))].index.to_list()
        for r in datetime_list: output_frame.loc[r, 'min_date'] = pd.Timestamp(output_frame.loc[r, 'min_date'])
        output_frame['min_time'] = output_frame['min_date'].apply(lambda x: x.time())

        # trim to first and last day
        output_frame['start_index'] = output_frame['start_date'].apply(lambda x: dtt.int_timestamp(x))
        output_frame['end_index'] = output_frame['end_date'].apply(lambda x: dtt.int_timestamp(x))
        output_frame = output_frame[output_frame['start_index'] >= self.first_day_index]
        output_frame = output_frame[output_frame['end_index'] < self.last_day_index]
        output_frame = output_frame[['start_date', 'start_time', 'arc_start', 'arc_end', 'end_time', 'min_time', 'min_degrees']]

        output_frame = output_frame.sort_index().reset_index(drop=True)
        output_frame['shape_name'] = self.boat_direction + str(self.boat_speed) + 'A'

        arc_count = 1
        for row in range(0, len(output_frame) - 1):
            output_frame.loc[row, 'shape_name'] = output_frame.loc[row, 'shape_name'] + str(arc_count)
            if output_frame.loc[row, 'start_date'] == output_frame.loc[row + 1, 'start_date']: arc_count += 1
            else: arc_count = 1

        for row in output_frame.index:
            if output_frame.loc[row, 'min_degrees'] != 'None':
                output_frame.loc[row + 0.5] = output_frame.loc[row]
                output_frame.loc[row + 0.5, 'shape_name'] = output_frame.loc[row, 'shape_name'] + 'm'
        output_frame = output_frame.sort_index().reset_index(drop=True)

        return output_frame

    def create_arcs(self, input_frame):

        arc_frame = input_frame[['tts', 'start_index', 'min_index', 'end_index']]
        Arc.name = self.shape_base_name

        arcs = [RoundedArc(*row.values.tolist()) for i, row in arc_frame.iterrows()]
        fractured_arc_list = list(filter(lambda n: n.fractured, arcs))
        arc_list = list(filter(lambda n: not n.fractured, arcs))

        for arc in fractured_arc_list:
            arc_list.append(FractionalArcStartDay(*arc.fractional_arc_args()))
            arc_list.append(FractionalArcEndDay(*arc.fractional_arc_args()))

        arc_df = pd.DataFrame([arc.df_angles() for arc in arc_list])
        arc_df.columns = Arc.columns

        return arc_df
