from os.path import exists
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from project_globals import seconds, timestep, minima_threshold

class TransitTimeJob:

    def __init__(self, route, speed, environ, chart_yr, intro=''):
        self.__speed = speed
        self.__intro = intro
        self.__output_file = Path(str(environ.transit_time_folder()) + '/TT_' + str(self.__speed) + '.csv')
        self.__start = chart_yr.first_day_minus_one()
        self.__end = chart_yr.last_day_plus_one()
        self.__seconds = seconds(self.__start, self.__end)
        self.__no_timesteps = int(self.__seconds / timestep)
        self.__speed_df = pd.DataFrame()
        for re in route.route_edges():
            col_name = re.name() + ' ' + str(self.__speed)
            self.__speed_df[col_name] = re.elapsed_time_dataframe()[col_name]

    def execute(self):
        if exists(self.__output_file):
            print(f'+     {self.__intro} Transit time ({self.__speed}) reading data file', flush=True)
            tt_minima_df = pd.read_csv(self.__output_file, header='infer')
            return tuple([self.__speed, tt_minima_df])
        else:
            print(f'+     {self.__intro} Transit time ({self.__speed}) calculation starting', flush=True)
            xx = np.fromiter([total_transit_time(row, self.__speed_df) for row in range(0, self.__no_timesteps)],
                             dtype=int)
            tt_minima_df = minima_table(xx)
            tt_minima_df.to_csv(self.__output_file)
            return tuple([self.__speed, tt_minima_df])

    def execute_callback(self, result):
        print(
            f'-     {self.__intro} {self.__speed} {"SUCCESSFUL" if isinstance(result[1], pd.DataFrame) else "FAILED"} {self.__no_timesteps}',
            flush=True)
    def error_callback(self, result):
        print(f'!     {self.__intro} {self.__speed} process has raised an error: {result}', flush=True)

def total_transit_time(init_row, d_frame):
    row = init_row
    tt = 0
    for col in d_frame.columns:
        val = d_frame.loc[row, col]
        tt += val
        row += val
    return tt

def minima_table(transit_time_array):
    threshold = minima_threshold
    tt_df = pd.DataFrame()
    tt_df['tt'] = transit_time_array
    tt_df['Savitzky-Golay'] = savgol_filter(transit_time_array, 500, 1)
    tt_df['sg-midline'] = savgol_filter(transit_time_array, 50000, 1)
    tt_df['gradient'] = np.gradient(tt_df['Savitzky-Golay'].to_numpy(), edge_order=2)
    tt_df['zero_ish'] = tt_df['gradient'].abs().apply(lambda x: True if x < threshold else None)
    tt_df['z2'] = tt_df['zero_ish'][tt_df['Savitzky-Golay'] < tt_df['sg-midline']]

    # convert clumps into single best estimate of minima

    clump = []
    minima = []
    savgol = tt_df['Savitzky-Golay'].to_numpy()
    zero_ish = tt_df['zero_ish'].to_numpy()
    midline = tt_df['sg-midline'].to_numpy()

    # for index, sg in enumerate(savgol):
    #     if zero_ish[index] and sg > midline[index] and len(clump) > 0:
    #         minima.append(int(np.median(clump)))
    #         clump = []
    #     elif zero_ish[index] and sg < midline[index]: clump.append(index)
    #
    # for val in minima: tt_df.loc[val, 'minima'] = True
    # tt_df.to_csv(Path(str('c:\\users\\jason\\downloads\\East River\\tt_df.csv')))
    # return tt_df.drop(columns=['gradient', 'zero_ish'])
    return tt_df
