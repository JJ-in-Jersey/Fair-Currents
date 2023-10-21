import pandas as pd
from tt_file_tools import file_tools as ft
from tt_geometry.geometry import time_to_degrees


def index_arc_df(frame, name):
    date_time_dict = {key: [] for key in sorted(list(set(frame['date'])))}
    date_angle_dict = {key: [] for key in sorted(list(set(frame['date'])))}
    columns = frame.columns.to_list()
    for i, row in frame.iterrows():
        # date_time_dict[row[columns.index('date')]].append(row[columns.index('time')])
        # date_angle_dict[row[columns.index('date')]].append(row[columns.index('angle')])
        date_time_dict[row.iloc[columns.index('date')]].append(row.iloc[columns.index('time')])
        date_angle_dict[row.iloc[columns.index('date')]].append(row.iloc[columns.index('angle')])

    df = pd.DataFrame(columns=['date', 'name', 'time', 'angle'])
    for key in date_time_dict.keys():
        times = date_time_dict[key]
        angles = date_angle_dict[key]
        for i in range(len(times)):
            df.loc[len(df.name)] = [key, name + ' ' + str(i + 1), times[i], angles[i]]
    return df


class HellGateSlackTimes:

    first_slack_lookup = {'flood_slack': 'ebb_begins', 'slack_flood': 'flood_begins', 'slack_ebb': 'ebb_begins', 'ebb_slack': 'flood_begins'}

    def __init__(self, cy, waypoints):

        print('\nCalculating slack water times at Hell Gate')
        hell_gate = list(filter(lambda wp: not bool(wp.unique_name.find('Hell_Gate')), waypoints))[0]
        if ft.csv_npy_file_exists(hell_gate.downloaded_data_filepath):
            slack_df = ft.read_df(hell_gate.downloaded_data_filepath)
            slack_df = slack_df[slack_df['Event'] == 'slack']
            slack_df.drop(columns=['Event', 'Speed (knots)', 'date_index', 'velocity'], inplace=True)
            slack_df = pd.concat([slack_df, pd.DataFrame(columns=['date_time'], data=slack_df['date_time'].apply(pd.to_datetime) + pd.Timedelta(hours=3))])
            slack_df['date_time'] = slack_df['date_time'].apply(pd.to_datetime)
            slack_df.sort_values('date_time')
            slack_df['date'] = slack_df['date_time'].apply(pd.to_datetime).dt.date
            slack_df['time'] = slack_df['date_time'].apply(pd.to_datetime).dt.time
            slack_df['angle'] = slack_df['time'].apply(time_to_degrees)
            slack_df = slack_df.filter(['date', 'time', 'angle'])
            slack_df = slack_df[slack_df['date'] >= cy.first_day.date()]
            slack_df = slack_df[slack_df['date'] <= cy.last_day.date()]
            self.hell_gate_slack = index_arc_df(slack_df, 'Hell Gate Line')