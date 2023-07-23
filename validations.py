import pandas as pd
from tt_file_tools import file_tools as ft
from tt_geometry.geometry import time_to_degrees

class EastRiverValidation:

    first_slack_lookup = {'flood_slack': 'ebb_begins', 'slack_flood': 'flood_begins', 'slack_ebb': 'ebb_begins', 'ebb_slack': 'flood_begins'}

    def __init__(self, cy, waypoints):
        self.slack_df = None
        print('Calculating slack water times at Hell Gate')
        hell_gate = list(filter(lambda wp: not bool(wp.unique_name.find('Hell_Gate')), waypoints))[0]
        if ft.csv_npy_file_exists(hell_gate.interpolation_data_file):
            self.slack_df = ft.read_df(hell_gate.interpolation_data_file)
            # first_event = self.hell_gate_df.at[0, 'Event']
            # second_event = self.hell_gate_df.at[1, 'Event']
            # first_slack = EastRiverValidation.first_slack_lookup[str(first_event+'_'+second_event)]
            # print(f'{first_event}_{second_event} {first_slack}')
            self.slack_df = self.slack_df[self.slack_df['Event'] == 'slack'].copy()
            self.slack_df['date'] = self.slack_df['date_time'].apply(pd.to_datetime).dt.date
            self.slack_df['time'] = self.slack_df['date_time'].apply(pd.to_datetime).dt.time
            self.slack_df['angle'] = self.slack_df['time'].apply(time_to_degrees)
            self.slack_df = self.slack_df.filter(['date', 'angle'])
            self.slack_df = self.slack_df[self.slack_df['date'] >= cy.first_day.date()]
            self.slack_df = self.slack_df[self.slack_df['date'] <= cy.last_day.date()]
