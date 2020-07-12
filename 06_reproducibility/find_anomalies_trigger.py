#!/usr/bin/env python3

import apache_beam as beam
from apache_beam.runners.interactive import interactive_runner
import apache_beam.runners.interactive.interactive_beam as ib
import datetime
import numpy as np
import pandas as pd

def parse_line(line):
    import datetime
    timestamp, delay = line.split(",")
    timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    return beam.window.TimestampedValue(
        {
            'scheduled': timestamp,
            'delay': float(delay)
        },
        timestamp.timestamp() # unix timestamp
    )

def is_anomaly(orig):
    import numpy as np
    outcome = orig[-1] # the last item

    # discard min & max value & current (last) item
    xarr = np.delete(orig, [np.argmin(orig), np.argmax(orig), len(orig)-1])
    if len(xarr) < 3:
        # need at least three items to compute a valid standard deviation
        return False

    # Fit a model (4-sigma deviations)
    prediction = np.mean(xarr)
    acceptable_deviation = 4 * np.std(xarr)
    result = np.abs(outcome - prediction) > acceptable_deviation
    return result

class AnomalyFn(beam.CombineFn):
    def create_accumulator(self):
        return pd.DataFrame()

    def add_input(self, df, window):
        return df.append(window, ignore_index=True)

    def merge_accumulators(self, dfs):
        return pd.concat(dfs)

    def extract_output(self, df):
        if len(df) < 1:
            return {}
        df = df.sort_values(by='scheduled').reset_index(drop=True);
        last_row = {}
        for col in df.columns:
            last_row[col] = df[col].iloc[len(df)-1]
        last_row['is_anomaly'] = is_anomaly(df['delay'].values)
        if last_row['is_anomaly']:
            print(df['delay'], last_row)
        return last_row


def run():
    p = beam.Pipeline()
    data = (p 
        | 'files' >> beam.io.ReadFromText('delays.csv')
        | 'parse' >> beam.Map(parse_line))
   
    windowed = (data
        | 'window' >> beam.WindowInto(
                beam.window.FixedWindows(2*60*60),
                trigger=beam.trigger.Repeatedly(beam.trigger.AfterCount(1)), # every element
                accumulation_mode=beam.trigger.AccumulationMode.ACCUMULATING))
    
    anomalies = (windowed 
        | 'find_anomaly' >> beam.transforms.CombineGlobally(AnomalyFn()).without_defaults())
    
    anomalies | 'write' >> beam.io.WriteToText('anomalies.json')
    p.run().wait_until_finish()
    
if __name__ == '__main__':
    run()

