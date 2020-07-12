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

class ModelFn(beam.CombineFn):
    def create_accumulator(self):
        return pd.DataFrame()

    def add_input(self, df, window):
        return df.append(window, ignore_index=True)

    def merge_accumulators(self, dfs):
        return pd.concat(dfs)

    def extract_output(self, df):
        if len(df) < 1:
            return {}
        # if model is order-dependent, then we also need this
        # df = df.sort_values(by='scheduled').reset_index(drop=True);
        orig = df['delay'].values
        xarr = np.delete(orig, [np.argmin(orig), np.argmax(orig)])
        if len(xarr) > 2:
            # need at least three items to compute a valid standard deviation
            prediction = np.mean(xarr)
            acceptable_deviation = 4 * np.std(xarr)
        else:
            prediction = 0.0
            acceptable_deviation = 60.0
        return {
            'prediction': prediction,
            'acceptable_deviation': acceptable_deviation
        }
    
class OnlineModelFn(beam.CombineFn):
    def create_accumulator(self):
        return (0.0, 0.0, 0) # x, x^2, count

    def add_input(self, sum_count, input_dict):
        (sum, sumsq, count) = sum_count
        input = input_dict['delay']
        return (sum + input, sumsq + input*input, count + 1)

    def merge_accumulators(self, accumulators):
        sums, sumsqs, counts = zip(*accumulators)
        return sum(sums), sum(sumsqs), sum(counts)

    def extract_output(self, sum_count):
        (sum, sumsq, count) = sum_count
        if count:
            mean = sum / count
            variance = (sumsq / count) - mean*mean
            # -ve value could happen due to rounding
            stddev = np.sqrt(variance) if variance > 0 else 0
            return {
                'prediction': mean,
                'acceptable_deviation': 4 * stddev
            }
        else:
            return {
                'prediction': float('NaN'),
                'acceptable_deviation': float('NaN')
            }

def is_anomaly(data, model_state):
    result = data.copy()
    if not isinstance(model_state, beam.pvalue.EmptySideInput):
        result['is_anomaly'] = np.abs(data['delay'] - model_state['prediction']) > model_state['acceptable_deviation']
        if result['is_anomaly']:
            print(result)
    return result

WINDOW_INTERVAL = 2 * 60 * 60. # 2 hours, in seconds
PANE_INTERVAL = 10*60 # 10 minutes, in seconds

def is_latest_slice(element, timestamp=beam.DoFn.TimestampParam, window=beam.DoFn.WindowParam):
    # in a sliding window, find whether we are in the last pane
    secs = (window.max_timestamp().micros - timestamp.micros)/(1000*1000)
    if secs < PANE_INTERVAL:
        yield element

def to_csv(d):
    return ','.join([d['scheduled'].strftime('%Y-%m-%dT%H:%M:%S'), 
                     str(d['delay']), 
                     str(d['is_anomaly'])
                    ])
        
def run():
    p = beam.Pipeline()
    data = (p 
        | 'files' >> beam.io.ReadFromText('delays.csv')
        | 'parse' >> beam.Map(parse_line))

    windowed = (data
        | 'window' >> beam.WindowInto(
                beam.window.SlidingWindows(WINDOW_INTERVAL, PANE_INTERVAL),
                accumulation_mode=beam.trigger.AccumulationMode.DISCARDING))
    
    model_state = (windowed 
        | 'model' >> beam.transforms.CombineGlobally(OnlineModelFn()).without_defaults())

    anomalies = (windowed 
        | 'latest_slice' >> beam.FlatMap(is_latest_slice)
        | 'find_anomaly' >> beam.Map(is_anomaly, beam.pvalue.AsSingleton(model_state)))
    
    anomalies | 'dict_csv' >> beam.Map(to_csv) |'write' >> beam.io.WriteToText('anomalies.csv', num_shards=1)
    
    p.run().wait_until_finish()
    
if __name__ == '__main__':
    run()

