#!/usr/bin/env python3

def parse_line(line):
    import datetime
    timestamp, delay = line.split(",")
    timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    return beam.window.TimestampedValue(
        {
            'scheduled': timestamp,
            'delay': delay
        },
        timestamp.timestamp() # unix timestamp
    )

def run(to_bq):
    import apache_beam as beam
    p = beam.Pipeline()
    
    data = (p 
            | 'files' >> beam.io.ReadFromText('delays.csv')
            | 'parse' >> beam.Map(parse_line))
    
    windowed = (data
                | 'window' >> beam.WindowInto(
                    beam.FixedWindows(2*60*60),
                    trigger=beam.AfterCount(1), # every element
                    accumulation_mode=beam.AccumulationMode.ACCUMULATING))
    
         | 'find_anomaly' >> beam.Map(is_anomaly)
    