import pickle
import pandas as pd
import numpy as np
import os
import sys


script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
dependencies_path = os.path.join(script_path, 'dependencies')
output_path = os.path.join(script_path, 'output')
input_path = os.path.join(script_path, 'input')
categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def predict (X_val, df):
    with open(os.path.join(input_path, 'model.pkl'), 'rb') as f_in:
        model = pickle.load(f_in)
    y_pred = model.predict(X_val)
    print("Standard deviation of duration:", y_pred.std())
    print("Mean duration:", y_pred.mean())
    y_pred = np.array(y_pred)
    predictions_df = pd.DataFrame({'ride_id': df['ride_id'],
                                   'prediction': y_pred})
    predictions_df.to_parquet(os.path.join(output_path, 'pred.parquet'), engine='pyarrow', compression=None,
                              index=False)


def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    taxi_type = sys.argv[3]
    with open(os.path.join(input_path, 'dv.pkl'), 'rb') as f_in:
        dv = pickle.load(f_in)
    df = read_data(os.path.join(input_path,f'{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'))
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    predict (X_val, df)


if __name__ == '__main__':
    run()
