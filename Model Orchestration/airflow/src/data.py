import pandas as pd

def preprocess_df(filename):
    df = pd.read_parquet(filename)
    df['duration'] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df
