
# import required module
import glob
import pandas as pd

# Set parquet files path
PATH = 's3://fruit-data/Results'

# iterate over files in folder PATH
for i, filename in enumerate(glob.iglob(f'{PATH}/*.parquet')):
    print(filename)
    df = pd.read_parquet(filename)
    df.to_csv(f'{PATH}/preprocessed_images_{i}.csv')