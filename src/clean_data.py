import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

from sklearn  import preprocessing, impute
from datetime import datetime

### CONSTANTS ###

DATA_DIR = "../data/kaggle"
ORIG_DIR = "../data/original"

TARGET_COLUMN = "Listening_Time_minutes"


### FUNCTIONS ###

def get_outfile_name( in_file ):

    f_split = in_file.split(".")
    now = datetime.now()

    s_date = now.strftime("%Y%m%d")
    s_time = now.strftime("%H%M%S")

    return f"{f_split[0]}.clean.{s_date}.{s_time}.{f_split[1]}"


### MAIN ###

parser = argparse.ArgumentParser( description='Pre-process data from Podcasts dataset' )
parser.add_argument('-w', '--write', action='store_true', help='Write processed output to files')

args = vars( parser.parse_args() )

# load train data
df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
df_test  = pd.read_csv(f"{DATA_DIR}/test.csv")
df_orig  = pd.read_csv(f"{ORIG_DIR}/podcast_dataset.csv")

# add the target column for the test data
df_test["Listening_Time_minutes"] = np.nan

df_ = pd.concat([df_train, df_test])
#df_.info()

df_["IsGuest"] = df_["Guest_Popularity_percentage"].apply( lambda v: 0 if np.isnan(v) else 1 )
df_["Num_Ads"] = df_["Number_of_Ads"].fillna(0)

hist, bin_edges = np.histogram( df_["Num_Ads"],
                                bins=[ df_["Num_Ads"].min(),
                                       0.5, 1.5, 2.5,
                                       df_["Num_Ads"].max()
                                     ]
                               )
df_["Num_Ads_Bin"] = pd.cut( df_[ "Num_Ads" ], bins=bin_edges,
                             labels=[0, 1, 2, 3], include_lowest=True
                           )
#print("Histogram with 4 bins:", hist)
#print("Bin edges:", bin_edges)

#hist_ads = df_.hist( column="Number_of_Ads", bins=4 )
#df_["Number_of_Ads"].hist(bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
#plt.show()
#print(df_["Number_of_Ads"].head())

# prep the cleaned training data to write
df_train_clean = df_[ df_[ TARGET_COLUMN ].notna() ].copy()

# prep the cleaned test data to write
df_test_clean = df_[ df_[ TARGET_COLUMN ].isna() ].copy()
df_test_clean.drop( TARGET_COLUMN, axis=1, inplace=True )

if args["write"]:

    train_outfile = get_outfile_name("train.csv")
    print(f"write train shape: {df_train_clean.shape} to file: {train_outfile}")
    df_train_clean.to_csv(f"{DATA_DIR}/{train_outfile}", index=False)

    test_outfile = get_outfile_name("test.csv")
    print(f"write test shape:  {df_test_clean.shape} to file: {test_outfile}")
    df_test_clean.to_csv(f"{DATA_DIR}/{test_outfile}", index=False)

else:

    print(df_train_clean.info())

    print("No output written to files")
    print(f"  train shape: {df_train_clean.shape}")
    print(f"  test shape:  {df_test_clean.shape}")

