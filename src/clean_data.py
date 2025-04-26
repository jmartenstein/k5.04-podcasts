import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer

import sklearn.preprocessing as pre
import sklearn.impute as imp

import argparse

from datetime import datetime

### CONSTANTS ###

DATA_DIR = "../data/kaggle"
ORIG_DIR = "../data/original"

TARGET1 = "Listening_Time_minutes"
TARGET2 = "Episode_Completion_percentage"

### FUNCTIONS ###

def get_datetime_string( str_note ):

    now = datetime.now()

    s_date = now.strftime("%Y%m%d")
    s_time = now.strftime("%H%M%S")

    return f"{str_note}.{s_date}.{s_time}"

def bin_feature( s_feature, l_bins, l_labels ):
    pass

### MAIN ###

parser = argparse.ArgumentParser( description='Pre-process data from Podcasts dataset' )
parser.add_argument('-w', '--write', action='store_true', help='Write processed output to files')

args = vars( parser.parse_args() )

# load train data
df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
df_test  = pd.read_csv(f"{DATA_DIR}/test.csv")
df_orig  = pd.read_csv(f"{ORIG_DIR}/podcast_dataset.csv")

# add the target column for the test data
df_test[TARGET1] = np.nan

df_ = pd.concat([df_train, df_test])
#df_.info()

df_["Length_Clean" ] = df_[ "Episode_Length_minutes" ].apply( lambda v: 120.5 if v > 120.5 else v )

df_["IsGuest"] = df_["Guest_Popularity_percentage"].apply( lambda v: 0 if np.isnan(v) else 1 )
df_["Num_Ads"] = df_["Number_of_Ads"].fillna(0).copy()
df_["Name_And_Episode"] = df_["Podcast_Name"] + " " + df_["Episode_Title"]
df_["Pub_Day_And_Time"] = df_["Publication_Day"] + " " + df_["Publication_Time"]

iter_impute = imp.IterativeImputer(missing_values=np.nan, add_indicator=True, max_iter=20)
impute_cols = ["Length_Clean", "Host_Popularity_percentage","Num_Ads",
               "Guest_Popularity_percentage" ]

iter_impute.fit(df_[impute_cols])
#df_train_clean["Length_Iter_Impute" ] = iter_impute.transform(df_train_clean[impute_cols])[:,0]

#iter_impute.fit(df_test_clean[impute_cols])
length_features = iter_impute.transform(df_[impute_cols])

#median_impute = imp.SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True)
#median_impute.fit(df_[['Length_Clean']])
#length_features = median_impute.transform(df_[['Length_Clean']])

df_['Length_Impute'] = length_features[:,0]
df_['Length_Missing'] = length_features[:,1]

df_[ TARGET2 ] = df_[TARGET1] / df_["Length_Impute"]
df_[ TARGET2 ] = df_[TARGET2].apply( lambda v: 1 if v > 1 else v )

feature = "Num_Ads"
hist, bin_edges = np.histogram( df_[feature],
                                bins=[ df_[feature].min(),
                                       0.5, 1.5, 2.5,
                                       df_[feature].max()
                                     ]
                               )
df_[f"{feature}_Bin"] = pd.cut( df_[feature], bins=bin_edges,
                             labels=[0, 1, 2, 3], include_lowest=True
                           )

length_column = df_["Length_Impute"]
df_["Cat_Mins"] = (length_column.astype(np.int64)).astype('category')
df_["Cat_Secs"] = ((length_column * 60).astype(np.int64)).astype('category')

df_["Ads_Mins"] = df_["Number_of_Ads"].astype(str) + " " + df_["Cat_Mins"].astype(str)
df_["All_Cats"] = df_["Podcast_Name"] + " " + df_["Publication_Day"] + " " \
                + df_["Publication_Time"] +  " " + df_["Num_Ads_Bin"].astype(str) + " " \
                + df_["Episode_Sentiment"]

df_["Ad_Rate_Per_Hour"] = df_["Num_Ads"] * (df_["Length_Impute"] / 60)
df_["Ads_And_Host_Pop"] = (df_["Num_Ads"] * 100) + df_["Host_Popularity_percentage"]
df_["Ads_Sentiment"] = str(df_["Num_Ads_Bin"]) + " " + df_["Episode_Sentiment"]


# prep the cleaned training data to write
df_train_clean = df_[ df_[ TARGET1 ].notna() ].copy()

# prep the cleaned test data to write
df_test_clean = df_[ df_[ TARGET1 ].isna() ].copy()
df_test_clean.drop( TARGET1, axis=1, inplace=True )

features = [ "Ads_Mins", "Cat_Secs", "All_Cats" ]
targets = [ TARGET1, TARGET2 ]

encoded_features = []

enc = pre.TargetEncoder( cv=3 )

for f in features:
    i = 1
    for t in targets:
        encoded_feature = f"T{i}_{f}"
        enc.fit( df_train_clean[[f]], df_train_clean[t] )
        df_train_clean[encoded_feature] = enc.transform(df_train_clean[[f]])
        df_test_clean[encoded_feature] = enc.transform(df_test_clean[[f]])
        i+=1

    encoded_features.append(encoded_feature)

poly_features = [ "Length_Impute", "Num_Ads", "Host_Popularity_percentage" ] + encoded_features

poly = pre.PolynomialFeatures(degree=3)
poly.fit( df_train_clean[ poly_features ] )
feature_names = poly.get_feature_names_out()

X_poly_train = poly.transform( df_train_clean[ poly_features ] )
df_poly_train = pd.DataFrame(X_poly_train)
df_poly_train.columns = feature_names

X_poly_test = poly.transform( df_test_clean[ poly_features ] )
df_poly_test = pd.DataFrame(X_poly_test)
df_poly_test.columns = feature_names

unique_poly_features = list(set(df_poly_train.columns) - set(df_train_clean.columns))
#print(unique_poly_features)

df_train_clean = df_train_clean.join(df_poly_train[unique_poly_features].add_prefix('Poly '))
df_test_clean = df_test_clean.join(df_poly_test[unique_poly_features].add_prefix('Poly '))


if args["write"]:

    outfile_str = get_datetime_string( "enc_and_poly" )

    train_outfile = f"train.{outfile_str}.csv"
    print(f"write train shape: {df_train_clean.shape} to file: {train_outfile}")
    df_train_clean.to_csv(f"{DATA_DIR}/{train_outfile}", index=False)

    test_outfile = f"test.{outfile_str}.csv"
    print(f"write test shape:  {df_test_clean.shape} to file: {test_outfile}")
    df_test_clean.to_csv(f"{DATA_DIR}/{test_outfile}", index=False)

else:

    print(df_train_clean.info())

    print("No output written to files")
    print(f"  train shape: {df_train_clean.shape}")
    print(f"  test shape:  {df_test_clean.shape}")

