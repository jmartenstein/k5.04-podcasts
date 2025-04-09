import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "../data/kaggle/"
ORIG_DIR = "../data/original/"

df_train = pd.read_csv( DATA_DIR + "train.clean.20250408.140401.csv" )
#df_test  = pd.read_csv( DATA_DIR + "test.csv" )

def p25( x ):
    return x.quantile(0.25)

def p50( x ):
    return x.quantile(0.50)

def p75( x ):
    return x.quantile(0.75)

def p95( x ):
    return x.quantile(0.92)

#def group_percentiles( df, x_colname, y_colname, func ):
#    return df.groupby(x_colname)[y_colname].agg(func)

#x_colname      = 'Episode_Sentiment'
x_colname       = 'Publication_Time'
target_colname  = 'Listening_Time_minutes'

print(df_train[target_colname].describe())
print()

df_funcs = pd.DataFrame()

funcs = [ p25, p50, p75, p95 ]
for f in funcs:
    series = df_train.groupby(x_colname)[target_colname].agg(f)
    func_name = f.__name__
    df_funcs[func_name] = series

df_funcs.sort_values(by="p25", inplace=True)
print(df_funcs.T)

df_funcs.T.plot.line()
plt.show()
