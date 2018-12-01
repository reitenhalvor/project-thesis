

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from cognite.v05.assets import get_assets
from cognite.v05.assets import get_asset
from cognite.v05.timeseries import get_datapoints_frame

# authenticate session with Cognite API
from cognite.config import configure_session
from credentials import COGNITE_API_KEY
configure_session(COGNITE_API_KEY, 'publicdata')

import os
ROOT_PATH = os.path.abspath(".").split("code2")[0]
MAIN_PATH = ROOT_PATH + "main/"
EXPORT_PATH = ROOT_PATH + "exports/"
DATA_PATH = ROOT_PATH + "data/"
SAVE_PATH = ROOT_PATH + "save/"


def load_data():
    input_tags = ['VAL_23-FT-92512:X.Value', 'VAL_23-PT-92532:X.Value', 'VAL_23-TT-92533:X.Value']
    control_tags = ['VAL_23_ZT_92543:Z.X.Value', 'VAL_23_ZT_92538:Z.X.Value', 'VAL_23-KA-9101_ASP:VALUE']
    output_tags = ['VAL_23-FT-92537-01:X.Value', 'VAL_23-TT-92539:X.Value', 'VAL_23-PT-92539:X.Value']

    feature_tags = input_tags + control_tags
    target_tags = output_tags

    start = datetime(2018, 1, 1)
    end = datetime(2018, 10, 1)
    time_interval_seconds = 30
    granularity = str(time_interval_seconds) + 's'
    aggregates = ['avg']

    data = get_datapoints_frame(feature_tags + target_tags,
                                start=start,
                                end=end,
                                granularity=granularity,
                                aggregates=aggregates)


    # extract column names for features and targets
    target_tags = [label for label in list(data.columns) if
                   any([label.startswith(var_name) for var_name in target_tags])]
    feature_tags = [label for label in list(data.columns) if
                    not any([label.startswith(var_name) for var_name in target_tags])]

    return data, {'target_tags': target_tags, 'feature_tags': feature_tags}

data, tags = load_data()
print(data.shape)
print(data.head())
print(data.tail())

T = pd.to_datetime(y.timestamp,unit="ms")
df2np = data.drop(['timestamp']).values

train_size = int(0.6 * len(data))
valid_size = int(0.2 * len(data))
test_size = len(data) - (train_size + valid_size)

means = np.nanmean(data).tolist()
stds = np.nanstd(data).tolist()



# standardize the data
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_data = x_scaler.fit_transform(data[tags['feature_tags']].values)
y_data = y_scaler.fit_transform(data[tags['target_tags']].values)

# merge all the data to a single data frame
all_data = np.concatenate([x_data,y_data], axis=1)

# divide into training and testing sets


print("== SUMMARY ==")
print("Total number of observations : {}".format(len(all_data)))
print("Size of train+valid+test     : {}".format(train_size+valid_size+test_size))
print("Size of training set         : {}".format(train_size))
print("Size of validation set       : {}".format(valid_size))
print("Size of testing set          : {}".format(test_size))



# save scalers as pickles
joblib.dump(x_scaler, SAVE_PATH + 'x_scaler.pkl')
joblib.dump(y_scaler, SAVE_PATH + 'y_scaler.pkl')

# save training, validation and testing sets as .npy files

