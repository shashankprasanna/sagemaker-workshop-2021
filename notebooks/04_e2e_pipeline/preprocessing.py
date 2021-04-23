import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", 'imblearn'])
    
    
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.compose import make_column_transformer

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    parser.add_argument('--random-split', type=int, default=0)
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    input_data_path = os.path.join('/opt/ml/processing/input', 'dataset.csv')
    
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)
    
    # move the target column to the begining based on XGBoost
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('default payment next month')))
    df = df.loc[:, cols]

    #  rename to `LABEL`
    df.rename(columns={"default payment next month": "LABEL"}, inplace=True)
    df['LABEL'] = df['LABEL'].astype('int')
    
    # upsampling minority class
    sm = SMOTE(random_state=42)
    df, _ = sm.fit_resample(df, df['SEX'])
    
    # split data to train and test    
    df_train = df.sample(frac=args.train_test_split_ratio, random_state=args.random_split)
    df_test = df.drop(df_train.index)
    
    print('Train data shape after preprocessing: {}'.format(df_train.shape))
    print('Test data shape after preprocessing: {}'.format(df_test.shape))
    
    train_output_path = os.path.join('/opt/ml/processing/output/train', 'train_data.csv')    
    test_output_path = os.path.join('/opt/ml/processing/output/test', 'test_data.csv')
    
    print('Saving training features to {}'.format(train_output_path))
    df_train.to_csv(train_output_path, index=False)
    
    print('Saving test features to {}'.format(test_output_path))
    df_test.to_csv(test_output_path, index=False)
