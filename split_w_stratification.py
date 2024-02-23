import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_stratified_into_train_val_test(df_input, stratify_colnames='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
  '''
  Splits a Pandas dataframe into three subsets (train, val, and test)
  following fractional ratios provided by the user, where each subset is
  stratified by the values in a specific column (that is, each subset has
  the same relative frequency of the values in the column). It performs this
  splitting by running train_test_split() twice.

  Parameters
  ----------
  df_input : Pandas dataframe
      Input dataframe to be split.
  stratify_colname : list [["coln1", "coln2"]]
      Col names
      The name of the column that will be used for stratification. Usually
      this column would be for the label.
  frac_train : float
  frac_val   : float
  frac_test  : float
      The ratios with which the dataframe will be split into train, val, and
      test data. The values should be expressed as float fractions and should
      sum to 1.0.
  random_state : int, None, or RandomStateInstance
      Value to be passed to train_test_split().

  Returns
  -------
  df_train, df_val, df_test :
      Dataframes containing the three splits.
  '''

  if frac_train + frac_val + frac_test != 1.0:
      raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                       (frac_train, frac_val, frac_test))

  # if stratify_colname not in df_input.columns:
  #     raise ValueError('%s is not a column in the dataframe' % (stratify_colname))
  # if stratify_colname2 not in df_input.columns:
  #     raise ValueError('%s is not a column in the dataframe' % (stratify_colname2))
  # if stratify_colname3 not in df_input.columns:
  #     raise ValueError('%s is not a column in the dataframe' % (stratify_colname3))
  # if stratify_colname4 not in df_input.columns:
  #     raise ValueError('%s is not a column in the dataframe' % (stratify_colname4))

  X = df_input # Contains all columns.

 
  y = df_input[stratify_colnames]


  # Split original dataframe into train and temp dataframes.
  df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=(1.0 - frac_train),
                                                        random_state=random_state)

  # Split the temp dataframe into val and test dataframes.
  relative_frac_test = frac_test / (frac_val + frac_test)
  df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                    y_temp,
                                                    stratify=y_temp,
                                                    test_size=relative_frac_test,
                                                    random_state=random_state)

  assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

  return df_train, df_val, df_test
