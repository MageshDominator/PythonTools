# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:37:28 2019

@author: MAGESHWARAN
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False)

def EncodeMultiNomialData(data, columns, del_original = True):
    """Function to perform one hot encoding for multinominal data\n

    parameters:
        data : dataframe
        columns : columns to be one hot encoded (contains multinominal data)
        del_original : whether to delete the original array after encoding\n

    returns: Dataframe with One hot encoded variables for data[column]
    """
    for column in columns:
        # Encode labels into numbers
        data[column] = label_encoder.fit_transform(data[column])

        # convert 1-D array to 2-D
        temp = data[column].values
        temp = temp.reshape(-1, 1)

        # use one hot encoder
        store = one_hot_encoder.fit_transform(temp)

        # create new column indexes for each class
        index_ = [column + "_" + str(i) for i in range(len(store[0]))]
        store_df = pd.DataFrame(store, columns=index_)

        # concat new one hot encoded features to dataframe
        data = pd.concat([data, store_df], axis=1)
        if del_original:
            data = data.drop(column, axis=1)
    return data