import os
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'resources/normalized.xlsx'

def split_data():

    # Построение абсолютного пути к файлу с данными
    data = pd.read_excel(file_path)

    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def get_column_names():
    data = pd.read_excel(file_path)
    column_names = data.columns
    return column_names
