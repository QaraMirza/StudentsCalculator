import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data():

    # Построение абсолютного пути к файлу с данными
    file_path = 'resources/normalized.xlsx'
    data = pd.read_excel(file_path)

    X = data.iloc[:, 2:].values
    y = data.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
