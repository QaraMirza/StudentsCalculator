import os
import pandas as pd
from sklearn.model_selection import train_test_split

# data.iloc[строка, столбец]


def split_data(file_path, file_type):
    if file_type == 'teachers':
        return split_data_teachers(file_path)
    elif file_type == 'students':
        return split_data_students(file_path)
    else:
        raise ValueError(f'Нет типа {file_type}. Только teachers и students')


def split_data_teachers(file_path):
    # Построение абсолютного пути к файлу с данными
    data = pd.read_excel(file_path)

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    # Вывод названия столбца целевой переменной
    y_column_name = data.columns[0]
    print("Название столбца целевой переменной:", y_column_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def split_data_students(file_path):
    data = pd.read_excel(file_path)
    print(f"Размер DataFrame: {data.shape}")
    print(f"Индексы DataFrame: {data.index.size}")
    print(f"Столбцы DataFrame: {data.columns.size}")
    X = data.iloc[:-1, 1:].values.astype(float).T
    y = data.iloc[-1, 1:].values.astype(float)
    print(f"Размер X: {X.shape}, размер y: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Размер обучающей выборки X: {X_train.shape}")
    print(f"Размер тестовой выборки X: {X_test.shape}")
    print(f"Размер обучающей выборки y: {y_train.shape}")
    print(f"Размер тестовой выборки y: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def get_column_names(file_path, file_type):
    if file_type == 'teachers':
        return get_column_names_teachers(file_path)
    elif file_type == 'students':
        return get_column_names_students(file_path)
    else:
        raise ValueError(f'Нет типа {file_type}. Только teachers и students')


def get_column_names_teachers(file_path):
    data = pd.read_excel(file_path)
    column_names = data.columns[1:]
    return column_names


def get_column_names_students(file_path):
    data = pd.read_excel(file_path)
    column_names = data.iloc[:-1, 0]
    return column_names