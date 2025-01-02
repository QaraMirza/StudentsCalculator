import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.data.data_spliter import split_data


def load_linear():
    X_train, X_test, y_train, y_test = split_data()

    # Создание и обучение модели
    model = LinearRegression()

    model.fit(X_train, y_train)
    # Тестирование
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Средняя ошибка на тестовом наборе: {mae}")

    # Создание таблицы с тестовыми и предсказанными данными
    results = pd.DataFrame({
        'Тестовые данные': y_test,
        'Предсказанные данные': y_pred.flatten()
    })

    print(results)

    # Убедитесь, что директория существует
    os.makedirs('results/linear', exist_ok=True)

    results.to_csv('results/linear/linear.csv', index=False)

    with open('results/linear/linear_error.txt', 'w', encoding='utf-8') as file:
        file.write('mae = ' + str(mae))


    # Вывод весов модели
    coefficients = model.coef_
    intercept = model.intercept_

    print("Веса модели (коэффициенты):", coefficients)
    print("Смещение модели (intercept):", intercept)

    # Визуализация весов модели
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(coefficients)), coefficients)
    plt.xlabel('Признаки')
    plt.ylabel('Веса')
    plt.title('Веса линейной регрессии')
    plt.show()