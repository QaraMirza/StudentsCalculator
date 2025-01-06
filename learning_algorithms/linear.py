import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.data.data_spliter import split_data, get_column_names


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
    results_dir = 'results/linear'

    results.to_csv('results/linear/linear.csv', index=False)

    with open('results/linear/linear_error.txt', 'w', encoding='utf-8') as file:
        file.write('mae = ' + str(mae))

    # Вывод весов модели
    coefficients = model.coef_
    intercept = model.intercept_

    print("Веса модели (коэффициенты):", coefficients)
    print("Смещение модели (intercept):", intercept)

    # Создание списка с коэффициентами и соответствующими именами столбцов
    feature_names = get_column_names()
    coefficients_list = list(zip(feature_names, coefficients))

    # Сортировка списка по значению коэффициента от большего к меньшему
    sorted_coefficients_list = sorted(coefficients_list, key=lambda x: x[1], reverse=False)

    print("Список коэффициентов и соответствующих признаков (отсортированный):")
    for feature, coef in sorted_coefficients_list:
        print(f"{feature}:\t\t\t\t\t\t {coef}")

    # Разделение отсортированного списка на имена признаков и коэффициенты
    sorted_feature_names, sorted_coefficients = zip(*sorted_coefficients_list)

    # Создание DataFrame для экспорта в Excel
    coefficients_df = pd.DataFrame({
        'Признак': sorted_feature_names,
        'Коэффициент': sorted_coefficients
    })

    # Сохранение DataFrame в файл Excel
    excel_file_path = os.path.join(results_dir, 'linear_coefficients.xlsx')
    coefficients_df.to_excel(excel_file_path, index=False)

    # Визуализация весов модели
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_feature_names, sorted_coefficients)
    plt.xlabel('Веса')
    plt.ylabel('Признаки')
    plt.title('Веса линейной регрессии (отсортированные)')

    # Настройка пространства вокруг графика
    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)

    # Сохранение изображения в файл
    image_file_path = os.path.join(results_dir, 'linear_weights_sorted.png')
    plt.savefig(image_file_path)

    plt.show()
