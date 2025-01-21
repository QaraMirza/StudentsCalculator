import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.data.data_spliter import split_data


def load_polynomial(file_path, file_type):
    X_train, X_test, y_train, y_test = split_data(file_path, file_type)

    # Преобразование данных в полиномиальные признаки
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_train)

    # Создание и обучение модели
    model = LinearRegression()
    model.fit(X_poly, y_train)


    X_test_poly = poly.fit_transform(X_test)

    # Предсказание
    y_pred = model.predict(X_test_poly)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

    results = pd.DataFrame({
        'Тестовые данные': y_test,
        'Предсказанные данные': y_pred.flatten(),
    })

    results_dir = 'results/' + file_type + '/polynomial'

    print(results)
    results.to_csv(results_dir + '/polynomial.csv', index=False)

    with open(results_dir + '/polynomial_error.txt', 'w', encoding='utf-8') as file:
        file.write('mae = ' + str(mae))



    # # Визуализация результатов
    # # Для визуализации используем только первый признак
    # X_vis = X[:, 0]  # Используем только первый признак для визуализации
    # plt.scatter(X_vis, y, color='blue', label='Фактические данные')
    # plt.scatter(X_vis, y_pred, color='red', label='Предсказанные данные')
    # plt.xlabel('Первый признак')
    # plt.ylabel('Средний балл')
    # plt.legend()
    # plt.show()
