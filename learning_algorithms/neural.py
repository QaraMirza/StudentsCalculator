import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.data.data_spliter import split_data


def load_neural(file_path, file_type):
    X_train, X_test, y_train, y_test = split_data(file_path, file_type)

    # Создание и обучение модели
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2)

    # Тестирование
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    # Создание таблицы с тестовыми и предсказанными данными
    results = pd.DataFrame({
        'Тестовые данные': y_test,
        'Предсказанные данные': y_pred.flatten()
    })

    print(results)

    results_dir = 'results/' + file_type + '/neural'
    results.to_csv(results_dir + '/neural.csv', index=False)

    with open(results_dir + '/neural_error.txt', 'w', encoding='utf-8') as file:
        file.write('mae = ' + str(mae))