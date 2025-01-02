import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.data.data_spliter import split_data


def load_neural():
    X_train, X_test, y_train, y_test = split_data()

    # Создание и обучение модели
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Определение callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('../best_model', monitor='val_loss', save_best_only=True, save_weights_only=False, save_format='tf')

    # # Обучение модели с использованием callbacks
    # history = model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
    #
    # # Загрузка лучшей модели
    # model = tf.keras.models.load_model('best_model')

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
    results.to_csv('results/neural/neural.csv', index=False)

    with open('results/neural/neural_error.txt', 'w', encoding='utf-8') as file:
        file.write('mae = ' + str(mae))


    # Вывод весов и сохранение модели
    # weights = model.get_weights()
    # print("Веса модели:", weights)
    #
    # model.save('neural_network_model/neural_network_model')
    #
    # # Визуализация нейронной сети
    # plot_model(model, to_file='neural_network_model/neural_network_model.png', show_shapes=True, show_layer_names=True)
    #
    # # Получение весов модели
    # weights = model.get_weights()
    #
    # # Визуализация весов первого слоя
    # weights_layer1 = weights[0]
    # biases_layer1 = weights[1]
    #
    # plt.figure(figsize=(10, 5))
    # plt.imshow(weights_layer1, aspect='auto', cmap='viridis')
    # plt.title('Веса первого слоя')
    # plt.colorbar()
    # plt.show()
    #
    # # Визуализация весов второго слоя
    # weights_layer2 = weights[2]
    # biases_layer2 = weights[3]
    #
    # plt.figure(figsize=(10, 5))
    # plt.imshow(weights_layer2, aspect='auto', cmap='viridis')
    # plt.title('Веса второго слоя')
    # plt.colorbar()
    # plt.show()
    #
    # # Визуализация весов выходного слоя
    # weights_output = weights[4]
    # biases_output = weights[5]
    #
    # plt.figure(figsize=(10, 5))
    # plt.imshow(weights_output, aspect='auto', cmap='viridis')
    # plt.title('Веса выходного слоя')
    # plt.colorbar()
    # plt.show()
