import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Чтение данных из Excel файла
file_path = 'res/results.xlsx'
data = pd.read_excel(file_path)


# Шаг 2: Подготовка данных
# Предположим, что первый столбец - номер учащегося, второй столбец - средний балл
# Столбцы 3-107 - ответы на опрос, столбцы 108-110 - названия любимых предметов

# Удалим номер учащегося и названия любимых предметов, так как они не нужны для обучения
data = data.fillna(0) # Заполняем пустые ячейки нулями
data = data.replace(' ', 0) # Меняем пробелы в ячейках на нули
data = data.drop(data.columns[[32, 83]], axis=1) # Удаляем данные не пятибального диапазона
data = data.drop(data.columns[-3:], axis=1) # Удаляем нечисловые данные
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Нормализуем данные
X = X / 5.0  # Предполагаем, что ответы на опрос в диапазоне от 1 до 5
y = y / 5.0  # Предполагаем, что средний балл в диапазоне от 0 до 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=500, batch_size=10)


# Тестирование
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка (MSE) на тестовом наборе: {mse}")

# Создание таблицы с тестовыми и предсказанными данными
results = pd.DataFrame({
    'Тестовые данные': y_test * 5,
    'Предсказанные данные': y_pred.flatten() * 5
})

print(results)

# Вывод весов и сохранение модели
weights = model.get_weights()
print("Веса модели:", weights)

model.save('neural_network_model.keras')

# Визуализация нейронной сети
plot_model(model, to_file='neural_network_model.png', show_shapes=True, show_layer_names=True)

# Получение весов модели
weights = model.get_weights()

# Визуализация весов первого слоя
weights_layer1 = weights[0]
biases_layer1 = weights[1]

plt.figure(figsize=(10, 5))
plt.imshow(weights_layer1, aspect='auto', cmap='viridis')
plt.title('Веса первого слоя')
plt.colorbar()
plt.show()

# Визуализация весов второго слоя
weights_layer2 = weights[2]
biases_layer2 = weights[3]

plt.figure(figsize=(10, 5))
plt.imshow(weights_layer2, aspect='auto', cmap='viridis')
plt.title('Веса второго слоя')
plt.colorbar()
plt.show()

# Визуализация весов выходного слоя
weights_output = weights[4]
biases_output = weights[5]

plt.figure(figsize=(10, 5))
plt.imshow(weights_output, aspect='auto', cmap='viridis')
plt.title('Веса выходного слоя')
plt.colorbar()
plt.show()