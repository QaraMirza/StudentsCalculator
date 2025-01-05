# Загружает и нормализует данные из таблиц.
# Используется перед классами обучения.

# !!! Нужно нормализовать путем приведения к 0-1 !!!
import pandas as pd

# Чтение данных из Excel файла
file_path = '../../resources/ClassTeacher_Answer.xlsx'
data = pd.read_excel(file_path)


data = data.drop(data.columns[[0, 73, 74]], axis=1)  # Удаляем бесполезные столбцы (время и коды)

data = data.replace('Нет', 0)
data = data.replace('Да', 1)

data = data.replace('0 - 5', 0.2)  # Переводим стаж в числа от 0 до 1
data = data.replace('5 - 10', 0.4)
data = data.replace('10 - 15', 0.6)
data = data.replace('более 15', 0.8)

data.iloc[:, 17] = data.iloc[:, 17] / 40  # Нагрузка часов в неделю
data.iloc[:, 24] = data.iloc[:, 24] / 60  # Возраст преподавателя

ranges = [(0, 4), (7, 11), (13, 16), (18, 23), (25, 56), (58, 59), (61, 69)]  # 5-значные в 0-1
for start_index, end_index in ranges:
    data.iloc[:, start_index: end_index+1] = data.iloc[:, start_index: end_index+1] / 5

# Проверка
# Подсчет значений больше 1
count_greater_than_1 = (data <= 1).sum().sum()

# Подсчет значений меньше 0
count_less_than_0 = (data >= 0).sum().sum()

dsize = data.size

if dsize != count_greater_than_1 or dsize != count_less_than_0:
    raise ValueError(f"Ячеек больше 1 = {dsize - count_greater_than_1}, ячеек меньше 0 = {dsize - count_less_than_0}")

# Вывод результатов
print(f"Количество значений больше 1: {count_greater_than_1}")
print(f"Количество значений меньше 0: {count_less_than_0}")
print(f"Всего значений: {dsize}")

normalized_file_path = '../../resources/normalized.xlsx'
data.to_excel(normalized_file_path, index=False)

print(f"Нормализованные данные сохранены в файл: {normalized_file_path}")

