import pandas as pd

# Чтение данных из Excel файла
file_path = '../../resources/students/6-2_28.xlsx'
data = pd.read_excel(file_path)

# !!! Проблема с индексами. Исправить !!!

# Проверить правильность телефонов и подобных переменных: 0 - нет, 1 - фонарик, 2 - смартфон

n = -1  # Смещение индекса строк
c = 1  # Смещение начала индекса столбцов
# Данные в: строки с индексом от 1, столбцы с индексом 3
data = data.fillna(0)

data.drop(data.columns[2], axis=1, inplace=True)
data.drop(data.columns[0], axis=1, inplace=True)

data.iloc[n+1:n+3, c:] /= 5
data.iloc[n+3:n+5, c:] = data.iloc[n+3:n+5, c:].replace(2,0)
data.iloc[n+5:n+13, c:] /= 5
data.iloc[n+13, c:] = data.iloc[n+13, c:].replace(2,0)
data.iloc[n+14:n+17, c:] /= 5
data.iloc[n+17:n+19, c:] = data.iloc[n+17:n+19, c:].replace(2,0)
data.iloc[n+19:n+21, c:] /= 5
data.iloc[n+21, c:] = data.iloc[n+21, c:].replace(2,0)
data.iloc[n+22:n+24, c:] /= 5
data.iloc[n+24, c:] = data.iloc[n+24, c:].replace(2,0)
data.iloc[n+25:n+28, c:] /= 5
data.iloc[n+28, c:] = data.iloc[n+28, c:].replace(2,0)
print(data.iloc[n+29, :c])
print(data.iloc[n+31, :c])
data.iloc[n+29:n+31, c:] /= 5
data.iloc[n+31, c:] = data.iloc[n+31, c:].replace('м',1)
data.iloc[n+31, c:] = data.iloc[n+31, c:].replace('ж',0)
data.iloc[n+32:n+34, c:] /= 5
data.iloc[n+34, c:] = data.iloc[n+34, c:].replace(2,0)
data.iloc[n+35:n+37, c:] /= 5
data.iloc[n+37:n+40, c:] = data.iloc[n+37:n+40, c:].replace(2,0)
data.iloc[n+40:n+45, c:] /= 5
data.iloc[n+45, c:] /= 11 # класс
data.iloc[n+46:n+65, c:] /= 5
data.iloc[n+65, c:] = data.iloc[n+65, c:].replace(2,0)
data.iloc[n+66:n+80, c:] /= 5
data.iloc[n+80, c:] /= 2
print(data.iloc[n+81, :c])
data.iloc[n+81, c:] /= 120  # время до школы
data.iloc[n+82:n+83, c:] /= 2
data.iloc[n+83:n+105, c:] = data.iloc[n+83:n+105, c:].replace(2,0)
# data.iloc[n+106:n+109, 3:] /= 3 любпредметы
data.drop(data.index[n+105:n+108], inplace=True) # временно удаляем данные о любимых предметах


# # Проверка
# # Подсчет значений больше 1
# count_greater_than_1 = (data <= 1).sum().sum()
#
# # Подсчет значений меньше 0
# count_less_than_0 = (data >= 0).sum().sum()
#
# dsize = data.size
#
# if dsize != count_greater_than_1 or dsize != count_less_than_0:
#     raise ValueError(f"Ячеек больше 1 = {dsize - count_greater_than_1}, ячеек меньше 0 = {dsize - count_less_than_0}")
#
# # Вывод результатов
# print(f"Количество значений больше 1: {count_greater_than_1}")
# print(f"Количество значений меньше 0: {count_less_than_0}")
# print(f"Всего значений: {dsize}")
#
normalized_file_path = '../../resources/students/normalized.xlsx'
data.to_excel(normalized_file_path, index=False)

print(f"Нормализованные данные сохранены в файл: {normalized_file_path}")
