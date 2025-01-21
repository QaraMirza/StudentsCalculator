from learning_algorithms.linear import load_linear
from learning_algorithms.polynomial import load_polynomial
from learning_algorithms.neural import load_neural

file_path = "resources/students/normalized.xlsx"
file_type = "students"  # Перевести в Enum

# load_linear(file_path, file_type)
# load_polynomial(file_path, file_type)
load_neural(file_path, file_type)
