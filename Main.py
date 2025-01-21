from learning_algorithms.linear import load_linear
from learning_algorithms.polynomial import load_polynomial
from learning_algorithms.neyral import load_neural

file_path = "resources/students/normalized.xlsx"
file_type = "students"  # Перевести в Enum
load_linear(file_path, file_type)
# load_polynomial()
# load_neural()
