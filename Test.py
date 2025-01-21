from utils.data.data_spliter import get_column_names_students

file_path = "resources/students/normalized.xlsx"

column_names = get_column_names_students(file_path)

print(column_names)