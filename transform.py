from torchvision import transforms
from PIL import Image
import os

# функции для преобразования
def flip_vertical(input_path, output_path):
	img = Image.open(input_path)
	flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
	flipped_img.save(output_path)

def flip_horizontal(input_path, output_path):
	img = Image.open(input_path)
	flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
	flipped_img.save(output_path)

def rotate_image(input_path, output_path, angle):
	img = Image.open(input_path)
	rotated_img = img.rotate(angle)
	rotated_img.save(output_path)

def rotate_image_45(input_path, output_path):
	rotate_image(input_path, output_path, 45)

def grayscale_image(input_path, output_path):
	img = Image.open(input_path).convert('L')
	img.save(output_path)

def resize_image(input_path, output_path, new_size):
	img = Image.open(input_path)
	resized_img = img.resize(new_size)
	resized_img.save(output_path)


# Путь к папке с исходными изображениями (input и output)
input_output_folder = "/home/vovan/нейронки/Circus/training_data"

# Путь к папке, в которую сохранятся аугментированные изображения
output_folder = "/home/vovan/нейронки/Circus/training_data"

# Создание папок для аугментированных изображений
augmented_input_folder = os.path.join(output_folder , "aug_input" )
augmented_output_folder = os.path.join(output_folder, "aug_output")
for folder in [augmented_input_folder, augmented_output_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Проход по изображениям
for filename_input, filename_output in zip(os.listdir(f"{input_output_folder}/input"),
		os.listdir(f"{input_output_folder}/output")):
	# Полный путь к исходным изображениям
	print(filename_input)
	input_path  = os.path.join(f"{input_output_folder}/input" , filename_input)
	output_path = os.path.join(f"{input_output_folder}/output" , filename_output)

	# Применяем все возможные комбинации трансформаций
	# Первой степени
	for func_name, func in zip(["flip_vertical", "flip_horizontal", "rotate_image_45"],
			[flip_vertical, flip_horizontal, rotate_image_45]):
		# flip_vertical
		input_transf  = f"{func_name}_{filename_input}"	# Генерируем имя для аугментированного изображения
		func(input_path,
			os.path.join(augmented_input_folder, input_transf)
		)
		output_transf = f"{func_name}_{filename_output}"	# Генерируем имя для аугментированного изображения
		func(output_path,
			os.path.join(augmented_output_folder, output_transf)
		)

print("Аугментация завершена. Аугментированные изображения сохранены в", output_folder)

