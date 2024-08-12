from PIL import Image
import os


class TransformFunction:
	def __init__(self, name, operation):
		self.name = name
		self.operation = operation

	# Сложная функция - с параметрами
	#def __call__(self, input_path, output_path, *args, **kwargs):
	#	img = Image.open(input_path)
	#	transformed_img = eval(f'img.{self.operation}', {"img": img, "args": args, "kwargs": kwargs, "Image": Image})
	#	transformed_img.save(output_path)

	# Простая реализация
	def __call__(self, input_path, output_path):
		img = Image.open(input_path)
		transformed_img = eval(f'img.{self.operation}')
		transformed_img.save(output_path)

	def __str__(self):
		return f"TransformFunction(name='{self.name}', operation='{self.operation}')"



# Пример сложной функции
# rotate_image = TransformFunction("rotate_image", "rotate(args[0])")
# resize_image = TransformFunction("resize_image", "resize(args[0])") # args[0] - new_size: (int, int)

flip_vertical = TransformFunction("flip_vertical", "transpose(Image.FLIP_TOP_BOTTOM)")
flip_horizontal = TransformFunction("flip_horizontal", "transpose(Image.FLIP_LEFT_RIGHT)")
rotate_image_45 = TransformFunction("rotate_image_45", "rotate(45)")
grayscale_image = TransformFunction("grayscale_image", "convert('L')")


if __name__ == '__main__':
	from Paths import training_data_path

	# Путь к основной папке с исходными и выходными данными	
	base_folder = training_data_path
	# Папки для оригинальных входных и выходных изображений
	input_images_folder  = os.path.join(base_folder, "input_images")
	output_images_folder = os.path.join(base_folder, "output_images")


	# Папки для аугментированных изображений
	augmented_input_folder  = os.path.join(output_images_folder, "augmented_inputs")
	augmented_output_folder = os.path.join(output_images_folder, "augmented_outputs")
	for folder in [augmented_input_folder, augmented_output_folder]:
		if not os.path.exists(folder):
			os.makedirs(folder)
	
	transformations = [flip_vertical, flip_horizontal, rotate_image_45]

	# Для каждой пары изображений...
	for filename_input, filename_output in zip(
			os.listdir(input_images_folder), os.listdir(output_images_folder)
		):
		print(f"Обработка {filename_input}")
		input_path  = os.path.join(input_images_folder , filename_input)
		output_path = os.path.join(output_images_folder, filename_output)

		# Применяем все возможные комбинации трансформаций
		for func in transformations:
			augmented_input_name  = f"{func.name}_{filename_input}"
			func(input_path,
				os.path.join(augmented_input_folder , augmented_input_name)
			)

			augmented_output_name = f"{func.name}_{filename_output}"
			func(output_path,
				os.path.join(augmented_output_folder, augmented_output_name)
			)

	print(f"Аугментация завершена. Аугментированные изображения сохранены в {output_folder}")

