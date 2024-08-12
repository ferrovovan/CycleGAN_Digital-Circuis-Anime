import torch


# Подготовка входного изображения
from PIL import Image
from Paths import dataset_A_path
input_image_path = os.path.join(dataset_A_path, 'pomni_input.jpg')
input_image = Image.open(input_image_path).convert('RGB')

# Применение преобразований к входному изображению
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((510, 680)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
input_tensor = transform(input_image).unsqueeze(0)


# Создание генератора и загрузка его обученных весов
from Generator import Generator
gen_A2B = Generator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Paths import models_save_path
model_weigths_path = os.path.join(models_save_path, 'gen_A2B.pth')
gen_A2B.load_state_dict(  torch.load(model_weigths_path, map_location=device)  )
gen_A2B.eval()


# Применение генератора
with torch.no_grad():
	generated_image = gen_A2B(input_tensor)


	# Преобразование тензора обратно в изображение
	generated_image = generated_image.squeeze().cpu()
	generated_image = transforms.ToPILImage()(generated_image)


# Отображение оригинального и сгенерированного изображений
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(generated_image)
plt.title('Generated Image')
plt.axis('off')

plt.show()

