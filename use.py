import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models import *
from utils import *

# Загрузка обученных весов генератора
gen_A2B = Generator()
if not torch.cuda.is_available():
	gen_A2B.load_state_dict(torch.load('saved_models/gen_A2B.pth', map_location=torch.device('cpu')))
else:
	gen_A2B.load_state_dict(torch.load('saved_models/gen_A2B.pth'))
gen_A2B.eval()

# Подготовка входного изображения
input_image_path = 'training_data/input/image2_input.jpg'
input_image = Image.open(input_image_path).convert('RGB')

# Применение преобразований к входному изображению
transform = transforms.Compose([
    transforms.Resize((510, 680)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

input_tensor = transform(input_image).unsqueeze(0)

# Преобразование с помощью генератора
with torch.no_grad():
    generated_image = gen_A2B(input_tensor)

# Преобразование тензора обратно в изображение
generated_image = generated_image.squeeze().cpu()
generated_image = transforms.ToPILImage()(generated_image)

# Отображение оригинального и сгенерированного изображений
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

