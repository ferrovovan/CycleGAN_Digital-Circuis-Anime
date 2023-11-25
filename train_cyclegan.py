import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools
import os
import time

from models import Generator, Discriminator
from utils import CycleGANTrainer, ImageDataset
from globals import *

# Параметры обучения
batch_size = 1
num_epochs = 200

# Загрузка датасета
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset_A = ImageDataset("training_data/input" , transform=transform)
dataset_B = ImageDataset("training_data/output", transform=transform)

dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

# Создание генераторов и дискриминаторов
gen_A2B = Generator()
gen_B2A = Generator()
disc_A = Discriminator()
disc_B = Discriminator()
trainied_models = [gen_A2B, gen_B2A, disc_A, disc_B]

# Путь к сохраненным весам
path_gen_A2B = 'saved_models/gen_A2B.pth'
path_gen_B2A = 'saved_models/gen_B2A.pth'
path_disc_A  = 'saved_models/disc_A.pth'
path_disc_B  = 'saved_models/disc_B.pth'
trainied_models_paths = [path_gen_A2B, path_gen_B2A, path_disc_A, path_disc_B]

# Загрузка весов
for i in range(len(trainied_models)):
	path = trainied_models_paths[i]
	model = trainied_models[i]
	if os.path.exists(path):
		print(f"Импортировали весы для {model}.")
		if not torch.cuda.is_available():
			model.load_state_dict(
				torch.load(path, map_location=torch.device('cpu'))
			)
		else:
			model.load_state_dict(
				torch.load(path)
			)
	else:
		print(f"Не найдены весы для {model}. Начинаем обучение с нуля.")

# перенос на железо
for model in trainied_models:
	model.to(device)

# Определение функции потерь
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

# Создание оптимизаторов
optimizer_G = torch.optim.Adam(itertools.chain(gen_A2B.parameters(), gen_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(disc_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(disc_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Создание трейнера CycleGAN
trainer = CycleGANTrainer(gen_A2B, gen_B2A, disc_A, disc_B, optimizer_G, optimizer_D_A, optimizer_D_B, criterion_GAN, criterion_cycle)

# Обучение
print("start of train")
for epoch in range(num_epochs):
    start = time.time()
    for batch in zip(dataloader_A, dataloader_B):
        real_A = batch[0].to(device)
        real_B = batch[1].to(device)

        trainer.train_step(real_A, real_B)
    print(f"epoch {epoch+1}/{num_epochs}")
    print(f"time lost = {time.time() - start} seconds")

    # Вывод промежуточных результатов
    if epoch % 10 == 0:
        trainer.save_models()
#        trainer.visualize_results(real_A, real_B)

