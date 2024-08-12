# Параметры обучения
batch_size = 1
total_epochs = 200

optimizers_lr = 0.0002
optimizers_betas = (0.5, 0.999)

save_ratio = 10
visualize_ratio = 5




# Загрузка датасета
from torchvision import transforms
transform = transforms.Compose([
	transforms.Resize((512, 512)),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

from ImageDataset import ImageDataset
from Paths import dataset_A_path, dataset_B_path

dataset_A = ImageDataset(dataset_A_path, transform=transform)
dataset_B = ImageDataset(dataset_B_path, transform=transform)

from torch.utils.data import DataLoader
dataloader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
dataloader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)


# Создание генераторов и дискриминаторов
from Generator import Generator
gen_A2B = Generator()
gen_B2A = Generator()
from Discriminator import Discriminator
disc_A = Discriminator()
disc_B = Discriminator()

models_list = [gen_A2B, gen_B2A, disc_A, disc_B]


# Путь к сохраненным весам
from Paths import models_save_path
path_gen_A2B = models_save_path + 'gen_A2B.pth'
path_gen_B2A = models_save_path + 'gen_B2A.pth'
path_disc_A  = models_save_path + 'disc_A.pth'
path_disc_B  = models_save_path + 'disc_B.pth'
models_paths_list = [path_gen_A2B, path_gen_B2A, path_disc_A, path_disc_B]


# Загрузка весов
from os.path import exists
from torch import device as torch_device, load as torch_load
device = torch_device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(4):
	path = models_paths_list[i]
	model = models_list[i]

	if os.path.exists(path):
		print(f"Импортировали весы для {model}.")
		model.load_state_dict(torch_load(path, map_location=device))
	else:
		print(f"Не найдены весы для {model}. Начинаем обучение с нуля.")

	model.to(device) # Переносим модель на выбранное устройство (CPU или GPU)



# Определение функций потерь
from torch.nn import MSELoss, L1Loss
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

# Создание оптимизаторов
from torch.optim import Adam
optimizer_G = torch.optim.Adam(itertools.chain(gen_A2B.parameters(), gen_B2A.parameters()), lr=optimizers_lr, betas=optimizers_betas)
optimizer_D_A = torch.optim.Adam(disc_A.parameters(), lr=optimizers_lr, betas=optimizers_betas)
optimizer_D_B = torch.optim.Adam(disc_B.parameters(), lr=optimizers_lr, betas=optimizers_betas)


# Создание трейнера CycleGAN
from CycleGANTrainer import CycleGANTrainer
#from Paths import models_save_path
trainer = CycleGANTrainer(gen_A2B, gen_B2A, disc_A, disc_B,
	optimizer_G, optimizer_D_A, optimizer_D_B, criterion_GAN, criterion_cycle,
	models_save_path
)


# Обучение
from itertools import zip
from time import time as get_time

print("start of train")
for epoch in range(1, total_epochs + 1):
	start = get_time()
	for batch in zip(dataloader_A, dataloader_B):
		real_A = batch[0].to(device)
		real_B = batch[1].to(device)

		trainer.train_step(real_A, real_B)
	print(f"epoch {epoch}/{total_epochs}")
	print(f"time lost = {get_time() - start} seconds")

	# Вывод промежуточных результатов
	if epoch % save_ratio == 0:
		trainer.save_models()

	if epoch % visualize_ratio == 0:
		trainer.visualize_results(real_A, real_B)

