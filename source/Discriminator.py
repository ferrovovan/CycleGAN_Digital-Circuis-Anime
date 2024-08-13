import torch.nn as nn


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		# Пример простой архитектуры дискриминатора
		self.model = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.InstanceNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
			nn.InstanceNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),

			# Пример других слоев...

			nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
		)

	def forward(self, x):
		return self.model(x)

