from torch.utils.data import Dataset
from PIL import Image
import os


class ImageDataset(Dataset):
	def __init__(self, root_dir: str = "", transform=None):
		self.image_list = []
		self.root_dir: str = root_dir
		self.transform = transform
		if root_dir != "":
			self.load_images()

	 def load_images(self, key=None, reverse=False):
		if self.root_dir == "":
			print("Can't use \'update_image_list\' method - \'self.root_dir\' attribute not set")
			return

		summary_list = []
		for image_name in os.listdir(self.root_dir):
			if os.path.isfile(
				os.path.join(self.root_dir, image_name)
				):
				summary_list.append(image_name)
		self.image_list = sorted(summary_list, key=key, reverse=reverse)

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		image_path = os.path.join(self.root_dir, self.image_list[idx])
		image = Image.open(image_path).convert('RGB')

		if self.transform:
			image = self.transform(image)

		return image

	def update_images(self, new_root_dir: str):
		# Обновление списка изображений
		self.root_dir = new_root_dir
		self.load_images()

