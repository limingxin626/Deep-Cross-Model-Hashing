import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import cv2


def opencvLoad(path, h, w):
	image = cv2.imread(path)
	image = cv2.resize(image, (h, w))
	image = image.astype(np.float32)
	image = np.transpose(image, (2, 1, 0))
	image = torch.from_numpy(image)
	return image


class ImgData(Data.Dataset):
	def __init__(self,list_path, transform=None, target_transform=None):
		super(ImgData, self).__init__()
		f_url = open(list_path, 'r')
		self.imgs = []
		for url in f_url:
			url = url.strip()
			self.imgs.append(url)
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, index):
		url = self.imgs[index]
		img = opencvLoad('NUSWIDE/'+url, 227, 227)
		return index,img


class ImgNet(nn.Module):
	def __init__(self):
		super(ImgNet, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 96, 11, 4, 0),
			nn.ReLU(),
			nn.MaxPool2d(3, 2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(96, 256, 5, 1, 2),
			nn.ReLU(),
			nn.MaxPool2d(3, 2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(256, 384, 3, 1, 1),
			nn.ReLU()
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(384, 384, 3, 1, 1),
			nn.ReLU()
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(384, 256, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(3, 2)
		)
		self.dense = nn.Sequential(
			nn.Linear(9216, 4096),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(4096, 16)
		)

	def forward(self, x):
		conv1_out = self.conv1(x)
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)
		conv4_out = self.conv4(conv3_out)
		conv5_out = self.conv5(conv4_out)
		res = conv5_out.view(conv5_out.size(0), -1)
		out = self.dense(res)
		return out
