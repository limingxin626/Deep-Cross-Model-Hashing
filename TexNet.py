import torch.nn as nn
import torch.utils.data as Data
import numpy as np

class TexData(Data.Dataset):
	def __init__(self, file_path, transform=None, target_transform=None):
		super(TexData, self).__init__()
		f_tex = open(file_path, 'r')
		self.tex = []
		for line in f_tex:
			line = line.strip()
			line = line.split()
			line = np.asarray(line)
			self.tex.append(line.astype(np.float32))
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, item):
		return item,self.tex[item]

	def __len__(self):
		return len(self.tex)

class TexNet(nn.Module):
	def __init__(self):
		super(TexNet, self).__init__()
		self.dense = nn.Sequential(
			nn.Linear(1000, 2000),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(2000, 16)
		)

	def forward(self, x):
		return self.dense(x)