import torch.utils.data as Data
import numpy as np
import cv2
import torch


def opencvLoad(path, h, w):
	image = cv2.imread(path)
	image = cv2.resize(image, (h, w))
	image = image.astype(np.float32)
	image = np.transpose(image, (2, 1, 0))
	image = torch.from_numpy(image)
	return image


def loadS(path):
	f = open(path, 'r')
	l = 0
	line = f.readline().strip().split()
	s = np.zeros((len(line), len(line)))
	s[l] = line
	l += 1
	for line in f:
		line = line.strip().split()
		line = np.asarray(line)
		s[l] = line.astype(np.int32)
	return s

def loadImgLabel(path):
	f=open(path,'r')
	ans=[]
	for line in f:
		line=line.strip()
		line=line.split()
		line=np.asarray(line)
		ans.append(line.astype(np.int))
	return ans

def loadTexLabel(path):
	f = open(path, 'r')
	ans = []
	for line in f:
		line = line.strip()
		line = line.split()
		line = np.asarray(line)
		ans.append(line.astype(np.int))
	return ans

class LabelData(Data.Dataset):
	def __init__(self, file_path, transform=None, target_transform=None):
		super(LabelData, self).__init__()
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
		return self.tex[item]

	def __len__(self):
		return len(self.tex)


class myData(Data.Dataset):
	def __init__(self, img_path, tex_path, label_path, transform=None, target_transform=None):
		super(myData, self).__init__()
		f_img = open(img_path, 'r')
		f_tex = open(tex_path, 'r')
		# f_s = open(label_path,'r')
		f_label = open(label_path, 'r')
		self.imgs = []
		self.texs = []
		labels = []
		index = 0
		i = 0
		for line in f_label:
			line = line.strip()
			labels.append(int(line))
		for url in f_img:
			index += 1
			if (index == labels[i]):
				i += 1
				url = url.strip()
				self.imgs.append(url)
		index = 0
		i = 0
		for line in f_tex:
			index += 1
			if (index == labels[i]):
				i+=1
				line = line.strip()
				line = line.split()
				line = np.asarray(line)
				self.texs.append(line.astype(np.float32))
		# for line in f_label:
		# 	line = line.strip()
		# 	line = line.split()
		# 	line = np.asarray(line)
		# 	self.labels.append(line.astype(np.float32))
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		url = self.imgs[index]
		img = opencvLoad('train/' + url, 227, 227)
		return index, img, self.texs[index]

	def __len__(self):
		return len(self.texs)
