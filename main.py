import torch
import torchvision.transforms as transforms
import ImgNet
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import TexNet
import Label

device = torch.device('cuda:0')

Epoch = 50
Batch_size = 10
LR = 0.001
gamma = 0.5
eta = 0.5
train_img_label = Label.loadImgLabel('lite/Lite_GT_Train.txt')
train_tex_label = Label.loadTexLabel('lite/Lite_Tags81_Train.txt')


# test_img_label = Label.loadImgLabel('lite/Lite_GT_Train.txt')
# test_tex_label = Label.loadTexLabel('lite/Lite_Tags81_Train.txt')


def cal_sim(F, G):
	sum = torch.zeros(1)
	for i in range(train_img_label.__len__()):
		for t in range(train_tex_label.__len__()):
			if (np.dot(train_img_label[i], train_tex_label[t])):
				ans = torch.dot(F[i], G[t])
				sum[0] += ans
	# sum+=ans
	return sum[0]


class cal_Loss(nn.Module):
	def __init__(self, gamma, eta):
		super(cal_Loss, self).__init__()
		# self.S = S
		self.gamma = gamma
		self.eta = eta
		return

	def forward(self, F, G, B):
		# theta = np.matmul(np.transpose(F), G) / 2
		term1_1 = torch.sum(torch.log(1 + torch.exp(torch.mm(F, torch.t(G)) / 2)))
		term1_2 = cal_sim(F, G)
		# term1 = np.sum(np.log(1 + np.exp(theta)) - np.multiply(self.S, theta))
		term1 = term1_1 - term1_2
		term2 = torch.sum(torch.pow((B - F), 2) + torch.pow(B - G, 2))
		term3 = torch.sum(torch.pow(torch.matmul(F, torch.ones((F.shape[1], 1))), 2)) + torch.sum(
			torch.pow(torch.mm(G, torch.ones((F.shape[1], 1))), 2))
		loss = term1 + self.gamma * term2 + self.eta * term3
		print('loss1:', term1)
		print('loss2:', term2)
		print('loss3:', term3)
		# loss = term1
		return loss


def train_imgNet(img_loader, imgNet, F):
	for index, x in img_loader:
		F[index] = imgNet(x.to(device) / 128 - 1)


def train_texNet(tex_loader, texNet, G):
	for index, x in tex_loader:
		G[index] = texNet(x.to(device))


def T_to_I(B_img, B_tex, len, label_img, label_tex):
	tt = 0
	tf = 0
	ft = 0
	ff = 0
	for x in range(len):
		flag_p = False
		flag_g = False
		dis = torch.zeros(len)
		for i in range(len):
			dis[i] = torch.sum(torch.abs(torch.add(B_img[x], -1, B_tex[i])))
		sorted, index = torch.sort(dis)
		if (sorted[0].data > 2):
			flag_p = False
			for i in range(label_tex.__len__()):
				if (np.dot(label_img[x], label_tex[i])):
					flag_g = True
					tf += 1
					break
			if (not flag_g):
				ff += 1
		else:
			flag_p = True
			for i in index[0:10]:
				if (np.dot(label_img[x], label_tex[i])):
					flag_g = True
					tt += 1
					break
			if (not flag_g):
				ft += 1
	print('tt:', tt, 'tf:', tf, 'ft:', ft, 'ff:', ff)


train_img = ImgNet.ImgData(list_path='lite/Train_imageOutPutFileList.txt', transform=transforms.ToTensor())
train_tex = TexNet.TexData(file_path='lite/Lite_Tags1k_Train/txt', transform=transforms.ToTensor())
train_img_loader = Data.DataLoader(dataset=train_img, batch_size=Batch_size, shuffle=True)
train_tex_loader = Data.DataLoader(dataset=train_tex, batch_size=Batch_size, shuffle=True)
len_train = train_img.__len__()
B = torch.zeros((len_train, 16)).to(device)
F = torch.zeros((len_train, 16)).to(device)
G = torch.zeros((len_train, 16)).to(device)

# test_img = ImgNet.ImgData(list_path='lite/Train_imageOutPutFileList.txt', transform=transforms.ToTensor())
# test_tex = TexNet.TexData(file_path='lite/Lite_Tags1k_Train/txt', transform=transforms.ToTensor())
# test_img_loader = Data.DataLoader(dataset=test_img, batch_size=Batch_size, shuffle=True)
# test_tex_loader = Data.DataLoader(dataset=test_tex, batch_size=Batch_size, shuffle=True)
# len_test = test_img.__len__()
# F_test = torch.zeros((len_test, 16)).to(device)
# G_test = torch.zeros((len_test, 16)).to(device)

imgNet = ImgNet.ImgNet().to(device)
texNet = TexNet.TexNet().to(device)

loss_func = cal_Loss(gamma, eta).to(device)
optimizer1 = torch.optim.Adam(imgNet.parameters(), lr=LR)
optimizer2 = torch.optim.Adam(texNet.parameters(), lr=LR)

for epoch in range(Epoch):
	# print('epoch{}'.format(epoch))
	train_imgNet(train_img_loader, imgNet, F)
	train_texNet(train_tex_loader, texNet, G)
	B = torch.sign(gamma * (F + G))
	optimizer1.zero_grad()
	optimizer2.zero_grad()
	loss = loss_func(F, G, B)
	loss.backward(retain_graph=True)
	optimizer1.step()
	optimizer2.step()
	print('loss:', loss, "\n")

	if (epoch % 5 == 4):
		# train_imgNet(test_img_loader, imgNet, F_test)
		# train_texNet(test_tex_loader, texNet, G_test)
		# T_to_I(torch.sign(F_test), torch.sign(G_test), len_test)
		T_to_I(torch.sign(F), torch.sign(G), len_train, train_img_label, train_tex_label)

torch.save(imgNet, 'imgNet.pkl')
torch.save(texNet, 'texNet.pkl')
