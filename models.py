import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import ViTForImageClassification

#from DenseNet import DenseNet 

class Flatten(nn.Module):

   def forward(self, input):

	   return input.view(input.size(0), -1)


class DenseCNNModel(nn.Module):

	def __init__(self, 
		in_channel=32, growth_rate=16, bottleneck_size=4, block_config=(6,12,8), 
		dropout=0, transition_compression=0.5, num_classes=4
	):
		raise NotImplementedError		
	# 	super().__init__()

	# 	self.DenseNet = DenseNet(
	# 		in_channel, growth_rate, bottleneck_size, block_config, dropout, transition_compression, num_classes
	# 	)

	# def forward(self, X):
	# 	X_out = self.Dense(X)
	# 	return X_out


class LogisticRegression (nn.Module):
	""" Simple logistic regression model """

	def __init__(self, n_classes):
		super().__init__() 

		self.fc = nn.Sequential(
			Flatten(), 
			nn.Linear(3 * 224 * 224, n_classes)
		)

	def forward(self, X):
		X_out = self.fc(X)
		return X_out


class BasicCNNModel (nn.Module):
	""" Simple 2D-CNN model """
	def __init__(self, n_classes, filter_sizes = [3, 16, 64], dropout = 0.8):
		
		super().__init__()

		self.conv1 = nn.Sequential( # input 3 by 224 by 224
			nn.BatchNorm2d(filter_sizes[0]), 
			nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size = 3, padding = 1), # output 16 by 224 by 224 
			nn.ReLU(inplace=True), 
			nn.MaxPool2d(2) # output filter_sizes[1] by 112 by 112 
		)
		self.conv2 = nn.Sequential( # input 3 by 224 by 224
			nn.BatchNorm2d(filter_sizes[1]), 
			nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size = 3, padding = 1), # output 16 by 224 by 224 
			nn.ReLU(inplace=True), 
			nn.MaxPool2d(2) # output filter_sizes[2] by 56 by 56 
		)
		self.fc = nn.Sequential(
			Flatten(), 
			nn.Dropout(dropout), 
			nn.Linear(filter_sizes[2] * 56 * 56, n_classes)
		)

	def forward(self, X):
		X_conv1 = self.conv1(X)
		X_conv2 = self.conv2(X_conv1)
		X_out = self.fc(X_conv2)
		return X_out


class MediumCNNModel (nn.Module):
	""" Simple 2D-CNN model """
	def __init__(self, n_classes, filter_sizes = [3, 16, 64, 128], dropout = 0.8):
		
		super().__init__()

		self.conv1 = nn.Sequential( # input 3 by 224 by 224
			nn.BatchNorm2d(filter_sizes[0]), 
			nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size = 3, padding = 1), # output 16 by 224 by 224 
			nn.ReLU(inplace=True), 
			nn.MaxPool2d(2) # output filter_sizes[1] by 112 by 112 
		)
		self.conv2 = nn.Sequential( # input 3 by 224 by 224
			nn.BatchNorm2d(filter_sizes[1]), 
			nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size = 3, padding = 1), # output 16 by 224 by 224 
			nn.ReLU(inplace=True), 
			nn.MaxPool2d(2) # output filter_sizes[2] by 56 by 56 
		)
		self.conv3 = nn.Sequential(
			nn.BatchNorm2d(filter_sizes[2]), 
			nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_size = 3, padding = 1), 
			nn.ReLU(inplace=True), 
			nn.MaxPool2d(2)
		)

		self.fc = nn.Sequential(
			Flatten(), 
			nn.Dropout(dropout), 
			nn.Linear(filter_sizes[3] * 28 * 28, n_classes)
		)

	def forward(self, X):
		X_conv1 = self.conv1(X)
		X_conv2 = self.conv2(X_conv1)
		X_conv3 = self.conv2(X_conv2)
		X_out = self.fc(X_conv2)
		return X_out

