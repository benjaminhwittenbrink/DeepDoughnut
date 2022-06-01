import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import ViTForImageClassification


class DenseCNNModel(nn.Module):

	def __init__(self):
		raise NotImplementedError


class LogisticRegression (nn.Module):
    """ Simple logistic regression model """

	def __init__(self):
		raise NotImplementedError

class BasicCNNModel (nn.Module):
	""" Simple 2D-CNN model """
	def __init__(self):
		raise NotImplementedError