# standard libraries 
import os 
import pickle 
import argparse 
import random 
import numpy as np 
import collections 
from tqdm import tqdm 
from datetime import datetime

# PyTorch libraries 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler 
from torchvision.datasets import DatasetFolder, ImageFolder 
from torchvision import models, transforms 

# Hugging Face datasets 
#import datasets 

# Transformers libraries 
from transformers import TrainingArguments, Trainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import get_linear_schedule_with_warmup 

# import sklearn models 
from sklearn.naive_bayes import BernoulliNB, MultinomialNB 
# import naive_bayes_utils
#import naive_bayes_utils 

# simple models
from models import LogisticRegression, BasicCNNModel, DenseCNNModel

from sklearn.metrics import confusion_matrix


RANDOM_SEED = 231 
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ImageClassificationCollator:
	def __init__(self, feature_extractor, transforms = False): 
		self.feature_extractor = feature_extractor
		self.transforms = transforms 

	def __call__(self, batch):
		if self.transforms: 
			transformed = [self.feature_extractor(x[0]) for x in batch]
			encodings = {"pixel_values":torch.stack(transformed)}
		else: 
			encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')   
		encodings['labels'] = torch.tensor([x[1] for x in batch],  dtype=torch.long)
		return encodings

# create model and collator
def create_model_and_collator(args, model_name):

	if model_name == "ViT":
		feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
		collator = ImageClassificationCollator(feature_extractor)
		collators = (collator, collator)
		model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=CLASSES)
	
	elif model_name in ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet']:
		# note all models expect image of (3, 224, 224)

		train_data_transforms = transforms.Compose([
			transforms.RandomResizedCrop(224), # i.e. want 224 by 224 
			transforms.RandomHorizontalFlip(),  
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		val_data_transforms = transforms.Compose([
			transforms.Resize(224), # i.e. want 224 by 224 
			transforms.CenterCrop(224), 
			transforms.ToTensor(), 
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
		train_collator = ImageClassificationCollator(train_data_transforms, transforms=True)
		val_collator = ImageClassificationCollator(val_data_transforms, transforms=True)

		collators = (train_collator, val_collator)

		if model_name == 'resnet':
			model = models.resnet18(pretrained=True)
			model.fc = nn.Linear(model.fc.in_features, CLASSES)

		elif model_name == 'alexnet':
			model = models.alexnet(pretrained=True)
			model.classifier[6] = nn.Linear(model.classifier[6].in_features, CLASSES)

		elif model_name == 'vgg':
			model = models.vgg11_bn(pretrained=True)
			model.classifier[6] = nn.Linear(model.classifier[6].in_features, CLASSES)

		elif model_name == 'squeezenet': 
			model = models.squeezenet1_0(pretrained=True)
			model.classifier[1] = nn.Conv2d(512, CLASSES, kernel_size=1, stride=1)
			model.num_classes = CLASSES

		else: 
			# dense net 
			model = models.densenet121(pretrained=True)
			model.classifier = nn.Linear(model.classifier.in_features, CLASSES) 

	elif model_name in ['basic_cnn', 'dense_cnn', 'logistic_regression']:
		# ADD IN transforms though feature extractor might be easier 
		feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
		collator = ImageClassificationCollator(feature_extractor)
		collators = (collator, collator)
		if model_name == "logistic_regression":
			model = LogisticRegression(n_classes=CLASSES)
		elif model_name == "basic_cnn":
			model = BasicCNNModel(n_classes=CLASSES)
		elif model_name == "dense_cnn":
			model = DenseCNNModel(n_classes=CLASSES)

	else: 
		raise NotImplementedError

	print(f'Model name: {model_name}')

	return collators, model 


class RegressionImageFolder(ImageFolder):

# basically need to pass in dict that takes filename --> score 

	def __init__(self, root, image_scores_pkl_path, **kwargs): 
		super().__init__(root, **kwargs)

		with open(image_scores_pkl_path, "rb") as f:
			image_scores = pickle.load(f) 

		paths, _ = zip(*self.imgs)

		self.targets = [image_scores[path] for path in paths]
		self.samples = self.imgs = list(zip(paths, self.targets))


def make_weights_for_balanced_classes(images, n_classes):
	count = [0] * n_classes
	for item in images: 
		count[item[1]] += 1 
	weight_per_class = [0.] * n_classes 
	N = float(sum(count))
	for i in range(n_classes):
		weight_per_class[i] = N / float(count[i])
	weight = [0] * len(images)
	for idx, val in enumerate(images):
		weight[idx] = weight_per_class[val[1]]
	
	return weight, weight_per_class

def create_dataset(args, collator_fns, extensions = ['.jpg.npy'], val_split = 0.15):

	# def npy_loader(path):
	# 	sample = torch.from_numpy(np.load(path))
	# 	return sample 
	# load in dataset frmom directory 
	# dataset = DatasetFolder(
	# 	root = args.data_dir, 
	# 	loader = npy_loader, 
	# 	extensions = extensions
	# )
	# load in dataset from folder 
	class_weights = None 

	if args.regression: 
		dataset = RegressionImageFolder(
			root = args.data_dir, 
			image_scores_pkl_path = args.reg_scores_dict
		)
	else: 
		dataset = ImageFolder(
			root = args.data_dir
		)

		if len(dataset.classes) != args.n_classes:
			raise ValueError("Class argument does not match number of classes found in data directory.")

		weights, class_weights = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
		weights = torch.FloatTensor(weights)
		class_weights = torch.FloatTensor(class_weights)

	# split up into train val data  
	indices = torch.randperm(len(dataset)).tolist()
	n_val = int(np.floor(len(indices) * val_split))
	train_ds = torch.utils.data.Subset(dataset, indices[:-n_val])
	val_ds = torch.utils.data.Subset(dataset, indices[-n_val:])

	if args.regression: 
		train_sampler = RandomSampler()
	else: 
		train_weights = weights[indices[:-n_val]]
		train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

	train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collator_fns[0], shuffle = 1)
	val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator_fns[1], shuffle=0)

	return [train_dl, val_dl], class_weights

def label_statistics(args, dataset_loader):
	if args.regression: 
		return "No class labels for regression problem."
	class_counts = collections.Counter()
	for i, batch in enumerate(dataset_loader):
		inputs, labels = batch['pixel_values'], batch['labels']
		labels = labels.cpu().numpy().flatten()
		class_counts += collections.Counter(labels)
	return class_counts


def measure_accuracy(outputs, labels):
	preds = np.argmax(outputs, axis = 1).flatten()
	labels = labels.flatten()
	correct = np.sum(preds == labels)
	c_matrix = confusion_matrix(labels, preds, labels=CLASS_NAMES)
	return correct, len(labels), c_matrix 

def validation(args, val_loader, model, criterion, device, name = 'Validation', write_file=None):

	model.eval()
	total_loss = 0. 
	if not args.regression: 
		total_correct = 0 
		total_sample = 0 
		total_confusion = np.zeros((CLASSES, CLASSES))

	for i, batch in enumerate(tqdm(val_loader)):
		inputs, labels = batch['pixel_values'], batch['labels'] 
		inputs = inputs.to(device)
		labels = labels.to(device)

		with torch.no_grad():
			if args.model_name in [
			'basic_cnn', 'dense_cnn', 'logistic_regression',
			'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet'
			]:
				outputs = model(inputs)
			else: 
				outputs = model(inputs)['logits'] 

		logits = outputs.to(device)
		loss = criterion(logits, labels)

		logits = outputs 
		total_loss += loss.cpu().item()

		if not args.regression: 
			correct_n, sample_n, c_matrix = measure_accuracy(logits.cpu().numpy(), labels.cpu().numpy())
			total_correct += correct_n 
			total_sample += sample_n 
			total_confusion += c_matrix 

	if args.regression:
		print(f'*** Total mean squared error on the {name} set: {total_loss}')
		if write_file:
			write_file.write(f'*** Total mean squared error on the {name} set: {total_loss}\n')

		# want to pick the best "accuracy" <> equivalent to picking greatest neg loss 
		val_acc = - total_loss
	else: 
		print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
		print(f'*** Confusion matrix:\n{total_confusion}')
		if write_file:
			write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
			write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')

		val_acc = float(total_correct / total_sample) * 100

	return total_loss, val_acc



def train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file=None):
	print("\n>>> Training starts...")

	if write_file: 
		write_file.write("\n>>> Training starts...")

	model.train()

	best_val_acc = 0
	for epoch in range(epoch_n):
		print("*** Epoch:", epoch)
		total_train_loss = 0. 
		total_correct = 0
		total_sample = 0

		for i, batch in enumerate(tqdm(data_loaders[0])): 
			optim.zero_grad()
			inputs, labels = batch['pixel_values'], batch['labels'] 
			inputs = inputs.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			# forward pass 
			if args.model_name in [
				'basic_cnn', 'dense_cnn', 'logistic_regression', 
				'resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet'
			]:
				outputs = model(inputs)
			else: 
				outputs = model(inputs)['logits']

			logits = outputs.to(device)
			loss = criterion(logits, labels)
			
			#correct_n, sample_n, c_matrix = measure_accuracy(logits.cpu().detach().numpy(), labels.cpu().detach().numpy())
			#total_correct += correct_n 
			#total_sample += sample_n 

			total_train_loss += loss.cpu().item()

			# backward pass 
			loss.backward()
			optim.step()

			if scheduler: scheduler.step()

			if i % args.val_every == 0: 
				print(f'*** Loss: {loss}')
				#print(f'*** Running accuracy on the train set: {total_correct/total_sample}')
				if write_file:
					write_file.write(f'\nEpoch: {epoch}, Step: {i}\n')
					write_file.write(f'*** Loss: {loss}\n')
					#write_file.write(f'*** Running accuracy on the train set: {total_correct/total_sample}\n')

				_, val_acc = validation(args, data_loaders[1], model, criterion, device, write_file=write_file)

				model.train()

				if best_val_acc < val_acc: 
					best_val_acc = val_acc 

					if args.save_path:
						if args.model_name in ['ViT']:
							model.save_pretrained(args.save_path)
						else: 
							torch.save(model.state_dict(), args.save_path)






if __name__ == '__main__':

	# set device to GPU if possible
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', default='sample', type=str, help='Image data location.')
	parser.add_argument('--reg_scores_dict', default=None, type=str, help="Path to pickle of dictionary mapping file names to their target scores. Only for regression problem.")

	parser.add_argument('--regression', action='store_true', help='Whether classification or regression task.')
	parser.add_argument('--n_classes', default=4, type=int, help='Number of classes in outcome variable.')	
	parser.add_argument('--batch_size', default=16, type=int, help='Batch size.')
	parser.add_argument('--epoch_n', default=3, type=int, help='Number of epochs for training.')
	parser.add_argument('--val_every', default=100, type=int, help="Number of iterations we should take to perform validation.")
	parser.add_argument('--lr', default=2e-5, type=float, help="Learning rate.")
	parser.add_argument('--eps', default=1e-8, type=float, help='Epsilon value for learning rate.')

	parser.add_argument('--filename', default=None, type=str, help='Name of results file to be saved.')

	parser.add_argument('--model_name', default=None, type=str, help='Name of model.')
	parser.add_argument('--save_path', default=None, type=str, help='The path where the model is going to be saved.')

	parser.add_argument('--n_filters', type=int, default=25, help='Number of filters in the CNN (if applicable)')
	parser.add_argument('--filter_sizes', type=int, nargs='+', action='append', default=[[3,4,5], [5,6,7], [7,9,11]], help='Filter sizes for the CNN (if applicable).')


	args = parser.parse_args()

	# Number of classes 
	REGRESSION = args.regression 
	CLASSES = args.n_classes
	CLASS_NAMES = [i for i in range(CLASSES)]

	if REGRESSION and CLASSES != 1: 
		raise Warning("Arguments indicate regression problem but non unit number of classes. Setting classes equal to 1.")
		CLASSES = 1
		CLASS_NAMES = [i for i in range(CLASSES)]

	if REGRESSION and not args.reg_scores_dict: 
		raise ValueError("Need to specify a dictionary for RegressionImageFolder. See help for more information.")

	epoch_n = args.epoch_n

	filename = args.filename 


	if filename is None: 
		filename = f'./results/{args.model_name}/{datetime.now()}.txt'

	write_file = open(filename, "w")

	if write_file:
		write_file.write(f'*** args: {args}\n\n')

	# create model 
	collators, model = create_model_and_collator(
		args = args, 
		model_name = args.model_name
	)

	# put on GPU 
	if torch.cuda.is_available():
		model.cuda() 

	# load data 
	data_loaders, train_class_weights = create_dataset(
		args = args, collator_fns = collators
	)

	# calculate generic dataset statistics 
	train_label_stats = label_statistics(args, data_loaders[0])
	val_label_stats = label_statistics(args, data_loaders[1])
	print(f'*** Training set label statistics: {train_label_stats}')
	print(f'*** Validation set label statistics: {val_label_stats}')

	if write_file:
		write_file.write(f'*** Training set label statistics: {train_label_stats}')
		write_file.write(f'*** Validation set label statistics: {val_label_stats}')	


	if args.model_name in ['logistic_regression', 'basic_cnn', 'dense_cnn']:
		optim = torch.optim.Adam(params = model.parameters())
	else: 
		optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, eps=args.eps)
	
	total_steps = len(data_loaders[0]) * args.epoch_n 
	scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps = total_steps)

	# ADD SUPPORT FOR CLASS WEIGHTS 
	# would be passed as weight =class_weights
	if args.regression: 
		criterion = nn.MSELoss()
	else: 
		train_class_weights = train_class_weights.cuda()
		criterion = nn.CrossEntropyLoss(weight = train_class_weights)

	if write_file: 
		write_file.write(f'\nModel:\n {model}\nOptimizer:{optim}\n')

	train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file)

	if write_file:
		write_file.close()



















