# standard libraries 

import argparse 
import random 
import numpy as np 
import collections 
from tqdm import tqdm 
from datetime import datetime

# PyTorch libraries 
import torch 
from torch.utils.data import DataLoader 
from torchvision.datasets import DatasetFolder

# Hugging Face datasets 
import datasets 

# Transformers libraries 
from transformers import TrainingArguments, Trainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import get_linear_schedule_with_warmup 

# import sklearn models 
from sklearn.naive_bayes import BernoulliNB, MultinomialNB 
# import naive_bayes_utils
#import naive_bayes_utils 

# simple models
#from models import LogisticRegression, BasicCNNModel, DenseCNNModel

from sklearn.metrics import confusion_matrix


RANDOM_SEED = 231 
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Number of classes 
CLASSES = 4
CLASS_NAMES = [i for i in range(CLASSES)]


class ImageClassificationCollator:
   def __init__(self, feature_extractor): 
      self.feature_extractor = feature_extractor
      
   def __call__(self, batch):  
      encodings = self.feature_extractor([x[0] for x in batch],
      return_tensors='pt')   
      encodings['labels'] = torch.tensor([x[1] for x in batch],    
      dtype=torch.long)
      return encodings

# create model and feature extractor 
def create_model_and_extractor(args, model_name):

	if model_name == "ViT":
		feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
		collator = ImageClassificationCollator(feature_extractor)
		model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=CLASSES)
	
	elif model_name in ['basic_cnn', 'dense_cnn', 'logistic_regression']:
		# ADD IN transforms though feature extractor might be easier 
		feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

		if model_name == "logistic_regression":
			model = LogisticRegression(n_classes=CLASSES)
		elif model_name == "basic_cnn":
			model = BasicCNNModel(n_classes=CLASSES)
		elif model_name == "dense_cnn":
			model = DenseCNNModel(n_classes=CLASSES)

	else: 
		raise NotImplementedError

	print(f'Model name: {model_name}')

	return feature_extractor, collator, model 


def create_dataset(args, collator_fn, extensions = ['.jpg.npy']):

	def npy_loader(path):
		sample = torch.from_numpy(np.load(path))
		return sample 

	# load in dataset frmom directory 
	dataset = DatasetFolder(
		root = args.data_dir, 
		loader = npy_loader, 
		extensions = extensions
	)
	# split up into train val data  
	indices = torch.randperm(len(dataset)).tolist()
	n_val = int(np.floor(len(indices) * .15))
	train_ds = torch.utils.data.Subset(dataset, indices[:-n_val])
	val_ds = torch.utils.data.Subset(dataset, indices[-n_val:])

	train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collator_fn, shuffle = 1)
	val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator_fn, shuffle=0)

	return [train_dl, val_dl]

def dataset_statistics(args, dataset_loader):
	label_stats = collections.Counter()
	for i, batch in enumerate(dataset_loader):
		inputs, labels = batch['pixel_values'], batch['labels']
		labels = labels.cpu().numpy().flatten()
		label_stats += collections.Counter(labels)
	return label_stats


def measure_accuracy(outputs, labels):
	preds = np.argmax(outputs, axis = 1).flatten()
	labels = labels.flatten()
	correct = np.sum(preds == labels)
	c_matrix = confusion_matrix(labels, preds, labels=CLASS_NAMES)
	return correct, len(labels), c_matrix 

def validation(args, val_loader, model, criterion, device, name = 'Validation', write_file=None):

	model.eval()
	total_loss = 0. 
	total_correct = 0 
	total_sample = 0 
	total_confusion = np.zeros((CLASSES, CLASSES))

	for i, batch in enumerate(tqdm(val_loader)):
		inputs, labels = batch['pixel_values'], batch['labels'] 
		inputs = inputs.to(device)
		labels = labels.to(device)

		with torch.no_grad():
			if args.model_name in ['basic_cnn', 'dense_cnn', 'logistic_regression']:
				outputs = model(inputs)
			else: 
				outputs = model(inputs)['logits'] 

		loss = criterion(outputs, labels)

		logits = outputs 
		total_loss += loss.cpu().item()

		correct_n, sample_n, c_matrix = measure_accuracy(logits.cpu().numpy(), labels.cpu().numpy())
		total_correct += correct_n 
		total_sample += sample_n 
		total_confusion += c_matrix 

	print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
	print(f'*** Confusion matrix:\n{total_confusion}')
	if write_file:
		write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
		write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')

	return total_loss, float(total_correct / total_sample) * 100 _


def train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file=None):
	print("\n>>> Training starts...")

	if write_file: 
		write_file.write("\n>>> Training starts...")

	model.train()

	best_val_acc = 0
	for epoch in range(epoch_n):
		print("*** Epoch:", epoch)
		total_train_loss = 0. 

		for i, batch in enumerate(tqdm(data_loaders[0])): 
			optim.zero_grad()
			inputs, labels = batch['pixel_values'], batch['labels'] 
			inputs = inputs.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			# forward pass 
			if args.model_name in ['basic_cnn', 'dense_cnn', 'logistic_regression']:
				outputs = model(inputs)
			else: 
				outputs = model(inputs)['logits']

			loss = criterion(outputs, labels)
			total_train_loss += loss.cpu().item()

			# backward pass 
			loss.backward()
			optim.step()
			if scheduler:
				scheduler.step()

			if i % args.val_every == 0: 
				print(f'*** Loss: {loss}')

				if write_file: 
					write_file.write(f'\nEpoch: {epoch}, Step: {i}\n')

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
	epoch_n = args.epoch_n

	filename = args.filename 


	if filename is None: 
		filename = f'./results/{args.model_name}/{datetime.now().time()}.txt'

	write_file = open(filename, "w")

	if write_file:
		write_file.write(f'*** args: {args}\n\n')

	# create model 
	feature_extractor, collator, model = create_model_and_extractor(
		args = args, 
		model_name = args.model_name
	)

	# load data 
	data_loaders = create_dataset(
		args = args, collator_fn = collator
	)

	train_label_stats = dataset_statistics(args, data_loaders[0])
	val_label_stats = dataset_statistics(args, data_loaders[1])
	print(f'*** Training set label statistics: {train_label_stats}')
	print(f'*** Validation set label statistics: {val_label_stats}')

	if write_file:
		write_file.write(f'*** Training set label statistics: {train_label_stats}')
		write_file.write(f'*** Validation set label statistics: {val_label_stats}')	


	if args.model_name in ['logistic_regression', 'basic_cnn', 'dense_cnn']:
		optim = torch.optim.Adam(parmas = model.parameters())
	else: 
		optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, eps=args.eps)
		total_steps = len(data_loaders[0]) * args.epoch_n 


	scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps = total_steps)

	# ADD SUPPORT FOR CLASS WEIGHTS 
	# would be passed as weight =class_weights
	criterion = torch.nn.CrossEntropyLoss()

	if write_file: 
		write_file.write(f'\nModel:\n {model}\nOptimizer:{optim}\n')

	train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file)

	if write_file:
		write_file.close()



















