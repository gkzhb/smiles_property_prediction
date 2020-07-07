import torch
import torch as tc
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random

inputs_col = {'GCN': 3}


class Sampler:
	def __init__(self, dataset, batch_size, shuffle = True):
		self.dataset = dataset 
		self.pos_dataset = [data for data in dataset if data[1] == 1]
		self.neg_dataset = [data for data in dataset if data[1] == 0]
		self.batch_size = batch_size
		self.count_batch = 0
		self.shuffle = shuffle

	def prepare(self, equal=False): # equal: 正例和负例数量相同
		if equal:
			pass
			# 待实现
		else:
			dataset_size = len(self.dataset)
			batch_size = self.batch_size
			if self.shuffle:
				random_map = random.sample(range(dataset_size), dataset_size)
			else:
				random_map = [i for i in range(dataset_size)]
			self.batch_list = [ random_map[i:i+batch_size] for i in range(0, dataset_size+1-batch_size,batch_size) ]
			self.total_batch = len(self.batch_list)
			self.count_batch = 0
	def fetch(self, equal=False):
		if equal:
			pass
		else:
			if(self.count_batch >= self.total_batch):
				print("Error: Fetching batch, but out of index")
			else:

				batch = self.batch_list[self.count_batch]
				batch_data = [ self.dataset[b] for b in batch]#self.dataset[batch]
			self.count_batch += 1
			return batch_data



class Trainer:
	def __init__(self, data_bundle, vocab, l_r, batch_size, epoch, models, device, criterion, optimizer, evaluate_every=20):
		self.data_bundle = data_bundle
		self.l_r = l_r
		self.batch_size = batch_size
		self.models = models
		self.epoch = epoch
		self.device = device
		self.criterion = criterion
		self.optimizer = optimizer
		self.evaluate_every = evaluate_every


	def train(self):
		device = self.device
		models = self.models
		models_class = [model.clas for model in models]
		batch_size = self.batch_size 
		epoch = self.epoch 
		l_r = self.l_r
		for model in models:
			model.train()
		dataset = self.data_bundle['train']
		dataset_size = len(dataset)

		sampler = Sampler(dataset, batch_size)
		batch_num = dataset_size // batch_size
		criterion = self.criterion
		optimizer = self.optimizer

		ensemble_weights = F.softmax(torch.randn(len(models)), dim = -1).unsqueeze(1).to(device)
		
		for epc in range(epoch):
			if (epc % self.evaluate_every == 0):
				self.evaluate()
			total_loss = 0
			sampler.prepare()
			for bt in (range(batch_num)):
				batch_data = sampler.fetch()
				labels = [ data[1] for data in batch_data]
				labels = torch.LongTensor(labels).to(device)
				preds_list = []
				for m, model in enumerate(models):
					inputs = [data[inputs_col[models_class[m]]] for data in batch_data]
					preds = model(inputs)
					preds_list.append(preds.unsqueeze(0)) 
				preds_list = torch.cat(preds_list, dim = 0)
				preds_ensemble = ensemble_weights * preds_list # 按元素乘
				preds_ensemble = preds_ensemble.sum(dim = 0)
				loss = criterion(preds_ensemble, labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += loss.item() * len(batch_data)

			avg_loss = total_loss / dataset_size
			print("Train Epoch :{0} Avg Loss:{1} \n".format(epc, avg_loss))
			

	def evaluate(self):
		device = self.device
		models = self.models
		models_class = [model.clas for model in models]
		batch_size = self.batch_size 

		for model in models:
			model.eval()
		dataset = self.data_bundle.get('dev') 
		if dataset == None:
			dataset = self.data_bundle['test'] 
		dataset_size = len(dataset)

		sampler = Sampler(dataset, batch_size, shuffle= False)
		batch_num = dataset_size // batch_size
		criterion = self.criterion

		ensemble_weights = F.softmax(torch.randn(len(models)), dim = -1).unsqueeze(1).to(device)
		
		for epc in range(1):
			total_loss = 0
			sampler.prepare()
			for bt in (range(batch_num)):
				batch_data = sampler.fetch()
				labels = [ data[1] for data in batch_data]
				labels = torch.LongTensor(labels).to(device)
				preds_list = []
				for m, model in enumerate(models):
					inputs = [data[inputs_col[models_class[m]]] for data in batch_data]
					preds = model(inputs)
					preds_list.append(preds.unsqueeze(0)) 
				preds_list = torch.cat(preds_list, dim = 0)
				preds_ensemble = ensemble_weights * preds_list # 按元素乘
				preds_ensemble = preds_ensemble.sum(dim = 0)
				loss = criterion(preds_ensemble, labels)
				total_loss += loss.item() * len(batch_data)
			avg_loss = total_loss / dataset_size
			print("Evaluation : Avg Loss:{0} \n".format(avg_loss))


				
				


class Tester:
	def __init__(self, data_bundle, vocab, l_r, batch_size, epoch, models, device, criterion, optimizer, evaluate_every=20, have_label = True):
		self.data_bundle = data_bundle
		self.l_r = l_r
		self.batch_size = batch_size
		self.models = models
		self.epoch = epoch
		self.device = device
		self.criterion = criterion
		self.optimizer = optimizer
		self.evaluate_every = evaluate_every
		self.have_label = have_label

	def test(self):
		device = self.device
		models = self.models
		models_class = [model.clas for model in models]
		batch_size = self.batch_size 

		for model in models:
			model.eval()
		dataset = self.data_bundle.get('test') 
		dataset_size = len(dataset)

		sampler = Sampler(dataset, batch_size, shuffle= False)
		batch_num = dataset_size // batch_size
		criterion = self.criterion

		ensemble_weights = F.softmax(torch.randn(len(models)), dim = -1).unsqueeze(1).to(device)
		
		for epc in range(1):
			total_loss = 0
			sampler.prepare()
			for bt in (range(batch_num)):
				batch_data = sampler.fetch()
				preds_list = []
				for m, model in enumerate(models):
					inputs = [data[inputs_col[models_class[m]]] for data in batch_data]
					preds = model(inputs)
					preds_list.append(preds.unsqueeze(0)) 
				preds_list = torch.cat(preds_list, dim = 0)
				preds_ensemble = ensemble_weights * preds_list # 按元素乘
				preds_ensemble = preds_ensemble.sum(dim = 0)

				if self.have_label:
					labels = [ data[1] for data in batch_data]
					labels = torch.LongTensor(labels).to(device)
					loss = criterion(preds_ensemble, labels)
					total_loss += loss.item() * len(batch_data)
			avg_loss = total_loss / dataset_size
			if self.have_label:
				print("Test : Avg Loss:{1} \n".format(avg_loss))
			else:
				print("Test finish .\n")
