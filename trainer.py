import torch
import torch as tc
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc 

inputs_col = {'GCN': 3, 'BERT':4}

def proprocessing(model_type, inputs):
	if model_type == 'BERT':
		pdb.set_trace()
		batch_size = len(inputs)
		bert_inputs = torch.tensor(batch_size, 512, 512)
		attention_mask = torch.tensor(batch_size, 512)
		for bat in batch_size:
			len_ = len(inputs[bat])
			input_tensor = torch.cat([torch.tensor(inputs[bat])]*5 + [torch.tensor(inputs[bat])[:,:12]], dim = -1)
			bert_inputs[batch] = input_tensor
			attention_mask[bat] = torch.tensor([ int(i<len_) for i in range(512)])
	elif model_type == 'LSTM':
		pass
	else:
		_ = None
		return inputs, _

class Sampler:
	def __init__(self, dataset, batch_size, shuffle = True, equal = False):
		self.dataset = dataset 
		self.pos_dataset = [data for data in dataset if data[1] == 1]
		self.neg_dataset = [data for data in dataset if data[1] == 0]
		self.batch_size = batch_size
		self.count_batch = 0
		self.shuffle = shuffle
		self.equal = equal

	def prepare(self): # equal: 正例和负例数量相同
		equal = self.equal
		if False:
			pass
		else:
			dataset_size = len(self.dataset)
			batch_size = self.batch_size
			if self.shuffle:
				random_map = random.sample(range(dataset_size), dataset_size)
			else:
				random_map = [i for i in range(dataset_size)]
			self.batch_list = [ random_map[i:i+batch_size] for i in range(0, dataset_size ,batch_size) ]
			self.total_batch = len(self.batch_list)
			self.count_batch = 0
	def fetch(self):
		equal = self.equal
		if equal:
			batch_pos = random.sample(self.pos_dataset, self.batch_size // 2)
			batch_neg = random.sample(self.neg_dataset, self.batch_size // 2)
			batch_data = batch_pos + batch_neg
			random.shuffle(batch_data)
			return batch_data
		else:
			if(self.count_batch >= self.total_batch):
				print("Error: Fetching batch, but out of index")
			else:
				batch = self.batch_list[self.count_batch]
				batch_data = [ self.dataset[b] for b in batch]#self.dataset[batch]
			self.count_batch += 1
			return batch_data



class Trainer:
	def __init__(self, data_bundle, vocab, l_r, batch_size, epoch, models, device, criterion, optimizer, equal = False, evaluate_every=10):
		self.data_bundle = data_bundle
		self.l_r = l_r
		self.batch_size = batch_size
		self.models = models
		self.epoch = epoch
		self.device = device
		self.criterion = criterion
		self.optimizer = optimizer
		self.evaluate_every = evaluate_every
		self.equal = equal


	def train(self):
		equal = self.equal
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

		sampler = Sampler(dataset, batch_size, equal=equal, shuffle=False)
		sampler.prepare()
		batch_num = sampler.total_batch
		criterion = self.criterion
		optimizer = self.optimizer

		ensemble_weights = F.softmax(torch.randn(len(models)), dim = -1).unsqueeze(1).to(device)
		
		
		for epc in range(epoch):
			all_preds = []
			all_labels  = []
			if (epc % self.evaluate_every == 0):
				self.evaluate(dataset_name = 'dev')
				self.evaluate(dataset_name = 'test')
			total_loss = 0
			sampler.prepare()
			for bt in (range(batch_num)):
				batch_data = sampler.fetch()
				
				labels = [ data[1] for data in batch_data]
				all_labels += labels
				labels = torch.LongTensor(labels).to(device)
				preds_list = []
				for m, model in enumerate(models):
					#pdb.set_trace()
					inputs = [data[inputs_col[models_class[m]]] for data in batch_data]
					inputs, attention_mask = proprocessing(model_class[m], inputs)
					preds = model(inputs)
					preds_list.append(preds.unsqueeze(0)) 
				preds_list = torch.cat(preds_list, dim = 0)
				preds_ensemble = ensemble_weights * preds_list # 按元素乘
				preds_ensemble = preds_ensemble.sum(dim = 0)
				preds_list = [p[1].item() for p in preds_ensemble]
				all_preds += preds_list
				
				loss = criterion(preds_ensemble, labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += loss.item() * len(batch_data)

			#pdb.set_trace()
			avg_loss = total_loss / dataset_size
			roc_auc = roc_auc_score(all_labels, all_preds)
			pr, re, _ = precision_recall_curve(all_labels, all_preds)
			prc_auc = auc(re, pr)

			print("Train Epoch :{0} Avg Loss:{1:.5f} Roc_Auc:{2:.5f} Prc_Auc:{3:.5f} \n".format(epc, avg_loss,roc_auc, prc_auc))
			

	def evaluate(self, dataset_name):
		device = self.device
		models = self.models
		models_class = [model.clas for model in models]
		batch_size = self.batch_size 

		for model in models:
			model.eval()
		dataset = self.data_bundle.get(dataset_name) 
		if dataset == None:
			dataset = self.data_bundle['test'] 
		dataset_size = len(dataset)

		sampler = Sampler(dataset, batch_size)
		sampler.prepare()
		batch_num = sampler.total_batch
		criterion = self.criterion


		ensemble_weights = F.softmax(torch.randn(len(models)), dim = -1).unsqueeze(1).to(device)
		
		if(dataset[0][1] == -1):
			have_label = False
		else:
			have_label = True

		for epc in range(1):
			all_preds = []
			all_labels  = []
			total_loss = 0
			sampler.prepare()
			for bt in (range(batch_num)):
				batch_data = sampler.fetch()
				labels = [ data[1] for data in batch_data]
				all_labels += labels
				labels = torch.LongTensor(labels).to(device)
				preds_list = []
				for m, model in enumerate(models):
					inputs = [data[inputs_col[models_class[m]]] for data in batch_data]
					inputs, attention_mask = proprocessing(model_class[m], inputs)
					preds = model(inputs)
					preds_list.append(preds.unsqueeze(0)) 
				preds_list = torch.cat(preds_list, dim = 0)
				preds_ensemble = ensemble_weights * preds_list # 按元素乘
				preds_ensemble = preds_ensemble.sum(dim = 0)
				preds_list = [ p[1].item() for p in preds_ensemble]
				all_preds += preds_list

				loss = criterion(preds_ensemble, labels)
				total_loss += loss.item() * len(batch_data)
			avg_loss = total_loss / dataset_size
			#pdb.set_trace()
			if have_label:
				roc_auc = roc_auc_score(all_labels, all_preds)
				pr, re, _ = precision_recall_curve(all_labels, all_preds)
				prc_auc = auc(re, pr)
			else:
				roc_auc = None
				prc_auc = None
				avg_loss = None

			print("Evaluation Epoch :{0} Avg Loss:{1:.5f} Roc_Auc:{2:.5f} Prc_Auc:{3:.5f} \n".format(epc, avg_loss, roc_auc, prc_auc))
		
		for model in models:
			model.train()
				
				


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
