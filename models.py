import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import batch , unbatch
from dgl.nn.pytorch import GraphConv
import pdb

class graph_embedding(nn.Module):
	def __init__(self, vocab, embedding_dim, device=torch.device('cuda')):
		super().__init__()
		self.embedding_element = nn.Embedding(len(vocab['elements']), embedding_dim)
		self.embedding_aromatic = nn.Embedding(len(vocab['aromatics']), embedding_dim)
		self.embedding_charge = nn.Embedding(len(vocab['charges']), embedding_dim)
		self.embedding_hcount = nn.Embedding(len(vocab['hcounts']), embedding_dim)
		self.device = device

	def forward(self, inputs):
		device = self.device
		tensor_element = self.embedding_element(inputs.ndata['element'].to(device))
		tensor_aromatic = self.embedding_aromatic(inputs.ndata['aromatic'].to(device))
		tensor_charge = self.embedding_charge(inputs.ndata['charge'].to(device))
		tensor_hcount = self.embedding_hcount(inputs.ndata['hcount'].to(device))

		tensor_inputs = torch.cat((tensor_element, tensor_aromatic, tensor_charge, tensor_hcount), dim = -1)
		return tensor_inputs

class GCN(nn.Module):
	def __init__(self, g_embedding, device, hidden_size = [300, 300], embedding_dim = 150, num_layers = 2, output_dim = 2):
		super().__init__()
		self.clas = 'GCN'
		hidden_size = [embedding_dim * 4] + hidden_size
		self.hidden_size = hidden_size
		
		self.g_embedding = g_embedding
		self.gconvs = nn.ModuleList([GraphConv(hidden_size[i], hidden_size[i+1]) for i in range(num_layers)])
		self.fc = nn.Linear(hidden_size[-1], output_dim)
		#pdb.set_trace()
		self.to(device)


	def forward(self, inputs):

		binputs = batch(inputs)

		tensor_inputs = self.g_embedding(binputs)
		tensor_outputs = tensor_inputs

		for i, gconv in enumerate(self.gconvs):
			tensor_outputs = gconv(binputs, tensor_outputs)
			tensor_outputs = F.relu(tensor_outputs)

		tensor_outputs = self.fc(tensor_outputs)
		tensor_outputs = F.softmax(tensor_outputs, dim = -1)

		binputs.ndata['preds'] = tensor_outputs
		outputs = unbatch(binputs)
		preds = [ g.ndata['preds'] for g in outputs] # 每个图中，每个点的预测值
		preds = [ pred.mean(dim = 0).unsqueeze(0) for pred in preds]
		preds = torch.cat(preds, dim = 0)

		return preds

