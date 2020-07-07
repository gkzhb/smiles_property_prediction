import os
import argparse
from tqdm import tqdm
import torch
import pdb
import random
#import fastNLP as fnlp

from data_bundle import *
from trainer import Trainer, Tester
from models import graph_embedding, GCN

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--folder_path', type=str, default='.')
	parser.add_argument('--l_r', type=float, default=1e-4)
	parser.add_argument('--batch_size', type=int, default=6)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--embedding_dim', type=int, default=50)
	parser.add_argument('--hidden_size', type=int, nargs='+', default=[50, 50])
	parser.add_argument('--num_layers', type=int, default=2)
	parser.add_argument('--use_gpu', type=bool, default=True)
	parser.add_argument('--weight_decay', type=float, default=1e-7)
	parser.add_argument('--models', type=str, nargs='+', default=['gcn'])
	
	arg = parser.parse_args()

	random.seed(arg.seed)
	np.random.seed(arg.seed)
	torch.manual_seed(arg.seed)

	n_gpu = torch.cuda.device_count()
	if n_gpu > 0:
		torch.cuda.manual_seed_all(arg.seed)

	if (arg.folder_path == '.'):
		include_dev = False
	else:
		include_dev = True

	if arg.use_gpu == True:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	data_bundle, vocab = load_data_bundle(arg.folder_path)


	g_embedding = graph_embedding(vocab, arg.embedding_dim, device)
	models = []
	for model_name in arg.models:
		if (model_name=='gcn'):
			model = GCN(g_embedding, device, arg.hidden_size, arg.embedding_dim, arg.num_layers)
		models.append(model)

	criterion = torch.nn.CrossEntropyLoss()
	#pdb.set_trace()
	optimizer = torch.optim.Adam(params = models[0].parameters(), lr =arg.l_r, weight_decay = arg.weight_decay)
		#fnlp.AdamW(params = models[0].parameters(), lr = arg.l_r, weight_decay = 1e-8) # torch.optim.Ada

	trainer = Trainer(data_bundle, vocab, arg.l_r, arg.batch_size, arg.epoch, models, device, criterion, optimizer)
	trainer.train()



