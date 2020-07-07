import os
import numpy as np 
import torch
import pdb
import pysmiles
from dgl import DGLGraph

def load_data(path):
	data_bundle = []
	#pdb.set_trace()
	if not os.path.exists(path):
		print("Data file not exist :", path)
		return False
	with open(path, 'r+') as f:
		#pdb.set_trace()
		f.readline()
		for i, line in enumerate(f):
			line_ = line.strip().split(",")
			if (len(line_) > 2 ): # train.csv文件有3列，第一行为id，迷惑
				line_ = line_[-2:]
			if (line_[1] == ''): # test.csv文件只有1列，没有test
				line_[1] = -1
			mol, label = line_
			data_bundle.append([mol, int(label), pysmiles.read_smiles(mol)])
	return data_bundle

def load_data_bundle(folder_path, include_dev = False):
	datasets = ['train', 'test'] + (['dev'] if include_dev else [])
	paths = [folder_path+'/'+dataset+'.csv' for dataset in datasets]
	data_bundle = { dataset: load_data(paths[i]) for i, dataset in enumerate(datasets)}
	vocab = parse_graph(data_bundle) 
	# data_bundle的每个dataset增加第三列：Dgraph 
	# 同时可以通过vocab['element/aromatics'] 来访问element的标签集合
	return data_bundle, vocab


def parse_graph(data_bundle):
	
	elements = []
	aromatics = []
	charges = []
	hcounts = []
	orders = []
	for ky in data_bundle.keys():
		dataset = data_bundle[ky]
		for i, data in enumerate(dataset):
			graph = data[2]
			graph = graph.to_directed()
			elements += [graph.nodes[g]['element'] for g in graph]
			aromatics += [graph.nodes[g]['aromatic'] for g in graph]
			charges += [graph.nodes[g]['charge'] for g in graph]
			hcounts += [graph.nodes[g]['hcount'] for g in graph]
			orders += [graph[u][v]["order"] for u, v in graph.edges]
			#pdb.set_trace()

	vocab = {}
	elements = list(set(elements))
	aromatics = list(set(aromatics))
	charges = list(set(charges))
	hcounts = list(set(hcounts))
	orders = list(set(orders))

	vocab['elements'] = elements
	vocab['aromatics'] = aromatics
	vocab['charges'] = charges
	vocab['hcounts'] = hcounts
	vocab['orders'] = orders
	
	ele_to_idx = { ele: i for i, ele in enumerate(elements)}
	cha_to_idx = { cha: i for i, cha in enumerate(charges)}

	for ky in data_bundle.keys():
		dataset = data_bundle[ky]
		for i, data in enumerate(dataset):
			graph = data[2]
			graph = graph.to_directed()
			feature_element = torch.LongTensor([ele_to_idx[graph.nodes[g]['element']] for g in graph])
			feature_aromatic = torch.LongTensor([int(graph.nodes[g]['aromatic']) for g in graph])
			feature_charge = torch.LongTensor([cha_to_idx[int(graph.nodes[g]['charge'])] for g in graph])
			feature_hcount = torch.LongTensor([int(graph.nodes[g]['hcount']) for g in graph])
			feature_order = torch.LongTensor([int(graph[u][v]["order"]) for u, v in graph.edges])

			dgraph = DGLGraph(graph)
			dgraph.ndata['element'] = feature_element
			dgraph.ndata['aromatic'] = feature_aromatic
			dgraph.ndata['charge'] = feature_charge
			dgraph.ndata['hcount'] = feature_hcount
			dgraph.edata['order'] = feature_order

			dgraph.add_edges(dgraph.nodes(), dgraph.nodes())
			data_bundle[ky][i].append(dgraph)

	return vocab