import os
import numpy as np 
import torch
import pdb
import pysmiles
from dgl import DGLGraph

from mol2vec.features import MolSentence, sentences2vec, mol2alt_sentence
from gensim.models import word2vec
from rdkit import Chem

model = word2vec.Word2Vec.load('./model/model_300dim.pkl')

def mole2vec(mol: str):
	"""
	mol: str(SMILES)
	return: np.array
	"""
	rd_mol = Chem.MolFromSmiles(mol)
	sentence = MolSentence(mol2alt_sentence(rd_mol, 1))
	return sentences2vec(sentence, model, unseen='UNK')

def moles2vec(mols: list):
	"""
	mols: list of str(SMILES)
	return: list of np.array
	"""
	rd_mol = [Chem.MolFromSmiles(i) for i in mols]
	sentences = [MolSentence(mol2alt_sentence(i, 1)) for i in rd_mol]
	vecs = [sentences2vec(i, model, unseen='UNK') for i in sentences]
	return vecs

def load_data(path):
	data_bundle = []
	if not os.path.exists(path):
		print("Data file not exist :", path)
		return False
	with open(path, 'r+') as f:
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
	include_dev = 'fold' in folder_path
	datasets = ['train', 'test'] + (['dev'] if include_dev else [])
	paths = [folder_path+'/'+dataset+'.csv' for dataset in datasets]
	data_bundle = { dataset: load_data(paths[i]) for i, dataset in enumerate(datasets)}
	#pdb.set_trace()
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

	#pdb.set_trace()
	
	ele_to_idx = { ele: i for i, ele in enumerate(elements)}
	cha_to_idx = { cha: i for i, cha in enumerate(charges)}

	for ky in data_bundle.keys():
		dataset = data_bundle[ky]
		smiles = [data[0] for data in dataset] 

		mol_vec = moles2vec(smiles)

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
			data_bundle[ky][i].append(mol_vec[i])

	return vocab
