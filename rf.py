import sys, os
import numpy as np
import pandas as pd
from mol2vec.features import MolSentence, mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier

if len(sys.argv) < 2:
    print("python rf.py <data path>\n")
    exit()

model = word2vec.Word2Vec.load('./model/model_300dim.pkl')

def moles2vec(mols: list):
	"""
	mols: list of str(SMILES)
	return: list of np.array
	"""
	rd_mol = [Chem.MolFromSmiles(i) for i in mols]
	sentences = [MolSentence(mol2alt_sentence(i, 1)) for i in rd_mol]
	vecs = sentences2vec(sentences, model, unseen='UNK')
	return vecs

df = pd.read_csv(os.path.join(sys.argv[1], 'train.csv'))

x = df['smiles']
x = moles2vec(x)
x = np.array(x)
y = df['activity']

clf = RandomForestClassifier(n_estimators=500, random_state=0)
clf.fit(x, y)

df = pd.read_csv(os.path.join(sys.argv[1], 'test.csv'))
x_test = df['smiles']
x_test = moles2vec(x_test)
y_test = df['activity']

y_pred = clf.predict(x_test)
prob = clf.predict_proba(x_test).T[1] # Probabilities for class 1
del clf
print('roc auc: ', roc_auc_score(y_test, prob))

pr, re, _ = precision_recall_curve(y_test, prob)
prc_auc = auc(re, pr)
print('prc auc: ', prc_auc)
