import sys, os
import numpy as np
import pandas as pd
from mol2vec.features import MolSentence, mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit import Chem
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier


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

def train(path):
    print('training ', path)
    df = pd.read_csv(os.path.join(path, 'train.csv'))

    x = df['smiles']
    x = moles2vec(x)
    x = np.array(x)
    y = df['activity']

    clf = RandomForestClassifier(n_estimators=500, oob_score=False, random_state=0)
    clf.fit(x, y)

    df = pd.read_csv(os.path.join(path, 'test.csv'))
    x_test = df['smiles']
    x_test = moles2vec(x_test)
    y_test = df['activity']

    y_pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test).T[1]
    del clf
    roc_auc = roc_auc_score(y_test, prob)

    pr, re, _ = precision_recall_curve(y_test, prob)
    prc_auc = auc(re, pr)
    return roc_auc, prc_auc

def train_all(path):
    t = 'fold_{}'
    rocs, prcs = [], []
    for i in range(10):
        roc, prc = train(os.path.join(path, t.format(i)))
        rocs.append(roc)
        prcs.append(prc)
    for i in range(10):
        print('fold_{}:\n  roc auc: {}\n  prc auc: {}'.format(i, rocs[i], prcs[i]))
    print(f'mean and std for roc aucs: {np.mean(rocs), np.std(rocs)}')
    print(f'mean and std for prc aucs: {np.mean(prcs), np.std(prcs)}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/train_cv')
    arg = parser.parse_args()
    train_all(arg.path)

