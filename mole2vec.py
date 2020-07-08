from mol2vec import features
from gensim.models import word2vec
from rdkit import Chem

def mole2vec(mols: list):
    """
    mols: list of str(SMILES)
    return: list of np.array
    """
    pdb.set_trace()
    model = word2vec.Word2Vec.load('./model/model_300dim.pkl')
    rd_mol = [Chem.MolFromSmiles(i) for i in mols]
    sentences = [features.MolSentence(features.mol2alt_sentence(i, 1)) for i in rd_mol]
    vecs = [features.sentences2vec(i, model, unseen='UNK') for i in sentences]
    return vecs