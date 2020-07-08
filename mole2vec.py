from mol2vec.features import MolSentence, sentences2vec, mol2alt_sentence

from gensim.models import word2vec
from rdkit import Chem

model = word2vec.Word2Vec.load('./model/model_300dim.pkl')

def mole2vec(mol: str):
    """
    mol: single mol SMILES
    """
    rd_mol = Chem.MolFromSmiles(mol)
    return sentences2vec(MolSentence(mol2alt_sentence(rd_mol, 1), model, unseen='UNK'))

def moles2vec(mols: list):
    """
    mols: list of str(SMILES)
    return: list of np.array
    """
    # model = word2vec.Word2Vec.load('./model/model_300dim.pkl')
    rd_mol = [Chem.MolFromSmiles(i) for i in mols]
    sentences = [MolSentence(mol2alt_sentence(i, 1)) for i in rd_mol]
    vecs = [sentences2vec(i, model, unseen='UNK') for i in sentences]
    return vecs