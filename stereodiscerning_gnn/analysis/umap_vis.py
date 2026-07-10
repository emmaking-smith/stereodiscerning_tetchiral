'''
Visualization of Chemical Space
'''

import pandas as pd
import numpy as np
import deepchem as dc
import umap
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.patches as mpatches
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import KMeans

def get_smiles(dataset):
    smiles = []
    for item in dataset:
        smiles += item.to_dataframe()['ids'].tolist()
    return smiles

def run_umap(df):
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.05,spread=5.0,metric='jaccard')
    embedding = reducer.fit_transform(np.array(df['fingerprint'].tolist()))
    return embedding

def make_fingerprint(mol_list, generator=rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)):
    fps = []
    for mol in tqdm(mol_list):
        fps.append(np.array(generator.GetFingerprint(mol)))
    return fps

def find_mediods(fingerprints, clusters):
    medoids = []
    for cluster_id in range(clusters.n_clusters):
        cluster_points = np.array(fingerprints)[clusters.labels_ == cluster_id]
        centroid = clusters.cluster_centers_[cluster_id]
        # Find the actual data point closest to the centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        medoid_idx = np.argmin(distances)
        medoids.append(cluster_points[medoid_idx])
    return medoids

def main():
    featurizer = dc.feat.DummyFeaturizer()
    _, bace, _ = dc.molnet.load_bace_classification()
    bace = get_smiles(bace)
    _, bbbp, _ = dc.molnet.load_bbbp()
    bbbp = get_smiles(bbbp)
    _, clintox, _ = dc.molnet.load_clintox()
    clintox = get_smiles(clintox)
    _, freesolv, _ = dc.molnet.load_freesolv()
    freesolv = get_smiles(freesolv)
    _, lipo, _ = dc.molnet.load_lipo()
    lipo = get_smiles(lipo)
    _, tox21, _ = dc.molnet.load_tox21()
    tox21 = get_smiles(tox21)
    _, z, _ = dc.molnet.load_zinc15(featurizer=featurizer)
    zinc_15 = []
    for item in z:
        zinc_15 += item.to_dataframe()['X'].tolist()
    del z
    _, qm9, _ = dc.molnet.load_qm9(featurizer=featurizer)
    qm9 = get_smiles(qm9)

    drug_like_mols = bace + bbbp + clintox + freesolv + lipo + tox21 + zinc_15
    drug_like_mols = [Chem.MolToSmiles((Chem.MolFromSmiles(x))) for x in drug_like_mols]
    drug_like_mols = np.unique(drug_like_mols).tolist()
    drug_like_mols = [Chem.MolFromSmiles(x) for x in drug_like_mols]
    drug_like_fps = make_fingerprint(drug_like_mols)
    del drug_like_mols, bace, bbbp, clintox, freesolv, lipo, tox21, zinc_15

    qm9_mols = [Chem.MolFromSmiles(x) for x in qm9]
    qm9_fps = make_fingerprint(qm9_mols)
    del qm9_mols, qm9

    reaxys = pd.read_pickle('data/processed_data_with_xyz.pickle')
    reaxys_mols = [Chem.MolFromSmiles(x) for x in reaxys['SMILES']]
    reaxys_fps = make_fingerprint(reaxys_mols)
    del reaxys, reaxys_mols

    # Clustering
    kmeans = KMeans(n_clusters=500, random_state=12)
    reaxys_clusters = kmeans.fit(reaxys_fps)
    reaxys_medoids = find_mediods(reaxys_fps, reaxys_clusters)

    drug_like_clusters = kmeans.fit(drug_like_fps)
    drug_like_medoids = find_mediods(drug_like_fps, drug_like_clusters)

    qm9_clusters = kmeans.fit(qm9_fps)
    qm9_medoids = find_mediods(qm9_fps, qm9_clusters)

    clustered_umap_df = pd.DataFrame()
    clustered_umap_df['fingerprint'] = reaxys_medoids + drug_like_medoids + qm9_medoids
    clustered_umap_df['dataset'] = ['Reaxys'] * len(reaxys_medoids) + ['Drug Like'] * len(drug_like_medoids) + ['QM9'] * len(qm9_medoids)

    embedding = run_umap(clustered_umap_df)
    clustered_umap_df['UMAP 1'] = embedding[:, 0]
    clustered_umap_df['UMAP 2'] = embedding[:, 1]

    clustered_umap_df.to_pickle('clustered_umap_embeddings.pickle')

    palette = sns.color_palette()
    dataset_map = {'Reaxys': 0, 'Drug Like': 1, 'QM9': 2}

    handles = [mpatches.Patch(color=palette[v], label=k) for k, v in dataset_map.items()]

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[palette[x] for x in clustered_umap_df.dataset.map(dataset_map)],
        s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.legend(handles=handles, title='Dataset', fontsize=12, title_fontsize=13)
    plt.title('UMAP Projection of Datasets', fontsize=24)

    plt.savefig('clustered_umap.png')

if __name__ == '__main__':
    main()




