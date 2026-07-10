# Tetrahedral Chirality Stereodiscernment with Graph-Based Machine Learning

Code for associated paper as described in Zhu *et al.*.

## Dependencies

Code was developed for python 3.10.

* torch==2.2.2
* torch-geometric==2.6.1
* rdkit==2025.03.6
* numpy==1.24
* pandas

Dependencies can be installed individually *via* ```pip install PACKAGE_NAME==PACKAGE_VERSION``` but it is recommended to use the below install instructions.

## Install

To install, create a new conda environment with:

```
conda create -n stereodiscern_gnn3.10 python==3.10 -y
conda activate stereodiscern_gnn3.10
```

Then clone this repository:

```
git clone https://github.com/emmaking-smith/stereodiscerning_tetchiral
cd stereodiscerning_tetchiral/
pip install -e .
pip install "numpy<2"
```

## Using the model

To use this model for inference, first create your list of SMILES strings in a .txt file. See [this file](stereodiscerning_gnn/data/misassigned_molecules_test_smiles.txt) for an example.
Then, type:

```
cd stereodiscerning_gnn
python predict_optical_rotation.py --file YOUR_SMILES_STRING_TXT_FILE_NAME
```

This will save out the model predictions as ```predictions.csv```.

> [!NOTE] 
> The dataset used for training the model can be downloaded at this [Figshare link](https://figshare.com/articles/dataset/full_dataset/32923205?file=66386654). Please place this file in the ```stereodiscerning_gnn/data``` directory.


## Using the model for the Coding-Shy

If you're still new to coding, you can download and watch our 5-min video tutorial [here](User's%20Guide/Predictions_in_Colab_Movie.mp4) which is step-by-step guide to running the model in a Google Colab notebook. The associated .ipynb file is in the [User's Guide](User's%20Guide) directory.

## Citation

Coming Soon!