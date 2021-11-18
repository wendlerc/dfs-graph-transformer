import pickle
import tqdm
import glob
import sys
sys.path = ['./src'] + sys.path
from dfs_transformer.datasets.utils import smiles2properties, smiles2positions
import logging
from sacred import Experiment

exp = Experiment('compute data')

@exp.config
def cfg(_log):
    nr = 0
    total = 10
    n_splits = 64
    log_level = logging.INFO
    use_Hs = False
    path = "../../results/pubchem/amd-server/noH/timeout60_4"

@exp.automain
def main(path, n_splits, nr, total, log_level, use_Hs, _run, _log):
    logging.basicConfig(level=log_level)
    for i in range(n_splits):
        if i % total == nr:
            dname = glob.glob(path+"/%d/min_dfs_codes_split*.json"%(i+1))[0]
            didx = int(dname.split("split")[-1][:-5])
            with open(path+"/%d/min_dfs_codes_split%d.pkl"%(i+1, didx), 'rb') as f:
                codes = pickle.load(f)
            d_dict = {}
            p_dict = {}
            for smiles, code in tqdm.tqdm(codes.items()):
                props, vals = smiles2properties(smiles, useHs=use_Hs)
                pos = smiles2positions(smiles, useHs=use_Hs)
                data = {prop:val for prop, val in zip(props, vals)}
                d_dict[smiles] = data
                p_dict[smiles] = pos.numpy().tolist()
            with open(path+"/%d/properties_split%d.pkl"%(i+1, didx), 'wb') as f:
                pickle.dump(d_dict, f)
            with open(path+"/%d/positions_split%d.pkl"%(i+1, didx), 'wb') as f:
                pickle.dump(p_dict, f)
            

        

