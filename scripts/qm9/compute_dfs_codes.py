import dfs_code
import json
from torch_geometric.datasets.qm9 import QM9
import tqdm
import numpy as np

path = 'datasets/qm9_torch_geometric/'
dataset = QM9(path)

dfs_codes = {}

for i in tqdm.tqdm(range(len(dataset))):
    data = dataset[i]
    vertex_features = data.x.detach().cpu().numpy()
    edge_features = data.edge_attr.detach().cpu().numpy()
    vertex_labels = np.argmax(vertex_features[:, :5], axis=1).tolist()
    edge_labels = np.argmax(edge_features, axis=1).tolist()
    code, id2row = dfs_code.min_dfs_code_from_torch_geometric(data, vertex_labels, edge_labels)
    dfs_codes[data.name] = {'min_dfs_code':code, 'edge_id_2_edge_number':id2row}

with open(path+'min_dfs_codes.json', 'w') as f:
    json.dump(dfs_codes, f)
