import torch
import numpy as np
import networkx as nx

from collections import Counter
from lmfit.models import PowerLawModel
from scipy.stats import skew, kurtosis
from train_utils import label2onehot


def img_transform(model, img, transform, device):
    """ make all label fake image from graph """
    graph_tensor = transform(img).reshape(1, 1, 100, 100).to(device)  # img to tensor
    trg_label = label2onehot(torch.tensor([0, 1, 2, 3]), 4).to(device)  # make target label
    origin_tensors = torch.cat([graph_tensor for _ in range(4)])  # cat origin image tensor
    # generate fake image
    with torch.no_grad():
        fake_tensor = model(origin_tensors, trg_label)
    return fake_tensor.detach().cpu()


def tensor_to_graph(tensor, thres):
    A = (tensor + tensor.T).numpy()
    A = (A > thres).astype(int)
    G = nx.from_numpy_array(A)
    return G


def power_law_fit(G):
    """ degree dist fitting """
    c = Counter(dict(G.degree()).values())
    score_sorted = sorted(c.items(), key=lambda x: x[0])
    x = [k for k, v in score_sorted if k != 0]
    y = [v for k, v in score_sorted if k != 0]
    model = PowerLawModel()
    params = model.guess(y, x=x)
    result = model.fit(y, params, x=x)
    return result.values


def net_info(G):
    """ get network info """
    def _average_path_len(G):
        if nx.is_connected(G):
            return nx.average_shortest_path_length(G)
        else:
            return None

    try:
        beta = power_law_fit(G)['exponent']
    except:
        beta = None

    degs = np.array(list(dict(G.degree()).values()))
    info = {
        "max_degree": max(degs),
        "min_degree": min(degs),
        "average_degree": np.average(degs),
        "num_edges": nx.number_of_edges(G),
        "beta": beta,  # べきfittingの指数
        "variance": np.var(degs, ddof=1),
        "mode": Counter(degs).most_common()[0][0],
        "Skewness": skew(degs),  # 歪度
        "Kurtosis": kurtosis(degs),  # 尖度
        "average_cluster": nx.average_clustering(G),  # 平均クラスター
        "average_shortest_path": _average_path_len(G)  # 平均最短経路長
    }
    return info
