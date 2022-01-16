import numpy as np
import networkx as nx
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from train_utils import label2onehot
from collections import Counter
from lmfit.models import PowerLawModel
from scipy.stats import skew, kurtosis


def make_graph(graph_index, graph_indicator, edges_list, direct=True):
    """ make graph from txt data """
    # node id list
    node_ids = np.where(graph_indicator == graph_index)[0] + 1
    # edge抽出
    start_row_index = np.where(edges_list == node_ids[0])[0][0]
    fin_row_index = np.where(edges_list == node_ids[-1])[0][1]
    edges = []
    for e in edges_list[start_row_index : fin_row_index + 1]:
        if e[0] in node_ids and e[1] in node_ids:
            edges.append(e)
    # make graph
    if direct:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(node_ids)
    G.add_edges_from(edges)

    return G


def relabel_nodes_byDegree(G):
    """ relabel nodes by degree"""
    # relabel nodes range(0,100)
    mapping = {n: i for i, n in enumerate(list(G.nodes))}
    nx.relabel_nodes(G, mapping, copy=False)
    degs = dict(G.degree())
    sort_degs = sorted(degs.items(), key=lambda x: x[1])
    sort_nodes = [n for n, d in sort_degs]
    mapping = {n: i for i, n in enumerate(sort_nodes)}
    G = nx.relabel_nodes(G, mapping)
    return G


def network_to_image(G, sort=False, t=False):
    def _sort_by_degree(A, G):
        # 隣接行列を次数の昇順に並び替える
        # 次数の辞書を取得
        degs = dict(G.degree())
        # value(次数)で並び替え
        sort_degs = sorted(degs.items(), key=lambda x: x[1])
        sort_nodes = [node[0] for node in sort_degs]
        # 行, 列並び替え
        A = A[:, sort_nodes]
        A = A[sort_nodes, :]
        return A

    # 隣接行列の作成
    A = nx.to_numpy_array(G, nodelist=range(100))
    # sort
    if sort:
        A = _sort_by_degree(A, G)
    if t:
        A = A + A.T
    # array to image
    img = Image.fromarray(A * 255).convert("L")
    return img


def thres_tensor(tensor, thres, t=True):
    if t:
        A = (tensor + tensor.T).numpy()
    else:
        A = tensor.numpy()
    A = (A > thres).astype(int)
    return A


def degs(G):
    return list(dict(G.degree()).values())


def make_input_tensor(size100_index, graph_indicator, edges_list, d=True):
    transform = transforms.Compose(
        [
            transforms.Resize(100),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    imgs_tensor = torch.Tensor()
    graphs = []

    for index in tqdm(size100_index):
        # load graph
        G = make_graph(index, graph_indicator, edges_list, direct=d)
        G = relabel_nodes_byDegree(G)
        graphs.append(G)
        # graph -> image
        img = network_to_image(G)
        imgs_tensor = torch.cat([imgs_tensor, transform(img)])

    imgs_tensor = imgs_tensor[:, None, :, :]
    return imgs_tensor, graphs


def make_fake_tensor(input_tensor, model, device):
    """ make all label fake image from graph """
    labels = [0 for _ in range(input_tensor.shape[0])]
    trg_label = label2onehot(torch.tensor(labels), 4).to(device)  # make target label
    # generate fake image
    with torch.no_grad():
        fake_tensor = model(input_tensor.to(device), trg_label)
    return fake_tensor.detach().cpu()


def fake_tensor_toGraph(fake_tensor, d):
    if d:
        fake_A = thres_tensor(fake_tensor, 0.5, t=False)
    else:
        fake_A = thres_tensor(fake_tensor, 1, t=True)
    # adjacecy matrix -> Graph
    if d:
        kind_graph = nx.DiGraph()
    else:
        kind_graph = nx.Graph()
    fake_G = nx.from_numpy_array(fake_A, create_using=kind_graph)
    return fake_G


def net_info(G):
    """ get network info """

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

    try:
        beta = power_law_fit(G)["exponent"]
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
    }
    return info


def count_same_edges(G1, G2, d=True):
    """ 同一エッジ数 """
    if d:
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
    else:
        edges1 = set()
        for e in G1.edges():
            edges1.add(tuple(sorted(e)))
        edges2 = set()
        for e in G2.edges():
            edges2.add(tuple(sorted(e)))
    return len(edges1 & edges2)
