import numpy as np
import scipy.sparse as sp
import utils.helpers

def preprocess_graph(G):
    nodes = list(G.nodes())
    node_to_id = {node: i for i, node in enumerate(nodes)}
    n_links = len(G.edges())
    n_nodes = len(nodes)
    return nodes, node_to_id, n_links, n_nodes


def incidence_matrix(G, node_to_id, n_links, n_nodes):
    '''
    compute the incidence matrix A for a directed network comprising nodes and links (node, edge)
    :param G: a networkx MultiDiGraph network
    :param node_to_id: reduced edges of the directed network (comprised of source nodes, target nodes and length)
    :param n_links: amount of links in the directed network
    :param n_nodes: amount of nodes in the directed network
    :return:
    '''
    datArr = np.array([0 for x in range(0, 2 * n_links)])
    rowArr = np.array([0 for x in range(0, 2 * n_links)])
    colArr = np.array([0 for x in range(0, 2 * n_links)])
    for k, (u, v) in enumerate(list(G.edges())):
        datArr[2 * k] = -1
        datArr[2 * k + 1] = 1
        rowArr[2 * k] = node_to_id[u]
        rowArr[2 * k + 1] = node_to_id[v]
        colArr[2 * k] = k
        colArr[2 * k + 1] = k
    incidence_matrix = sp.csr_matrix((datArr, (rowArr, colArr)), shape=(n_nodes, n_links));
    return incidence_matrix

def flow_matrix(a_matrix, od_flows):
    '''
    generates the flow matrix
    :param a_matrix: the incidence matrix for the directed network
    :param od_flows:
    :return:
    '''
    n_nodes, n_links = a_matrix.shape
    # for the flows
    # flows = ODFlowDF[["EdgeId", "Flow"]];
    # flows.index = flows.EdgeId;
    # flows = flows.Flow.to_dict();
    # createMMatrixAndEstY(flows, A, l, usedFunction, o, d, matrixFolder,
    #                      estYFolder, flowXFolder);

if __name__ == '__main__':
    G = utils.helpers.get_network('Borna')
    nodes, node_to_id, n_links, n_nodes = preprocess_graph(G)
    incidence_matrix = incidence_matrix(G, node_to_id, n_links, n_nodes)
    print('done')