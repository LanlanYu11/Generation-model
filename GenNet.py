import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt


class GenData(object):
  def __delete_list(self, data_list, element):
    index = data_list.index(element)
    return data_list[0: index] + data_list[index+1: len(data_list)], index
  
  def __delete_index(self, data_list, index):
    return data_list[0: index] + data_list[index+1: len(data_list)]

  def __find_graph(self, graph):
    tuple = []
    for (u, v, wt) in graph.edges(data=True):   # G.edges.data('weight', default=1)
      tuple.append([u, v, wt['weight']])
    return sorted(tuple, key=lambda x:x[2])

  def draw_graph(self, graph):
    #plt.figure()
    nx.draw(graph)
    plt.show()
  """
    generate E-R model
    node_num: the number of nodes in each network
    inter_p: probility of each edge
    graph_num: the number of networks
  """
  def gen_er(self, node_num, inter_p, graph_num = 1):
    graphs = []
    for i in range(graph_num):
      graphs.append(nx.erdos_renyi_graph(node_num, inter_p))
    if graph_num == 1:
      #self.draw_graph(graphs[0])
      return graphs[0]
    else:
      return graphs
  """
    generate regular graph model
    node_num: the number of nodes in each network
    degree_num: degree of each node
    graph_num: the number of networks
  """
  def gen_rg(self, node_num, degree_num, graph_num = 1):
    graphs = []
    for i in range(graph_num):
      graphs.append(nx.random_graphs.random_regular_graph(degree_num, node_num))
    if graph_num == 1:
      return graphs[0]
    else:
      return graphs
  """
    generate BA model
    node_num: the number of nodes in each network
    inter_p: probility of each edge
    join_edge_num: the number of node for each iteration
    graph_num: the number of networks
  """
  def gen_ba(self, node_num, inter_p, join_edge_num = 1, graph_num = 1):
    graphs = []
    for i in range(graph_num):
      graphs.append(nx.barabasi_albert_graph(node_num, join_edge_num))
    if graph_num == 1:
      return graphs[0]
    else:
      return graphs
  """
    generate WS model
    node_num: the number of nodes in each network
    neigh_num: number of neighbors for each node
    redge_p: probility of reconnection
    graph_num: the number of networks
  """
  def gen_ws(self, node_num, neigh_num, redge_p, graph_num = 1):
    graphs = []
    for i in range(graph_num):
      graphs.append(nx.watts_strogatz_graph(node_num, neigh_num, redge_p))
    if graph_num == 1:
      return graphs[0]
    else:
      return graphs
  """
    generate Chung-Lu model
    node_num: the number of nodes in each network
    edge_num: number of edges in each network
    phi: weight of nodes in each network
    graph_num: the number of networks
  """
  def gen_cl(self, node_num, edge_num, phi, graph_num = 1):
    graphs = []
    for i in range(graph_num):
      # creat a null graph
      graph = nx.Graph()
      # give the graph all element of this list, which is {0, 1, ..., n-1}
      nodes = [node for node in range(node_num)]
      graph.add_nodes_from(nodes)
      print(graph.nodes(), edge_num)
      # number of phi is equal to node_num
      queue = []; tphi = phi
      while len(graph.edges()) < edge_num:
        if len(queue) == 0:
          node0 = np.random.choice(nodes, size=1, replace=False, p=tphi)[0]  # sample
        else:
          node0 = queue[0]    # pop for queue
          queue = self.__delete_index(queue,0)
        node1 = np.random.choice(nodes, size=1, replace=False, p=tphi)[0]   # sample
        #print(node0, node1)
        if node0 == node1:
          continue
        if (node0, node1) in graph.edges():   # it needs to repeat sample
          queue.append(node0)
          queue.append(node1)
        else:
          print(node0, node1)
          graph.add_weighted_edges_from([(node0, node1, 1.0),(node1, node0, 1.0)])
      graphs.append(graph)
    if graph_num == 1:
      return graphs[0]
    else:
      return graphs
  """
    generate Transitive Chung Lu (TCL) model(from Fast Generation of Large Scale Social Networks with Clustering)
    node_num: the number of nodes in each network
    edge_num: number of edges in each network
    phi: weight of nodes in each network
    beta: probility of reconnection
    iteration: the number of reconnection
    graph_num: the number of networks
  """
  def gen_tcl(self, node_num, edge_num, phi, beta, graph_num = 1, iteration=5):
    graphs = self.gen_cl(node_num, edge_num, phi, graph_num)
    print(graphs)
    for i in range(graph_num):
      graph = graphs[i]
      queue = [];
      t = 0
      while t < iteration:
        if len(queue) == 0:
          node0 = np.random.choice(node_num, size=1, replace=False, p=phi)[0]  # sample
        else:
          node0 = queue[0]    # pop for queue
          queue = self.__delete_index(queue,0)
        r = np.random.choice([0, 1], size=1, replace=False, p =[beta, 1-beta])[0]   # bernoulli sample
        if r == 1:
          print(graph)
          neigh_node0 = np.random.choice(list(graph.neighbors(node0)), size=1)[0]
          node1 = np.random.choice(list(graph.neighbors(neigh_node0)), size=1)[0]
        else:
          node1 = np.random.choice(node_num, size=1, replace=False, p=phi)[0]   # sample
        if node0 == node1:
          continue
        for (u, v, d) in graph.edges(data=True):
          d['weight'] += 1
        if (node0, node1) in graph.edges():   # it needs to repeat sample
          queue.append(node0)
          queue.append(node1)
        else:
          graph.add_weighted_edges_from([(node0, node1, 1), (node1, node0, 1)])   # least weight is one
          tuple = self.__find_graph(graph)  # seriation with weight 0,1,2,3,4,..
          old_edge = tuple[-1]   # remove oldest edge
          graph.remove_edge(old_edge[0], old_edge[1])
    if graph_num == 1:
      return graphs[0]
    else:
      return graphs
  """
    generate Stochastic-Block model
    list_signal: the signal. if it is diffrent among the number of nodes for each clusters(communities), list_signal is 1, otherwise it's 0.
    node_num: the number of nodes in each network
    cluster_num: number of clusters in each network
    inter_p: probility for edge in the same cluster
    outer_p: probility of edge between teh different clusters
    graph_num: the number of networks
  """
  def gen_sbm(self, list_signal, node_num, cluster_num, inter_p, outer_p, graph_num = 1):
    # inter_p >> outer_p
    clusters = []
    if list_signal:   # node_num is list of the size of cluster, sum(node_num) == size of teh graph
      for j in range(cluster_num):
        clusters.append(self.gen_er(node_num[j], inter_p, 1))
    else:  # node_num is the size of each of cluster in graph
      clusters = self.gen_er(node_num, inter_p, cluster_num)
    graphs = []
    for i in range(graph_num):
      graph = nx.Graph()
      for j in range(cluster_num):
        graph = nx.disjoint_union(graph,clusters[j])
      node_sum = 0
      for j in range(cluster_num):
        for k in range(cluster_num):
          if j == k:
            continue
          node_num0 = clusters[j].number_of_nodes()
          node_num1 = clusters[k].number_of_nodes()
          for n in range(node_num0):
            if np.random.rand() < outer_p:
              target = np.random.choice(node_num1) + node_num0
              graph.add_edges_from([(n+node_sum, target), (target, n+node_sum)])
          node_sum = node_sum + node_num0
      graphs.append(graph)
    if graph_num == 1:
      return graphs[0]
    else:
      return graphs

# test class
if __name__ == '__main__':
  gen = GenData()
  #gen.draw_graph(gen.gen_rg(10,2))
  #gen.draw_graph(gen.gen_ba(10,0.2))
  #gen.draw_graph(gen.gen_er(10,0.2))
  #gen.draw_graph(gen.gen_ws(10,2, 0.3))
  #gen.draw_graph(gen.gen_sbm(0, 5, 2, 0.5, 0.1))
  #gen.draw_graph(gen.gen_sbm(1, [2,8], 2, 0.5, 0.1))
  #gen.draw_graph(gen.gen_tcl(10, 20, [0.1,0.05,0.2,0.1,0.1,0.05,0.1,0.1,0.1,0.1], 0.1))
  #gen.draw_graph(gen.gen_cl(10, 20, [0.1,0.1,0.11,0.1,0.11,0.09,0.1,0.1,0.09,0.1]))