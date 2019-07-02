import numpy as np
import networkx as nx
import scipy.sparse as scsp
import scipy.io as scio
import os
import copy

import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt


class GenData(object):
  def __init__(self):
    self.ratio_threshold = 0.2  # 在 受限的 SBM 中 任一节点 外部连变数 与 内部连边数目的 最高 比值
    self.ratio_upper = 0.4  # 在 受限的 SBM 中 指定节点 外部边数 与 内部连变数 相近，最低比值

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

  def __get_degree_ratio(self, G, cluster_size, type='sorted', node=None):
    """
    计算节点与其他社团的连边 与 自身社团内部的连边 的比值
    :param G:
    :param cluster_size:
    :param type:
    :return:
    """
    labels = []
    cluster_num = len(cluster_size)
    for i in range(cluster_num):
      labels = labels + [i for k in range(cluster_size[i])]
    # print(labels)
    ratio_dict = {}
    for u in list(G):
      u_label = labels[u]
      ratio = [0, 0]  # intar/ inter, {与外部的连边数， 与社团内部的连边数}
      for v in G.neighbors(u):
        v_label = labels[v]
        if u_label == v_label:
          ratio[1] += 1
        else:
          ratio[0] += 1
      ratio_dict[u] = ratio[0] * 1.0 / ratio[1]
    if type == 'sorted':
      ratio_tuple = sorted(ratio_dict.items(), key=lambda x: x[1], reverse=True)
      # print(ratio_tuple)
      return ratio_tuple
    elif type == 'dict':
      return ratio_dict
    elif type == 'node' and node != None:
      return ratio_dict[node]
    else:
      print('ParameterError')
      exit()

  def draw_graph(self, graph):
    # print(type(graph))
    if type(graph) == type([]):
      l = len(graph)
      for i in range(l):
        plt.figure()
        ax = nx.draw(graph[i])
    else:
      plt.figure()
      nx.draw(graph,)
    plt.show()

  def nullModel(self, graph, flage=0, repeat=1):
    G = nx.Graph(graph)
    # print(G.edges())
    if flage == 0:  # 0 null model
      while repeat>0:
        edge1 = graph.edges()[np.random.choice(graph.number_of_edges())]
        node1, node2 = np.random.choice(G.number_of_nodes(), size=2, replace=False)
        # print(G.has_edge(node1, node2))
        while G.has_edge(node1, node2) or G.has_edge(node2, node1):
          # print((node1, node2))
          node1, node2 = np.random.choice(G.number_of_nodes(), size=2, replace=False)
        # print(edge1, node1, node2)
        try:
          G.remove_edge(edge1[0], edge1[1])
          G.add_edge(node1, node2)
        finally:
          repeat -= 1
    elif flage == 1:  # 1 null model
      while repeat>0:
        index1, index2 = np.random.choice(graph.number_of_nodes(), size=2, replace=False)
        edge1 = graph.edges()[index1]
        edge2 = graph.edges()[index2]
        node1, node2 = edge1
        node3, node4 = edge2
        if ~graph.has_edge(node1, node3) and ~graph.has_edge(node2, node4):
          try:
            G.remove_edge(node1, node2)
            G.remove_edge(node3, node4)
            G.add_edge(node1, node4)
            G.add_edge(node3, node2)
          finally:
            repeat -= 1
    elif flage == 2:  # 2 null model
      while repeat>0:
        index1, index2 = np.random.choice(graph.number_of_nodes(), size=2, replace=False)
        edge1 = graph.edges()[index1]
        edge2 = graph.edges()[index2]
        node1, node2 = edge1
        node3, node4 = edge2
        if ~graph.has_edge(node1, node3) and ~graph.has_edge(node2, node4) and G.degree(node2) == G.degree(node4):
          try:
            G.remove_edge(node1, node2)
            G.remove_edge(node3, node4)
            G.add_edge(node1, node4)
            G.add_edge(node3, node2)
          except:
            print('error')
          finally:
            repeat -= 1
    return G

  def migrate_c0_c1(self, g_, clusters_, size=10):
      #  One of the communities is continuously diminished by migrating the 10-20 nodes
      #  to the other community(choice 10 nodes, label it).
      #  A total of 10 dynamic graphs are generated for the evaluation.
      g = copy.copy(g_)
      clusters = copy.copy(clusters_)
      graph = nx.Graph(g)
      from_c = 0; to_c = 1;
      node_sum0 = clusters[from_c].number_of_nodes()
      node_sum1 = clusters[to_c].number_of_nodes()
      # choice 10 nodes
      change_node_list = np.random.choice(clusters[from_c].number_of_nodes(), size=size, replace=False)
      print('0th communities is continuously diminished by migrating the 10 nodes to 1th community. ')
      print(change_node_list)
      # changes edges between two cluster
      remove_edge_list = []; remove_c1edge_list = []; remove_c2edge_list = []
      add_edge_list = []; add_c1edge_list = []; add_c2edge_list = []
      for n in change_node_list:
        for n2 in graph.neighbors(n):
          if n2 < node_sum0:  # n2 in from_cluster
            remove_edge_list.append((n, n2))  # remove
            remove_c1edge_list.append((n, n2))
            if (n2+node_sum0)>=(node_sum0+node_sum1):  # 超出了就随机选一个
              nn = np.random.randint(node_sum0, node_sum1+node_sum0)
              add_edge_list.append((n, nn))  # add
              add_c2edge_list.append((n, nn))
            else:
              add_edge_list.append((n, n2 + node_sum0))  # add
              add_c2edge_list.append((n, n2 + node_sum0))
          else:  # n2 in to_cluster
            remove_edge_list.append((n, n2))  # remove
            add_edge_list.append((n, n2 - node_sum0))  # add
      graph.remove_edges_from(remove_edge_list)
      clusters[0].remove_edges_from(remove_c1edge_list)
      graph.add_edges_from(add_edge_list)
      clusters[1].add_edges_from(add_c2edge_list)
      return graph, clusters

  def migrate_c0_c1_new(self, g, clusters_, size=10):
      #  One of the communities is continuously diminished by migrating the 10-20 nodes
      #  to the other community(choice 10 nodes, label it).
      #  A total of 10 dynamic graphs are generated for the evaluation.
      # copy:
      clusters = []
      for cluster in clusters_:
        clusters.append(nx.Graph(cluster))
      graph = nx.Graph(g)
      # copy end
      from_cluster = clusters[0]
      to_cluster = clusters[1]
      node_num0 = from_cluster.number_of_nodes()
      node_sum0 = 0
      node_num1 = to_cluster.number_of_nodes()
      node_sum1 = node_num0
      # choice 10 nodes
      change_node_list = np.random.choice(from_cluster.nodes(), size=size, replace=False)
      print('0th communities is continuously diminished by migrating the 10 nodes to 1th community. ')
      print(change_node_list)
      # changes edges between two cluster
      node_counter = 0
      for node0 in change_node_list:
        neighbors = list(graph.neighbors(node0))
        for node1 in neighbors:
          if (node_sum0 < node1 < node_num0):  # n2 in from_cluster
            # remove edge inter-cluster
            if graph.has_edge(node0, node1):
              graph.remove_edge(node0, node1)
            if from_cluster.has_edge(node0, node1):
              from_cluster.remove_edge(node0, node1)
            # add edge itar-cluster
            n = np.random.choice(node_num1, replace=False)  # 从1号原始社团随机选取一个节点
            to_cluster.add_edge(node_counter + node_num1, n)
            node_counter += 1
            # 转移--->内部转外部
            node2 = n + node_sum1
            graph.add_edge(node0, node2)
          elif (node_num0 < node1 < (node_sum1 + node_num1)):  # n2 in to_cluster
            graph.remove_edge(node0, node1)  # remove
            n = np.random.choice(list(from_cluster.nodes()), replace=False)  # 从0号社团随机选取一个不是改变点的节点
            while n in change_node_list:
              n = np.random.choice(list(from_cluster.nodes()), replace=False)
            node2 = n
            graph.add_edge(node0, node2)
          else:  # n2 in others cluster
            graph.remove_edge(node0, node1)  # remove
            n = np.random.choice(list(from_cluster.nodes()), replace=False)  # 从0号社团随机选取一个不是改变点的节点
            while n in change_node_list:
              n = np.random.choice(list(from_cluster.nodes()), replace=False)
            node2 = n
            graph.add_edge(node2, node1)
      from_cluster.remove_nodes_from(change_node_list)
      return graph, clusters

  def migrate_clusters(self, g, clusters_, from_index, to_index, size=10, nodes=[]):
      '''
      One of the communities is continuously diminished by migrating some nodes to the other community
      :param g: nx.Graph
      :param clusters_: clusters in g
      :param from_index: index of cluster
      :param to_index:
      :param size: number of nodes which will be moved
      :param nodes: list of nodes which will be moved from from_index to to_index
      :return: new graph and cluster
      '''
      # copy:
      clusters = []
      cluster_size = []
      for cluster in clusters_:
        clusters.append(nx.Graph(cluster))
        cluster_size.append(cluster.number_of_nodes())
      graph = nx.Graph(g)
      # copy end
      from_cluster = clusters[from_index]
      to_cluster = clusters[to_index]
      node_num0 = from_cluster.number_of_nodes()
      node_sum0 = 0
      for i in range(from_index):
        node_sum0 += clusters[i].number_of_nodes()
      node_num1 = to_cluster.number_of_nodes()
      node_sum1 = 0
      for i in range(to_index):
        node_sum1 += clusters[i].number_of_nodes()

      if len(nodes) != 0:  # 指定变化节点
        change_node_list = nodes
      else: # 随机选取节点
        # choice 10 nodes
        change_node_list = np.random.choice(from_cluster.nodes(), size=size, replace=False)
      print('0th communities is continuously diminished by migrating the 10 nodes to 1th community. ')
      print(change_node_list)
      # changes edges between two cluster
      node_counter = 0
      for node0 in change_node_list:
        neighbors = list(graph.neighbors(node0))
        for node1 in neighbors:
          if (node_sum0 < node1 < (node_sum0 + node_num0)):  # n2 in from_cluster
            # remove edge inter-cluster
            if graph.has_edge(node0, node1):
              graph.remove_edge(node0, node1)
            if from_cluster.has_edge(node0-node_sum0, node1-node_sum0):
              from_cluster.remove_edge(node0-node_sum0, node1-node_sum0)
            # add edge itar-cluster
            n = np.random.choice(node_num1, replace=False)  # 从1号原始社团随机选取一个节点
            to_cluster.add_edge(node_counter + node_num1, n)  # 转移的新节点，到该社团内的节点
            node_counter += 1   # 转移的节点标记
            # 转移--->内部转外部
            node2 = n + node_sum1
            graph.add_edge(node0, node2)
          elif (node_sum1 < node1 < (node_sum1 + node_num1)):  # n2 in to_cluster
            graph.remove_edge(node0, node1)  # remove
            n = np.random.choice(node_num0, replace=False)  # 从0号社团随机选取一个不是改变点的节点
            while (n+node_sum0) in change_node_list:
              n = np.random.choice(node_num0, replace=False)
            node2 = n+node_sum0
            graph.add_edge(node0, node2)
          # else:  # n2 in others cluster 不操作
          #   graph.remove_edge(node0, node1)  # remove
          #   n = np.random.choice(list(from_cluster.nodes()), replace=False)  # 从0号社团随机选取一个不是改变点的节点
          #   while n in change_node_list:
          #     n = np.random.choice(list(from_cluster.nodes()), replace=False)
          #   node2 = n
          #   graph.add_edge(node2, node1)
      from_cluster.remove_nodes_from(change_node_list)
      # print(self.__get_degree_ratio(graph, cluster_size))
      return graph, clusters

  def change_sbm(self, g, cluster_num, clusters, change_inter_p=None, change_node_p=None,
                 change_edge=None, change_intar_p=None):
    """
    :param g: graph will be changed
    :param cluster_num: number of clusters in g
    :param clusters: clusters in g
    :param change_inter_p:  probability of changing in each cluster
    :param change_node_p:  ratio of changing node of cluster
    :param change_edge:  probability of changing edges of node
    :param change_intar_p: probability of changing edge between a pair of clusters
    :return:
    """
    graph = nx.Graph(g)
    f = np.random.rand()
    if f < change_inter_p:  # change in cluster
      j = np.random.randint(0, cluster_num)
      node_sum0 = 0
      for k in range(0, j):
        node_sum0 += clusters[k].number_of_nodes()
      node_num0 = clusters[j].number_of_nodes()
      neig = None
      for n in np.random.choice(node_num0, int(node_num0 * change_node_p), replace=False):
        if np.random.rand() < change_edge:
          neighbor = np.random.choice(graph.neighbors(n + node_sum0), replace=False)
          while neighbor >= node_num0 + node_sum0:
            neighbor = np.random.choice(graph.neighbors(n + node_sum0), replace=False)
          if neig == None:
            neig = n
          else:
            graph.remove_edges_from([(n, neighbor), (neighbor, n)])
            graph.add_edges_from([(n, neig), (neig, n)])
    elif f < change_intar_p and f > change_inter_p:  # change a pair of clusters
      j = np.random.randint(0, cluster_num)
      node_sum0 = 0
      for k in range(0, j):
        node_sum0 += clusters[k].number_of_nodes()
      node_num0 = clusters[j].number_of_nodes()
      node0 = np.random.choice(node_num0, replace=False) + node_sum0
      k = np.random.randint(0, cluster_num)
      node_sum1 = 0
      for j in range(0, k):
        node_sum1 += clusters[j].number_of_nodes()
      node_num1 = clusters[k].number_of_nodes()
      node1 = np.random.choice(node_num1, replace=False) + node_sum1
      c = 0
      while graph.has_edge(node0, node1) and node0 < node_sum0 + node_num0 and node1 < node_num1 + node_sum1:
        if c == 0:  # keep number of edge
          graph.remove_edges_from([(node0, node1), (node1, node0)])
          c += 1
        node0 += 1;
        node1 += 1
      if ~graph.has_edge(node0, node1):
        graph.add_edges_from([(node0, node1), (node1, node0)])
    return graph

  def change_sbm_new(self, g, cluster_num, clusters_, change_inter_p=None, change_node_p=None,
                 change_edge=None, change_intar_p=None):
    """
    :param g: graph will be changed
    :param cluster_num: number of clusters in g
    :param clusters: clusters in g
    :param change_inter_p:  probability of changing in each cluster
    :param change_node_p:  ratio of changing node of cluster
    :param change_edge:  probability of changing edges of node
    :param change_intar_p: probability of changing edge between a pair of clusters
    :return:
    """
    clusters = []
    for cluster in clusters_:
      clusters.append(nx.Graph(cluster))
    graph = nx.Graph(g)
    f = np.random.rand()
    print(f)
    if f < change_inter_p:  # change in cluster
      # 随机选取一个社团，以change_edge 的概率改变它change_node_p 数量的节点间连边
      j = np.random.randint(0, cluster_num)
      node_sum0 = 0
      for k in range(0, j):
        node_sum0 += clusters[k].number_of_nodes()
      cluster = clusters[j]
      node_num0 = cluster.number_of_nodes()
      neig_new = None
      for n in np.random.choice(node_num0, size=int(node_num0 * change_node_p), replace=False):
        node0 = n + node_sum0
        if np.random.rand() < change_edge:
          neig = np.random.choice(list(cluster.neighbors(n)), replace=False)  # in cluster
          neig_old = neig + node_sum0
          if neig_new == None:
            neig_new = node0  # 将变点连接
          else:
            if np.random.rand() < 0.5:
              graph.remove_edges_from([(node0, neig_old), (neig_old, node0)])
              cluster.remove_edges_from([(n, neig), (neig, n)])
            graph.add_edges_from([(node0, neig_new), (neig_new, node0)])
            cluster.add_edges_from([(n, neig_new-node_sum0), (neig_new-node_sum0, n)])
    if (f < change_intar_p):  # change a pair of clusters
      # 随机选取一个社团，改变它change_node_p 数量的节点与其他社团的连边
      j = np.random.randint(0, cluster_num)
      node_sum0 = 0
      for k in range(0, j):
        node_sum0 += clusters[k].number_of_nodes()
      cluster0 = clusters[j]
      node_num0 = cluster0.number_of_nodes()
      node0s = np.random.choice(node_num0, size=int(change_node_p*node_num0),
                               replace=False) + node_sum0
      for node0 in node0s:
        print(node0)
        k = np.random.randint(0, cluster_num)
        node_sum1 = 0
        for j in range(0, k):
          node_sum1 += clusters[j].number_of_nodes()
        cluster1 = clusters[k]
        node_num1 = cluster1.number_of_nodes()
        node1 = np.random.choice(node_num1, replace=False) + node_sum1 # in cluster
        while graph.has_edge(node0, node1):
          node1 = np.random.choice(node_num1, replace=False) + node_sum1
        if ~graph.has_edge(node0, node1):  # with no edge between pair
          graph.add_edges_from([(node0, node1), (node1, node0)])
    return graph, clusters

  def change_edges(self, g, cluster_num, clusters_, change_inter_p=None, change_node_p=None,
                 change_edge=None, change_intar_p=None):
    """
    根据要改变的节点比例，添加干扰边，同时保证不会加大节点对的外部和内部连边的比值
    :param g: graph will be changed
    :param cluster_num: number of clusters in g
    :param clusters: clusters in g
    :param change_inter_p:  probability of changing in each cluster
    :param change_node_p:  ratio of changing node of cluster
    :param change_edge:  probability of changing edges of node
    :param change_intar_p: probability of changing edge between a pair of clusters
    :return:
    """
    clusters = []
    clusters_size = []
    for cluster in clusters_:
      clusters.append(nx.Graph(cluster))
      clusters_size.append(cluster.number_of_nodes())
    graph = nx.Graph(g)
    f = np.random.rand()
    if f < change_inter_p:  # change in cluster
      # 随机选取一个社团，以change_edge 的概率改变它change_node_p 数量的节点间连边
      j = np.random.randint(0, cluster_num)
      node_sum0 = 0
      for k in range(0, j):
        node_sum0 += clusters[k].number_of_nodes()
      cluster = clusters[j]
      node_num0 = cluster.number_of_nodes()
      change_size = int(node_num0 * change_node_p)
      nodes = np.random.choice(node_num0, size=2*change_size, replace=False) + node_sum0
      nodes_index = np.random.choice(nodes, size=(change_size, 2), replace=False)
      for edge in nodes_index:
        node0 = edge[0]
        n = node0-node_sum0
        node1 = edge[1]
        if np.random.rand() < change_edge:
          neig = np.random.choice(list(cluster.neighbors(n)), replace=False)  # in cluster
          neig_old = neig + node_sum0
          if np.random.rand() < 0.5:
            graph.remove_edges_from([(node0, neig_old)])
            cluster.remove_edges_from([(n, neig), (neig, n)])
          graph.add_edges_from([(node0, node1)])
          cluster.add_edges_from([(n, node1-node_sum0)])
    if (f < change_intar_p):  # change a pair of clusters
      node_num0 = np.min(clusters_size)
      change_size = int(change_node_p*node_num0)
      node_sorted = self.__get_degree_ratio(graph, clusters_size, 'sorted')
      node0s = node_sorted[-change_size::]
      for node_r0 in node0s:
        node0 = node_r0[0]
        # 如果节点外部连接与内部连接比值稍大，重新选取
        if self.__get_degree_ratio(graph, clusters_size, 'node', node0) >= self.ratio_threshold:
          continue
        else:
          temp = 0
          for j in range(cluster_num):
            temp_c = clusters[j].number_of_nodes()
            if temp+temp_c > node0:
              break
            else:
              temp += temp_c
          cluster_index0 = j-1
          ks = np.random.randint(0, cluster_num, size=2)
          k = ks[0] if ks[0] != cluster_index0 else ks[1]
          node_sum1 = 0
          for j in range(0, k):
            node_sum1 += clusters[j].number_of_nodes()
          cluster1 = clusters[k]
          node_num1 = cluster1.number_of_nodes()
          node1 = np.random.choice(node_num1, replace=False) + node_sum1  # in cluster
          # 如果两个节点已经有边了，或者，节点外部连接与内部连接比值稍大，重新选取
          while graph.has_edge(node0, node1) or \
                  self.__get_degree_ratio(graph, clusters_size, 'node', node1) >= self.ratio_threshold:
            node1 = np.random.choice(node_num1, replace=False) + node_sum1
          graph.add_edges_from([(node0, node1)])
          if self.__get_degree_ratio(graph, clusters_size, 'node', node0) >= self.ratio_threshold or\
            self.__get_degree_ratio(graph, clusters_size, 'node', node1) >= self.ratio_threshold:
            graph.remove_edges_from([(node0, node1)])
    print(self.__get_degree_ratio(graph, clusters_size))
    return graph, clusters

  def change_cluster(self, g, list_signal, node_num, cluster_num, inter_p):
    graph = nx.Graph()
    clusters = []
    if list_signal:  # node_num is list of the size of cluster, sum(node_num) == size of teh graph
      for i in range(cluster_num):
        clusters.append(self.gen_er(node_num[i], inter_p, 1))
    else:  # node_num is the size of each of cluster in graph
      clusters = self.gen_er(node_num, inter_p, cluster_num)
    for i in range(cluster_num):
      graph = nx.disjoint_union(graph, clusters[i])
    node_sum0 = 0
    for j in range(cluster_num):
        node_num0 = clusters[j].number_of_nodes()
        node_sum1 = 0
        for k in range(cluster_num):
          node_num1 = clusters[k].number_of_nodes()
          if j == k:
            node_sum1 = node_sum1 + node_num1
            continue
          for n in range(node_num0):
            n1 = n+node_sum0
            for n in range(node_num1):
              n2 = n + node_sum1
              if g.has_edge(n1, n2):
               graph.add_edge(n1, n2)
          node_sum1 = node_sum1 + node_num1
        node_sum0 = node_sum0 + node_num0
    return graph, clusters

  def change_cluster_new(self, g, clusters, from_index, to_index, cluster_num, inter_p):
    '''
    change edges between and in clusters
    :param g:
    :param clusters:
    :param from_index:
    :param to_index:
    :param cluster_num:
    :param inter_p:
    :return:
    '''
    graph = nx.Graph(g)
    from_cluster = clusters[from_index]
    to_cluster = clusters[to_index]
    node_sum0 = 0
    node_sum1 = 0
    for i in range(cluster_num):
      if i<from_index:
        node_sum0+=clusters[i].number_of_nodes()
      if i<to_index:
        node_sum1 += clusters[i].number_of_nodes()

    for n0 in from_cluster.nodes():
      node0 = n0 + node_sum0
      for n1 in to_cluster.nodes():
        node1 = n1 + node_sum1
        if np.random.rand()<inter_p:
          if ~graph.has_edge(node0, node1):
            graph.add_edge(node0, node1)

    for n0 in from_cluster.nodes():
      node0 = n0 + node_sum0
      for neig in from_cluster.neighbors(n0):
        node1 = neig + node_sum0
        if np.random.rand() < 1-inter_p:
          if graph.has_edge(node0, node1):
            graph.remove_edge(node0, node1)
            node = np.random.choice(from_cluster.nodes(), replace=False)
            while node == n0:
              node = np.random.choice(from_cluster.nodes(), replace=False)
            graph.add_edge(node1, node + node_sum0)
    for n0 in to_cluster.nodes():
      node0 = n0 + node_sum1
      for neig in to_cluster.neighbors(n0):
        node1 = neig + node_sum1
        if np.random.rand() < 1 - inter_p:
          if graph.has_edge(node0, node1):
            graph.remove_edge(node0, node1)
            node = np.random.choice(to_cluster.nodes(), replace=False)
            while node == n0:
              node = np.random.choice(to_cluster.nodes(), replace=False)
            graph.add_edge(node1, node + node_sum1)

    return graph, clusters

  def change_cluster_stable(self, g, clusters, from_index, to_index, cluster_num, inter_p):
    '''
    change edges between clusters
    :param g:
    :param clusters:
    :param from_index:
    :param to_index:
    :param cluster_num:
    :param inter_p:
    :return:
    '''
    graph = nx.Graph(g)
    from_cluster = clusters[from_index]
    to_cluster = clusters[to_index]
    node_sum0 = 0
    node_sum1 = 0
    for i in range(cluster_num):
      if i<from_index:
        node_sum0 += clusters[i].number_of_nodes()
      if i<to_index:
        node_sum1 += clusters[i].number_of_nodes()

    for n0 in from_cluster.nodes():
      node0 = n0 + node_sum0
      for n1 in to_cluster.nodes():
        node1 = n1 + node_sum1
        if np.random.rand()<inter_p:
          if ~graph.has_edge(node0, node1):
            graph.add_edge(node0, node1)
    return graph, clusters

  """
    generate E-R model
    node_num: the number of nodes in each network
    inter_p: probility of each edge
    graph_num: the number of networks
  """
  def gen_er(self, node_num, inter_p, graph_num=1, flage=0, repeat=1):
    graph = nx.erdos_renyi_graph(node_num, inter_p)
    graphs = [graph]
    for i in range(1, graph_num):
      graphs.append(self.nullModel(graph, flage=flage, repeat=repeat))
    if graph_num == 1:
      return graphs[0]
    else:
      return graphs
  """
    generate regular graph model
    node_num: the number of nodes in each network
    degree_num: degree of each node
    graph_num: the number of networks
  """
  def gen_rg(self, node_num, degree_num, graph_num=1, flage=0, repeat=1):
    graph = nx.random_graphs.random_regular_graph(degree_num, node_num)
    graphs = [graph]
    for i in range(1, graph_num):
      graphs.append(self.nullModel(graph, flage=flage, repeat=repeat))
    if graph_num == 1:
      return graphs[0]
    else:
      return graphs
  """
    generate BA model
    node_num: the number of nodes in each network
    join_edge_num: the number of node for each iteration
    graph_num: the number of networks
  """
  def gen_ba(self, node_num, join_edge_num=1, graph_num=1, flage=0, repeat=1):
    graph = nx.barabasi_albert_graph(node_num, join_edge_num)
    graphs = [graph]
    for i in range(1, graph_num):
      graphs.append(self.nullModel(graph, flage=flage, repeat=repeat))
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
  def gen_ws(self, node_num, neigh_num, redge_p, graph_num=1, flage=0, repeat=1):
    graph = nx.watts_strogatz_graph(node_num, neigh_num, redge_p)
    graphs = [graph]
    for i in range(1, graph_num):
      graphs.append(self.nullModel(graph, flage=flage, repeat=repeat))
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
      # print(graph.nodes(), edge_num)
      # number of phi is equal to node_num
      queue = []; tphi = phi
      while len(graph.edges()) < edge_num:
        if len(queue) == 0:
          node0 = np.random.choice(node_num, size=1, replace=False, p=tphi)[0]  # sample
        else:
          node0 = queue[0]    # pop for queue
          queue = self.__delete_index(queue,0)
        node1 = np.random.choice(node_num, size=1, replace=False, p=tphi)[0]   # sample
        #print(node0, node1)
        if node0 == node1:
          continue
        if (node0, node1) in graph.edges():  # it needs to repeat sample
          queue.append(node0)
          queue.append(node1)
        else:
          # print(node0, node1)
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
  def gen_tcl(self, node_num, edge_num, phi, beta, graph_num=1, iteration=5):
    graphs = self.gen_cl(node_num, edge_num, phi, graph_num)
    # print(graphs)
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
          # print(graph)
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
    change_inter_p: change in one cluster
    change_node_p: number of changed node
    change_edge: change neighbor
    change_intar_p: change edge between clusters
  """
  def gen_sbm(self, list_signal, node_num, cluster_num, inter_p, outer_p):
    # inter_p >> outer_p
    clusters = []; nn = node_num
    if list_signal == 0:   # node_num is list of the size of cluster, sum(node_num) == size of teh graph
      node_num = [nn for i in range(cluster_num)]
    for j in range(cluster_num):
      clusters.append(self.gen_er(node_num[j], inter_p, 1))

    graph = nx.Graph()
    for j in range(cluster_num):
      graph = nx.disjoint_union(graph, clusters[j])
      # print(len(clusters), graph.number_of_nodes())
    node_sum0 = 0
    for j in range(cluster_num):
        node_num0 = clusters[j].number_of_nodes()
        node_sum1 = 0
        for k in range(cluster_num):
          node_num1 = clusters[k].number_of_nodes()
          if j == k:
            node_sum1 = node_sum1 + node_num1
            continue
          for n in range(node_num0):
            if np.random.rand() < outer_p:
              target = np.random.choice(node_num1) + node_sum1
              # print(target, n+node_sum0)
              graph.add_edges_from([(n+node_sum0, target), (target, n+node_sum0)])
          node_sum1 = node_sum1 + node_num1
        node_sum0 = node_sum0 + node_num0
    return graph, clusters

  def gen_sbm_limit(self, list_signal, node_num, cluster_num, inter_p, outer_p):
    """
    保证社团间的连边不会使节点的内外部连边数相近
    :param list_signal: if 0, node_num is number of each clusters, otherwise node_num is a list of number of clusters
    :param node_num: a size of each clusters or a list of the size of clusters
    :param cluster_num: the number of clusters
    :param inter_p:
    :param outer_p:
    :return: graph and clusters
    """
    # inter_p >> outer_p
    clusters = []; nn = node_num
    if list_signal == 0:   # node_num is list of the size of cluster, sum(node_num) == size of teh graph
      node_num = [nn for i in range(cluster_num)]
    for j in range(cluster_num):
      clusters.append(self.gen_er(node_num[j], inter_p, 1))

    graph = nx.Graph()
    for j in range(cluster_num):
      graph = nx.disjoint_union(graph, clusters[j])
      # print(len(clusters), graph.number_of_nodes())
    node_sum0 = 0
    for j in range(cluster_num):
        node_num0 = clusters[j].number_of_nodes()
        node_sum1 = 0
        for k in range(cluster_num):
          node_num1 = clusters[k].number_of_nodes()
          if j == k:
            node_sum1 = node_sum1 + node_num1
            continue
          for n in range(node_num0):
            source = n+node_sum0
            # 如果节点的外部内部边数较近，重选
            if self.__get_degree_ratio(graph, node_num, 'node', source) >= self.ratio_threshold:
              continue
            if np.random.rand() < outer_p:
              target = np.random.choice(node_num1) + node_sum1
              # print(target, n+node_sum0)
              graph.add_edges_from([(source, target)])
              # 如果target 的外部内部边数较近，重选
              if self.__get_degree_ratio(graph, node_num, 'node', source) >= self.ratio_threshold:
                graph.remove_edge(source, target)
                continue
              while self.__get_degree_ratio(graph, node_num, 'node', target) >= self.ratio_threshold:
                graph.remove_edge(source, target)
                target = np.random.choice(node_num1) + node_sum1
                graph.add_edges_from([(source, target)])

          node_sum1 = node_sum1 + node_num1
        node_sum0 = node_sum0 + node_num0
    print(self.__get_degree_ratio(graph, node_num))
    return graph, clusters

  def gen_sbm_limit_node(self, list_signal, node_num, cluster_num, inter_p, outer_p, nodes):
    """
    使网络中一些节点 nodes 的外部连边数 与 外部连边数 相近
    :param list_signal: if 0, node_num is number of each clusters, otherwise node_num is a list of number of clusters
    :param node_num: a size of each clusters or a list of the size of clusters
    :param cluster_num: the number of clusters
    :param inter_p:
    :param outer_p:
    :param nodes: a list of node
    :return: graph and clusters
    """
    # inter_p >> outer_p
    clusters = [];
    nn = node_num
    if list_signal == 0:  # node_num is list of the size of cluster, sum(node_num) == size of teh graph
      node_num = [nn for i in range(cluster_num)]
    for j in range(cluster_num):
      clusters.append(self.gen_er(node_num[j], inter_p, 1))

    graph = nx.Graph()
    for j in range(cluster_num):
      graph = nx.disjoint_union(graph, clusters[j])
      # print(len(clusters), graph.number_of_nodes())
    node_sum0 = 0
    for j in range(cluster_num):
      node_num0 = clusters[j].number_of_nodes()
      node_sum1 = 0
      for k in range(cluster_num):
        node_num1 = clusters[k].number_of_nodes()
        if j == k:
          node_sum1 = node_sum1 + node_num1
          continue
        for n in range(node_num0):
          source = n + node_sum0
          # 如果节点的外部内部边数较近，重选
          if self.__get_degree_ratio(graph, node_num, 'node', source) >= self.ratio_threshold:
            continue
          if np.random.rand() < outer_p:
            target = np.random.choice(node_num1) + node_sum1
            # print(target, n+node_sum0)
            graph.add_edges_from([(source, target)])
            # 如果target 的外部内部边数较近，重选
            if self.__get_degree_ratio(graph, node_num, 'node', source) >= self.ratio_threshold:
              graph.remove_edge(source, target)
              continue
            while self.__get_degree_ratio(graph, node_num, 'node', target) >= self.ratio_threshold:
              graph.remove_edge(source, target)
              target = np.random.choice(node_num1) + node_sum1
              graph.add_edges_from([(source, target)])

        node_sum1 = node_sum1 + node_num1
      node_sum0 = node_sum0 + node_num0
    node_sum = node_sum0 - node_num0
    # print(node_num0, node_sum)
    # node_sum = 0
    node_num0 = node_num[-1]
    for node in nodes:    # 增强与最后一个社团的连接
      print(node)
      while self.__get_degree_ratio(graph, node_num, 'node', node) < self.ratio_upper:

        target = np.random.choice(node_num0) + node_sum
        # print(target)
        graph.add_edges_from([(node, target)])
        if target not in nodes and \
                self.__get_degree_ratio(graph, node_num, 'node', target) >= self.ratio_threshold:
          graph.remove_edges_from([(node, target)])
    print(self.__get_degree_ratio(graph, node_num))
    return graph, clusters


  """
    generate Stochastic-Block model
    list_signal: the signal. if it is diffrent among the number of nodes for each clusters(communities), list_signal is 1, otherwise it's 0.
    node_num: the number of nodes in each network
    cluster_num: number of clusters in each network
    inter_p: probility for edge in the same cluster
    outer_p: probility of edge between teh different clusters
    graph_num: the number of networks
  """
  def gen_sbm_cl(self, list_signal, node_num, cluster_num, edge_num, phi, outer_p, graph_num = 1):
    # inter_p >> outer_p
    clusters = []
    if list_signal:   # node_num is list of the size of cluster, sum(node_num) == size of teh graph
      for j in range(cluster_num):
        clusters.append(self.gen_cl(node_num, edge_num, phi, 1))
    else:  # node_num is the size of each of cluster in graph
      clusters = self.gen_cl(node_num, edge_num, phi, cluster_num)
    graphs = []
    for i in range(graph_num):
      graph = nx.Graph()
      for j in range(cluster_num):
        graph = nx.disjoint_union(graph,clusters[j])
      node_sum0 = 0
      for j in range(cluster_num):
        node_sum1 = 0
        for k in range(cluster_num):
          if j == k:
            continue
          node_num0 = clusters[j].number_of_nodes()
          node_num1 = clusters[k].number_of_nodes()
          for n in range(node_num0):
            if np.random.rand() < outer_p:
              target = np.random.choice(node_num1) + node_sum1
              graph.add_edges_from([(n+node_sum0, target), (target, n+node_sum0)])
          node_sum1 = node_sum1 + node_num1
        node_sum0 = node_sum0 + node_num0
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
  #gen.draw_graph(gen.gen_sbm(0, 50, 4, 0.5, 0.2))
  #gen.draw_graph(gen.gen_sbm(1, [2,8], 2, 0.5, 0.1))
  #gen.draw_graph(gen.gen_tcl(10, 20, [0.1,0.05,0.2,0.1,0.1,0.05,0.1,0.1,0.1,0.1], 0.1))
  #gen.draw_graph(gen.gen_cl(10, 20, [0.1,0.1,0.11,0.1,0.11,0.09,0.1,0.1,0.09,0.1]))
  #gen.draw_graph(gen.gen_sbm_cl(0, 10, 2, 15, [0.1,0.1,0.11,0.1,0.11,0.09,0.1,0.1,0.09,0.1], 0.1))
  import scipy.sparse as scsp
  import scipy.io as scio

  # list_signal, node_num, cluster_num, inter_p, outer_p, graph_num = 1,
  #               change_inter_p=None, change_node_p=None, change_edge=None, change_intar_p=None
  # graphs1 = gen.gen_sbm(0, 25, 4, 0.2, 0.02, 2, 0.2, 0.3, 0.6, 0.1)
  # graphs2 = gen.gen_sbm(0, 25, 4, 0.4, 0.1, 2, 0.2, 0.3, 0.6, 0.1)
  # graphs3 = gen.gen_sbm(0, 25, 4, 0.5, 0.1, 2, 0.2, 0.3, 0.6, 0.1)
  # graphs3 = gen.gen_sbm(0, 25, 4, 0.6, 0.2, 2, 0.2, 0.3, 0.6, 0.1)
  # graphs = graphs1 + graphs2 + graphs3
  # gen.draw_graph(graphs)
  # graph2 = gen.gen_er(100, 0.3, graph_num=2, flage=2)
  # graph2 = gen.gen_ws(100, 2, 0.4, graph_num=2, flage=2)

  # node_num = 100; graph_num = 15
  # graph2 = gen.gen_ba(node_num, 2, graph_num=graph_num, flage=2, repeat=max(int(0.05 * node_num), 1))
  # graph3 = gen.gen_ba(node_num, 2, graph_num=graph_num, flage=2, repeat=max(int(0.05 * node_num), 1))
  # # graph2 = gen.gen_cl(100, 150, 0.03, graph_num=2)
  # graphs = graph2 + graph3
  # file = 'sample_30g_5r.mtx'

  '''
    1000个节点，9个网络，不断融合0，1的社团
  node_num = 1000
  graph1, cluster = gen.gen_sbm(0, 500, 2, 0.1, 0.01)
  graph2, cluster = gen.migrate_c0_c1(graph1, cluster)
  graph3, cluster = gen.migrate_c0_c1(graph2, cluster)
  graph4, cluster = gen.migrate_c0_c1(graph3, cluster)
  graph5, cluster = gen.migrate_c0_c1(graph4, cluster)
  graphs = [graph1,graph2,graph3,graph4,graph5]
  graph2, cluster = gen.migrate_c0_c1(graph5, cluster)
  graphs.append(graph2)
  graph3, cluster = gen.migrate_c0_c1(graph2, cluster)
  graphs.append(graph3)
  graph4, cluster = gen.migrate_c0_c1(graph3, cluster)
  graphs.append(graph4)
  graph5, cluster = gen.migrate_c0_c1(graph4, cluster)
  graphs.append(graph5)
  file = 'sbm_10g_1000n_0.1i_0.01o.mtx'
  '''
  """
    100个节点，24个网络， 7个变点
  node_num = 100; cluster_node_num = 25; cluster_num = 4

  graphs = []

  # 参数顺序含义：社团平分节点数， 每个社团25个节点， 4个社团，0.2的内部连接， 0.05的外部连接，
  # 0
  graphs1, clusters1 = gen.gen_sbm(0, cluster_node_num, cluster_num, 0.2, 0.05)
  graphs.append(graphs1)
  # 每个图0.01的概率发生演变， 改变时会改变1%的点的连边情况，
  # 并根据10%的概率改变他们与邻居的连边，外部连边不发生改变
  # 1
  for i in range(6):
    graph1 = gen.change_sbm(graphs1, cluster_num, clusters1, 0.01, 0.01, 0.1, 0)
    graphs.append(graph1)

  # 以同样生成社团的标准生成新的社团，并保持社团间的连边不变
  # 参数顺序含义：社团平分节点数， 每个社团25个节点，4个社团，0.2的内部连接
  # a 3
  graphs2, clusters2 = gen.change_cluster(graphs1, 0, cluster_node_num, cluster_num, 0.2)
  graphs.append(graphs2)
  # gen.draw_graph(graphs2)
  # 每个图0.01的概率发生演变， 改变时会改变1%的点的连边情况，
  # 并根据10%的概率改变他们与邻居的连边，外部连边不发生改变
  for i in range(4):
    graph1 = gen.change_sbm(graphs2, cluster_num, clusters2, 0.01, 0.01, 0.1, 0)
    graphs.append(graph1)

  # 和初始时相同，每个图0.01的概率发生演变， 改变时会改变1%的点的连边情况，
  # 并根据10%的概率改变他们与邻居的连边，外部连边不发生改变
  for i in range(3):
    graph1 = gen.change_sbm(graphs1, cluster_num, clusters1, 0.01, 0.01, 0.1, 0)
    graphs.append(graph1)

  # 从0号社团移动10个点到1号社团，同时改变连边
  # a 4
  graphs3, clusters3 = gen.migrate_c0_c1(graphs1, clusters1)
  graphs.append(graphs3)
  # 从0号社团移动10个点到1号社团，同时改变连边
  graphs4, clusters4 = gen.migrate_c0_c1(graphs3, clusters3)
  graphs.append(graphs4)

  # 和初始时相同，每个图0.01的概率发生演变， 改变时会改变1%的点的连边情况，
  # 并根据10%的概率改变他们与邻居的连边，外部连边不发生改变
  for i in range(3):
    graph1 = gen.change_sbm(graphs1, cluster_num, clusters1, 0.01, 0.01, 0.1, 0)
    graphs.append(graph1)

  # 参数顺序含义：社团节点数固定50，25，25， 一共100个节点， 3个社团，0.2的内部连接， 0.05的外部连接
  # a 5
  graphs5, clusters5 = gen.gen_sbm(1, [50, 25, 25], 3, 0.2, 0.05)
  graphs.append(graphs5)
  # 和初始时相同，每个图0.01的概率发生演变， 改变时会改变1%的点的连边情况，
  # 并根据10%的概率改变他们与邻居的连边，外部连边不发生改变
  for i in range(3):
    graph1 = gen.change_sbm(graphs1, cluster_num, clusters1, 0.01, 0.01, 0.1, 0)
    graphs.append(graph1)
  # gen.draw_graph(graphs3)

  file = 'simulant_24g_100n_a3a4a5.mtx'
  print(len(graphs))
  # gen.draw_graph(graphs)
  """
  '''
    write networks one by one
  mats = []
  for graph in graphs:
    adj = nx.adjacency_matrix(graph)
    adj.shape = 1, node_num*node_num
    mats.append(np.array(adj)[0])
  scio.mmwrite(file, scsp.csr_matrix(mats))
  '''

  # node_nums = [50, 100, 500, 1000]
  #
  # net_nums = [10, 30, 50]
  #
  # for net_num in net_nums:
  #   for node_num in node_nums:
  #     graphs = []
  #     anomaly = np.random.choice(net_num, size=int(0.4*net_num), replace=False)
  #     anomaly = sorted(anomaly)
  #     path = 'SBM_'+str(net_num)+'g_'+str(node_num)+'n'
  #     file = 'SBM_'+str(net_num)+'g_'+str(node_num)+'n'+'.mtx'
  #     # if ~os.path.exists(path):
  #     #   os.makedirs(path)
  #     # np.savetxt(os.path.join(path, 'anomaly.txt'), anomaly)
  #     t = 0
  #     for a in anomaly:
  #       if a == 0:
  #         continue
  #       graph1, cluster = gen.gen_sbm(0, int(node_num/2), 2, max(10.0/node_num, 0.1), max(1.0/node_num, 0.01))
  #       graphs.append(graph1)
  #       for i in range(t+1, a):
  #         graph = gen.change_sbm(graph1, 2, cluster, 0.01, 0.01, 0.01, 0)
  #         graphs.append(graph)
  #       t = a
  #     if t <= net_num-1:
  #       graph1, cluster = gen.gen_sbm(0, int(node_num/2), 2, max(10.0/node_num, 0.1), max(1.0/node_num, 0.01))
  #       graphs.append(graph1)
  #       for i in range(t+1, net_num):
  #         graph = gen.change_sbm(graph1, 2, cluster, 0.01, 0.01, 0.01, 0)
  #         graphs.append(graph)
  #     print(net_num, len(graphs))
  #     if 4 in anomaly or 5 in anomaly:
  #       gen.draw_graph(graphs[4:6])
  #     mats = []
  #     for graph in graphs:
  #       adj = nx.adjacency_matrix(graph)
  #       adj.shape = 1, node_num * node_num
  #       mats.append(np.array(adj)[0])
  #     # scio.mmwrite(os.path.join(path, file), scsp.csr_matrix(mats))

  """
  社团间连接概率增加
  
  inter_p = 0.4
  mats = []
  for i in [0.01, 0.05, 0.1, 0.15, 0.2]:
    graph, clusters = gen.gen_sbm(0, [50, 25, 25], 3, inter_p, i)
    adj = nx.adjacency_matrix(graph).todense()
    adj.shape = 1, 100 * 100
    mats.append(np.array(adj)[0])
  # print(mats.shape)
  scio.mmwrite('outerP', scsp.csr_matrix(mats))
  """
  '''
  2个社团：合并，分裂
  
  node_num = 100
  graph0, cluster = gen.gen_sbm(0, 100, 1, 0.2, 0.05)
  # 分裂
  graph1, cluster = gen.gen_sbm(1, [60, 40], 2, 0.4, 0.1)
  # 合并
  graph2, cluster = gen.migrate_c0_c1(graph1, cluster, size=20)
  mats = []
  adj = nx.adjacency_matrix(graph0).todense()
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])
  # 分裂
  adj = nx.adjacency_matrix(graph1).todense()
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])
  # 合并
  adj = nx.adjacency_matrix(graph2).todense()
  print(adj.shape)
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])
  # 保存
  scio.mmwrite('split_migrate.mtx', scsp.csr_matrix(mats))
  '''
  """
  三个社团，独立扰动，连续扰动，独立融合，连续融合
  
  inter_p = 0.4
  mats = []
  graphs = []
  # [0.01, 0.05, 0.1, 0.15, 0.2]:
  graph, clusters = gen.gen_sbm(1, [40, 30, 30], 3, inter_p, 0.1)
  graphs.append(graph)
  graph0 = copy.copy(graph)
  clusters0 = copy.copy(clusters)
  adj = nx.adjacency_matrix(graph).todense()
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])
  # 独立扰动 1、2、3
  t = 0
  while t < 3:
    t += 1
    graph_temp, _ = gen.change_sbm_new(graph, 3, clusters,
                                     change_inter_p=0.1,
                                     change_node_p=0.2,
                                     change_edge=0.5,
                                     change_intar_p=0.3)
    graphs.append(graph_temp)
    adj = nx.adjacency_matrix(graph_temp).todense()
    adj.shape = 1, 100 * 100
    mats.append(np.array(adj)[0])

  # 持续演变 4、5、6
  t = 0
  while t < 3:
    t += 1
    graph, clusters = gen.change_sbm_new(graph, 3, clusters,
                                     change_inter_p=0.01,
                                     change_node_p=0.2,
                                     change_edge=0.5,
                                     change_intar_p=0.2)
    graphs.append(graph)
    adj = nx.adjacency_matrix(graph).todense()
    adj.shape = 1, 100 * 100
    mats.append(np.array(adj)[0])
  # [40,30,30]
  # 独立融合社团 2 个节点 7、8、9
  t = 0
  while t < 3:
    t += 1
    graph_temp, _ = gen.migrate_c0_c1_new(graph0, clusters0, 2)
    graphs.append(graph_temp)
    adj = nx.adjacency_matrix(graph_temp).todense()
    adj.shape = 1, 100 * 100
    mats.append(np.array(adj)[0])
  graph = graph0
  clusters = clusters0
  t = 0
  # 连续融合社团，每次13个 10、11、12
  while t < 3:
    t += 1
    graph, clusters = gen.migrate_c0_c1_new(graph, clusters, 13)
    graphs.append(graph)
    adj = nx.adjacency_matrix(graph).todense()
    adj.shape = 1, 100 * 100
    mats.append(np.array(adj)[0])

  mats = np.array(mats)
  print(mats.shape)
  # scio.mmwrite('combination', scsp.csr_matrix(mats))
  gen.draw_graph(graphs)
  """
  '''
  inter_p = 0.4
  cluster_num = 4
  mats = []
  graphs = []
  graph, clusters = gen.gen_sbm(0, 25, cluster_num, inter_p, 0.2)
  graphs.append(graph)
  adj = nx.adjacency_matrix(graph).todense()
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])
  # 独立扰动 1、2、3
  t = 0
  while t < 3:
    t += 1
    graph_temp, _ = gen.change_sbm_new(graph, cluster_num, clusters,
                                       change_inter_p=1,
                                       change_node_p=0.2,
                                       change_edge=0.5,
                                       change_intar_p=0.6)
    graphs.append(graph_temp)
    adj = nx.adjacency_matrix(graph_temp).todense()
    adj.shape = 1, 100 * 100
    mats.append(np.array(adj)[0])
  t = 0
  while t < 3:
    t += 1
    graph_temp, _ = gen.migrate_c0_c1_new(graph, clusters, 2)
    graphs.append(graph_temp)
    adj = nx.adjacency_matrix(graph_temp).todense()
    adj.shape = 1, 100 * 100
    mats.append(np.array(adj)[0])

  graph_temp, _ = gen.change_cluster_new(graph, clusters, 0, 1, 3, inter_p)
  graphs.append(graph_temp)
  adj = nx.adjacency_matrix(graph_temp).todense()
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])

  graph_temp, _ = gen.change_cluster_stable(graph, clusters, 0, 1, 3, inter_p)
  graphs.append(graph_temp)
  adj = nx.adjacency_matrix(graph_temp).todense()
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])

  mats = np.array(mats)
  print(mats.shape)
  scio.mmwrite('G:\CodeSet\workspace\HGCN\sinmulateFordraw\draw.mtx', scsp.csr_matrix(mats))
  gen.draw_graph(graphs)
  '''
  inter_p = 0.4
  intar_p = 0.2
  cluster_num = 4
  mats = []
  graphs = []

  nodes_migr = [11, 24]
  graph, clusters = gen.gen_sbm_limit(0, 25, cluster_num, inter_p, intar_p)
  # save
  graphs.append(graph)
  adj = nx.adjacency_matrix(graph).todense()
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])
  t = 0
  while t < 8:
    t += 1
    # 社团内部 100%变化，选20%的节点，50%改变其连边，50%删除随机边，增加随即边
    # 社团外部 60%变化，选内部20%的节点，随机连接外部社团
    graph_temp, _ = gen.change_edges(graph, cluster_num, clusters, change_inter_p=1,
                                     change_node_p=0.2, change_edge=0.5, change_intar_p=0.6)
    print({node: list(graph_temp.neighbors(node)) for node in nodes_migr})
    graphs.append(graph_temp)
    adj = nx.adjacency_matrix(graph_temp).todense()
    adj.shape = 1, 100 * 100
    mats.append(np.array(adj)[0])
  # gen.draw_graph(graphs)
  graph_temp, _ = gen.migrate_clusters(graph, clusters, 0, 1, 2, nodes_migr)
  print({node: list(graph_temp.neighbors(node)) for node in nodes_migr})
  graphs.append(graph_temp)
  adj = nx.adjacency_matrix(graph_temp).todense()
  adj.shape = 1, 100 * 100
  mats.append(np.array(adj)[0])

  scio.mmwrite('G:\CodeSet\workspace\HGCN\sinmulateFordraw\\figure7.mtx', scsp.csr_matrix(mats))
