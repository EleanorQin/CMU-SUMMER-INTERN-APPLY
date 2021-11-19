import time, sys, copy
from pyspark import SparkContext, SparkConf
from itertools import combinations
from math import inf


class Graph(object):
    def __init__(self):
        self.adj_dict = {}
        self.original_adj_dict = {}
        self.betweenness_list = []
        self.edge_cnt = 0

    def nodes(self):
        return self.adj_dict.keys()

    def add_nodes(self, nodes: set) -> None:
        for node in nodes:
            if node not in self.nodes():
                self.adj_dict[node] = []

    def add_edges(self, edges: set) -> None:
        for u, v in edges:
            if v not in self.adj_dict[u]:
                self.adj_dict[u].append(v)
            if u not in self.adj_dict[v]:
                self.adj_dict[v].append(u)
            self.edge_cnt += 1

    def add_graph(self, adjacency_dict):
        self.adj_dict = adjacency_dict

    def A(self, i, j):
        if j in self.original_adj_dict[i]:
            return 1
        else:
            return 0

    def back_up_graph(self):
        self.original_adj_dict = copy.deepcopy(self.adj_dict)

    def bfs(self, root=None) -> dict:
        queue = [root]
        visited = dict()
        # level, parents
        visited[root] = (0, [])
        while root and len(queue) > 0:
            v = queue.pop(0)
            adj_nodes = self.adj_dict[v]
            for n in adj_nodes:
                if n not in visited:
                    queue.append(n)
                    visited[n] = (visited[v][0] + 1, [v])
                elif visited[v][0] == visited[n][0] - 1:
                    visited[n][1].append(v)
        return visited

    def GN2(self):
        between = {}
        for vertex in self.nodes():
            shortest_path = {}
            level = {}
            parentnode = []
            bfstree = self.bfs(vertex)
            for v, l in sorted(bfstree.items(), key=lambda pair: -pair[1][0]):
                level.setdefault(l[0], []).append((v, l[1]))
                parentnode.extend(l[1])

            for l in range(0, len(level)):
                for (child, parents) in level[l]:
                    if not parents:
                        shortest_path[child] = 1
                    else:
                        shortest_path[child] = sum([shortest_path[parent] for parent in parents])
            # print(shortest_path)

            parentnode = set(parentnode)
            nodeweight = {}
            for l, nodes in level.items():
                for (child, parents) in nodes:
                    # leaf nodes
                    if child not in parentnode:
                        nodeweight[child] = 1
                    else:
                        nodeweight[child] += 1

                    # edge betweenness
                    allparents = sum([shortest_path[parent] for parent in parents])
                    for parent in parents:
                        try:
                            nodeweight[parent] += nodeweight[child] * float(shortest_path[parent]) / allparents
                        except:
                            nodeweight[parent] = nodeweight[child] * float(shortest_path[parent]) / allparents

                        if parent < child:
                            v1, v2 = parent, child
                        else:
                            v1, v2 = child, parent

                        try:
                            between[(v1, v2)] += nodeweight[child] * float(shortest_path[parent]) / allparents
                        except:
                            between[(v1, v2)] = nodeweight[child] * float(shortest_path[parent]) / allparents
        return {edge: value / 2 for (edge, value) in between.items()}

    def Girvan_Newman(self):
        betweenness = {}
        for root in self.nodes():
            bfs_result = self.bfs(root)
            # print(bfs_result)
            shortest_path = dict()
            top_down_bfs = sorted(bfs_result.items(), key=lambda x: x[1][0])
            bottom_up_bfs = sorted(bfs_result.items(), key=lambda x: x[1][0], reverse=True)
            for node, _ in top_down_bfs:
                if node == root:
                    shortest_path[node] = 1
                else:
                    parents_cnts = [shortest_path[p] for p in _[1]]
                    shortest_path[node] = sum(parents_cnts)
            max_level = top_down_bfs[-1][1][0]
            if max_level == 0:
                continue
            # print(shortest_path)
            credit = {}
            for level in range(max_level, 0, -1):
                for node, _ in bottom_up_bfs[:-1]:
                    cur_level, parents = _
                    if cur_level == level:
                        if node not in credit:
                            credit[node] = 1
                        parents_sum = sum([shortest_path[p] for p in parents])
                        nc = credit[node]
                        for p in parents:
                            edge = tuple(sorted([node, p]))
                            edge_credit = nc*shortest_path[p]/parents_sum
                            credit[p] = credit.get(p, 1) + edge_credit
                            betweenness[edge] = betweenness.get(edge, 0.0) + edge_credit

        for k in betweenness:
            betweenness[k] = betweenness[k]/2

        sorted_btw = sorted(betweenness.items(), key=lambda x: (-x[1], x[0]))
        self.betweenness_list = sorted_btw

        return sorted_btw

    def Modularity(self, groups):
        m = self.edge_cnt
        degrees = {node: len(_) for node, _ in self.original_adj_dict.items()}
        Q = 0.0
        for s in groups:
            for i in s:
                for j in s:
                    Q += (self.A(i, j) - (degrees[i] * degrees[j] / (2 * m)))
        return Q/(2*m)

    def community_detection(self):
        self.back_up_graph()
        max_Q = -inf
        best_commnity = None
        while len(self.betweenness_list) > 0:
            group = [self.betweenness_list[0]]
            top_btw = self.betweenness_list[0][1]
            if len(self.betweenness_list) > 1:
                group += [_ for _ in self.betweenness_list[1:] if _[1] == top_btw]

            for elem in group:
                u, v = elem[0]
                self.adj_dict[u].remove(v)
                self.adj_dict[v].remove(u)
                self.betweenness_list.remove(elem)

            community = []
            cur_nodes = list(self.nodes())
            while len(cur_nodes) > 0:
                bfs_result = g.bfs(cur_nodes[0])
                added_nodes = list(bfs_result.keys())
                community.append(added_nodes)
                for an in added_nodes:
                    cur_nodes.remove(an)

            Q = g.Modularity(community)
            if Q > max_Q:
                max_Q = Q
                best_commnity = community
            self.Girvan_Newman()
        # print(max_Q)
        return best_commnity

if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) != 5:
        print("Not a valid input format!")
        exit(0)

    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    sc = SparkContext(appName='dsci553Zhenqinhw4')
    sc.setLogLevel("ERROR")
    # user id,business id
    text_rdd = sc.textFile(input_file_path).filter(lambda row: row != 'user_id,business_id') \
        .map(lambda row: row.split(',')).map(lambda row: (row[0], row[1]))
    # (userid,(bid1,bid2,bid3)

    def set_add(x, v):
        x.add(v)
        return x
    user_biz_set_rdd = text_rdd.combineByKey(
        lambda x: {x},
        lambda U, v: set_add(U, v),
        lambda U1, U2: U1.union(U2)).collectAsMap()
    # print(len(user_biz_set_rdd))
    users = text_rdd.map(lambda x: x[0]).distinct().collect()
    # print(user_rdd.count())

    edge_user_pairs = set()
    for user_pair in combinations(users, 2):
        u1_set = user_biz_set_rdd[user_pair[0]]
        u2_set = user_biz_set_rdd[user_pair[1]]
        if len(u1_set & u2_set) >= filter_threshold:
            edge_user_pairs.add(tuple(sorted(user_pair)))

    node = set()
    for pairs in edge_user_pairs:
        node.add(pairs[0])
        node.add(pairs[1])

    g = Graph()
    g.add_nodes(node)
    g.add_edges(edge_user_pairs)

    betweenness = g.Girvan_Newman()
    with open(betweenness_output_file_path, 'w') as f:
        for edge, value in betweenness:
            f.write(str(edge) + ', ' + str(round(value, 5)) + '\n')

    community = g.community_detection()
    tmp = []
    for c in community:
        tmp.append(sorted(c))

    with open(community_output_file_path, 'w') as f:
        for line in sorted(tmp, key=lambda x: (len(x), x[0])):
            f.write(str(line)[1:-1] + '\n')

    end = time.time()
    print("Duration:", end - start)