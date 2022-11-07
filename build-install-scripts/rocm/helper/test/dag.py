import networkx as nx
graph = nx.DiGraph()
graph.add_edges_from([("root", "a"), ("a", "b"), ("a", "e"), ("b", "c"), ("b", "d"), ("d", "e")])
print(nx.shortest_path(graph, 'root', 'e'))
#print(nx.dag_longest_path(graph, 'root', 'e'))
print(list(nx.topological_sort(graph)))
