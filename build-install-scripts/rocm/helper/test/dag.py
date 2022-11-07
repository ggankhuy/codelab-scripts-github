from matplotlib import pyplot as plt
import networkx as nx
graph = nx.DiGraph()
graph.add_edges_from([("root", "a"), ("a", "b"), ("a", "e"), ("b", "c"), ("b", "d"), ("d", "e")])
print(nx.shortest_path(graph, 'root', 'e'))
#print(nx.dag_longest_path(graph, 'root', 'e'))
print(list(nx.topological_sort(graph)))

plt.tight_layout()
nx.draw_networkx(graph, arrows=True)
plt.savefig("graph.png", format="PNG")
# tell matplotlib you're done with the plot: https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
plt.clf()
