from matplotlib import pyplot as plt
import networkx as nx
graph = nx.DiGraph()
graph.add_edges_from([("root", "a"), ("a", "b"), ("a", "e"), ("b", "c"), ("b", "d"), ("d", "e")])
print(nx.shortest_path(graph, 'root', 'e'))
#print(nx.dag_longest_path(graph, 'root', 'e'))
print(list(nx.topological_sort(graph)))

'''
plt.tight_layout()
nx.draw_networkx(graph, arrows=True)
plt.savefig("graph.png", format="PNG")
# tell matplotlib you're done with the plot: https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
plt.clf()
'''
all_pred=[]
indent=""
def recur_pred(lNode, indent):
    print(indent, "recur_pred: ", lNode)
    preds=list(graph.predecessors(lNode))
    print(indent, "predecessors returned for ", lNode, ": ", preds)
    indent+="  "
    for i in preds:
        if not i in all_pred:
            print("adding ", i, " to all_pred")
            all_pred.append(i)
        else:
            print(i, " is already in all_pred list, bypassing.")
        recur_pred(i, indent)

print("pred: e:")
print(list(graph.predecessors('e')))
print("subgraph: e:" )
print(list(graph.subgraph('e')))
print("succ: a:")
print(list(graph.successors('a')))
print("neigh: a:")
print(list(graph.neighbors('e')))

print("Gathering all predecessors of e:")
recur_pred('e', indent)

print("all predecesors: ", all_pred)
