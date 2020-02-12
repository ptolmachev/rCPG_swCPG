import networkx as nx
# importing matplotlib.pyplot
import matplotlib.pyplot as plt
import json
import numpy as np

file = open("rCPG_swCPG.json", "rb+")
params = json.load(file)
b = np.array(params["b"])
c = np.array(params["c"])


# Networkx
# g = nx.Graph()
# for i in range(b.shape[0]):
#     for j in range(b.shape[1]):
#         if b[i,j] != 0.0:
#             g.add_edge(i, j)
#
# nx.draw(g, with_labels = True)
# plt.savefig("../img/graph_simple.png")

txt_file = ""
txt_file += "digraph G {\n"
txt_file += "size =\"4,4\"; \n"

