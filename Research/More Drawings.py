# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:42:07 2021

@author: Ethan_000
"""

import networkx as nx
import networkx.drawing as dw
import networkx.drawing.layout as lt
import matplotlib.pyplot as plt
import matplotlib

#G = nx.Graph()
#G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
#G.add_edges_from([(1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (7, 8), (8, 9)])

#print(G.edges())

G = nx.karate_club_graph()
nx.write_graphml(G, 'kc.graphml')

#nx.draw_spring(G)

#pos = nx.layout.circular_layout(G)
#nx.draw_networkx(G, pos)