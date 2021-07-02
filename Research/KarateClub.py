# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 17:04:07 2021

@author: Ethan_000
"""

import networkx as nx

G = nx.karate_club_graph()


comms = list(nx.algorithms.community.girvan_newman(G))

kclique = list(nx.algorithms.community.kclique.k_clique_communities(G, 4))

print(comms)
print('Length:')
print(len(comms))

print('kclique')
print(kclique)
print(len(kclique))