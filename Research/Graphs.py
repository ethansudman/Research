# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:08:45 2021

@author: Ethan_000
"""

import networkx as nx
import numpy as np
import pandas as pd
import glob
import re
import networkx.drawing as dw
import networkx.drawing.layout as lt
import matplotlib.pyplot as plt
import matplotlib
import collections

#data = {}
#graphs = {}

base = 'C:\\Users\\Ethan_000\\Source\\Repos\\Research\\Research\\'

G = nx.read_graphml(base + 'ant_mersch_col1_day10_attribute.graphml')

C = G.copy()

for node in G.nodes(data = True):
    #print(node)
    x = dict(node[1])
    
    if ('group_period1' in x and x['group_period1'] != 'F'):
        
        if ((str(node[0])).startswith('Ant')):
            C.remove_node(node[0])
        
#print(G.degree())

print('Average clustering: ' + str(nx.average_clustering(C)))

print('Average shortest path: ' + str(nx.average_shortest_path_length(C)))

#Pairs = nx.all_pairs_shortest_path_length(C)

prs = [np.std(list(pair[1].values())) for pair in nx.all_pairs_shortest_path_length(C)]
print('Average std dev: ' + str(np.average(prs)))

#for pair in Pairs:
    #x = pd.DataFrame(Pairs)
    #y = pd.DataFrame(x[1])
    #z = pd.DataFrame(y.iloc[0][1])
    #print(z)
    #print('std: ' + str(np.std(y.iloc[0].values())))
#    print('std: ' + str(np.std(list(pair[1].values()))))

# TODO: Fix so that I get standard deviation

#print('Shortest path std real: ' + str(np.std(x[1][x[0]])))

#with open('degrees with all ants.csv', 'a') as f:
#    f.write('ant,degree\n')
#    f.writelines(ant + ',' + str(degree) + '\n' for ant, degree in G.degree())
    
#with open('degrees with other foragers.csv', 'a') as f:
#    f.write('ant,degree\n')
#    f.writelines(ant + ',' + str(degree) + '\n' for ant, degree in C.degree())
    #for ant, degree in G.degrees():
    #    f.writelines(ant + ',' + degree + '\n' f)
    
    #print(x['tag_id'])

#x = nx.betweenness_centrality(G)

#y = list(x.values())
#y.sort()
#plt.bar(list(range(len(y))), y)
#plt.bar(*zip(*x.items()))
#plt.show()
    
C = nx.watts_strogatz_graph(163, 61, 0.5)


print('Average clustering for Watts: ' + str(nx.average_clustering(C)))

print('Average shortest path for Watts: ' + str(nx.average_shortest_path_length(C)))

#Pairs = nx.all_pairs_shortest_path_length(C)

wts = [np.std(list(pair[1].values())) for pair in nx.all_pairs_shortest_path_length(C)]

print('average wts std: ' + str(np.average(wts)))
#print(wts)
#for pair in Pairs:
#    print('std watts: ' + str(np.std(list(pair[1].values()))))
#print('pairs: ')
#print(Pairs)
#print('Shortest path std watts: ' + str(np.std(pd.DataFrame(Pairs))))

#with open('degrees of watts strogatz.csv', 'a') as f:
#    f.write('node,degree\n')
#    f.writelines(str(node) + ',' + str(degree) + '\n' for node, degree in C.degree())

#degree_sequence = sorted([d for n, d in C.degree()], reverse=True)  # degree sequence
#print(degree_sequence)
#degreeCount = collections.Counter(degree_sequence)
#deg, cnt = zip(*degreeCount.items())

#fig, ax = plt.subplots()
#plt.bar(deg, cnt, width=0.80, color="b")

#plt.title("Degree Histogram")
#plt.ylabel("Count")
#plt.xlabel("Degree")
#ax.set_xticks([d + 0.4 for d in deg])
#ax.set_xticklabels(deg)

#plt.show()

#plt.figure()

#pos = lt.spring_layout(graph)

#nx.draw(graph, pos)

#plt.show()
#layout = lt.spring_layout(graph)

#nx.draw(graph, pos = layout)

#for graphml in glob.glob(base + "*.graphml"):
#    m = re.match('.*ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', graphml).groups()
#    graph = nx.read_graphml(graphml)
#    
#    pos = lt.spring_layout(graph)
    
#    nx.draw(graph, pos)
#    print('x')