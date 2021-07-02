# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:03:35 2021

@author: Ethan_000
"""

import numpy as np
import pandas as pd
#import glob
#import re
import networkx as nx
import networkx.drawing as dw
import networkx.drawing.layout as lt
import matplotlib.pyplot as plt
import matplotlib
#import collections

def run(key, events):
    if key in events:
        return events[key]
    else:
        return 0

base = 'C:\\Users\\Ethan_000\\Source\\Repos\\Research\\Research\\'

G = nx.read_graphml(base + 'ant_mersch_col1_day10_attribute.graphml')

C = G.copy()

for node in G.nodes(data = True):
    #print(node)
    x = dict(node[1])
    
    if ('group_period1' in x and x['group_period1'] != 'F'):
        
        if ((str(node[0])).startswith('Ant')):
            C.remove_node(node[0])

#foragerVisits = nx.get_node_attributes(G, 'nb_interaction_foragers').items()

foragerEvents = nx.get_node_attributes(C, 'nb_foraging_events')#.items()

vmax = max(foragerEvents[c] for c in foragerEvents)
print('vmax')
print(vmax)

#d = {}

#for event in foragerEvents:
    #C.nodes()[event[0]] = event[1] / vmax
    #nx.set_node_attributes(C, event[1] / vmax, 'Color')
    
#colors = [run(node, foragerEvents) / vmax for node in C.nodes()]
colors = [run(node, foragerEvents) for node in C.nodes()]
print(colors)

vmin = min(foragerEvents[c] for c in foragerEvents)
#layout = nx.fruchterman_reingold_layout(C)
#layout = nx.layout.kamada_kawai_layout(C)
layout = nx.layout.planar_layout(C)
cmap = plt.cm.coolwarm

nx.draw_networkx(C, pos = layout, node_color = colors, cmap = cmap, vmin = vmin, vmax = vmax)