# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 23:26:53 2021

@author: Ethan_000
"""

import glob
import re
import networkx as nx
import pandas as pd

#df = pd.DataFrame(columns = ['Colony', 'Day', 'Ant', 'Betweenness'])

def retrieve(d, key):
    if key in d:
        return d[key]
    else:
        return ''

data = []

for file in glob.glob("*.graphml"):    
    m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', file).groups()
    
    G = nx.read_graphml(file)
    nodes = G.nodes()
    
    betweenness = nx.betweenness_centrality(G)
    for ant in betweenness.keys():        
        data.append({'Colony': m[0],
                     'Day': m[1],
                     'Ant': ant,
                     'Betweenness': betweenness[ant],
                     'group_period1': retrieve(nodes[ant], 'group_period1'),
                     'group_period2': retrieve(nodes[ant], 'group_period2'),
                     'group_period3': retrieve(nodes[ant], 'group_period4'),
                     'group_period4': retrieve(nodes[ant], 'group_period4')})
        #df = df.append(data, ignore_index = True)
#pd.DataFrame.from_records(data).to_csv('Betwenness Centrality.csv')
pd.DataFrame.from_records(data).to_csv('Betwenness Centrality 2.csv')
    