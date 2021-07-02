# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 17:21:06 2021

@author: Ethan_000
"""

import glob
import re
import networkx as nx
import json
import pandas as pd

data = {}

for file in glob.glob("*.graphml"):
    m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', file).groups()
    
    G = nx.read_graphml(file)
    
    for ant in G.nodes(data = True):
        #print(ant)
        if 'group_period1' in ant[1]:
            if ant[0] not in data:
                data[ant[0]] = {}
            data[ant[0]][m[1]] = ant[1]['group_period1']
            data[ant[0]]['Interactions'] = ant[1]['nb_interaction_foragers']
            
updated = []

for key, value in data.items():
    dailyData = {'Ant': key}
    
    for key2, value2 in value.items():
        if key2 == 'Interactions':
            dailyData['Interactions'] = value2
        else:
            dailyData['Day'] = key2
            dailyData['Class'] = value2
    
    updated.append(dailyData)

pd.DataFrame.from_records(updated).to_csv('Ant forager interactions.csv')

with open('Json.json', 'w') as f:
    f.write(json.dumps(data))
    
pd.read_json('Json.json').to_csv('Test.csv')

#pd.DataFrame().json_

pd.json_normalize(data).to_csv('Age Data.csv')
            
print(data)

def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

for file in glob.glob("*.graphml"):
    m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', file).groups()
    
    G = nx.read_graphml(file)
    
    #communities = sorted(nx.algorithms.community.girvan_newman(G), key=len, reverse=True)
    
    communities = nx.algorithms.community.girvan_newman(G)
    
    for c, community in enumerate(communities):
        print('Community')
        print(community)
        for ant in community:
            print('Ant:')
            print(ant)
            G.node[ant]['community'] = c + 1