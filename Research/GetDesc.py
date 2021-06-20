# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 16:24:56 2021

@author: Ethan_000
"""

import networkx as nx
import pandas as pd
import glob
import re

data = {}

for file in glob.glob("*.graphml"):
    m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', file).groups()
    colony = m[0]
    day = m[1]

    if colony not in data.keys():
        data[colony] = {}

    if day not in data[colony].keys():
        data[colony][day] = {}

    G = nx.read_graphml(file)
    data[colony][day]['Degree Assortativity Coefficient'] = nx.degree_assortativity_coefficient(G)
    data[colony][day]['Average neighbor degree'] = nx.average_neighbor_degree(G)


df = pd.DataFrame(data)

df.to_csv('Graph Descriptive Statistics.csv')