# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:04:31 2021

@author: Ethan_000
"""

import networkx as nx
import glob
import re

for file in glob.glob("*.graphml"):
    m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', file).groups()
    G = nx.read_graphml(file)
    nx.write_edgelist(G, 'ant_mersch_col' + m[0] + '_day' + m[1] + '_attribute.edges')