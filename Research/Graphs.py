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

#data = {}
#graphs = {}

base = 'C:\\Users\\Ethan_000\\Source\\Repos\\Research\\Research\\'

graph = nx.read_graphml(base + 'ant_mersch_col1_day10_attribute.graphml')

plt.figure()

pos = lt.spring_layout(graph)

nx.draw(graph, pos)

plt.show()
#layout = lt.spring_layout(graph)

#nx.draw(graph, pos = layout)

#for graphml in glob.glob(base + "*.graphml"):
#    m = re.match('.*ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', graphml).groups()
#    graph = nx.read_graphml(graphml)
#    
#    pos = lt.spring_layout(graph)
    
#    nx.draw(graph, pos)
#    print('x')