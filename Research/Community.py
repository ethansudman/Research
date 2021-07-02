# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:42:14 2021

@author: Ethan_000
"""

import glob
import re
import networkx as nx
import pandas as pd
from os.path import exists

df = None

maxDay = 0
maxColony = 0

if exists('Community.csv'):
    df = pd.read_csv('Community.csv')
    maxDay = df['Day'].max()
    maxColony = df['Colony'].max()
else:
    df = pd.DataFrame(columns = ['Colony', 'Day', 'Communities'])

for file in glob.glob("*.graphml"):
    m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', file).groups()
    
    if int(m[0]) > maxColony or int(m[1]) > maxDay:
        data = {'Colony': m[0], 'Day': m[1]}
    
        G = nx.read_graphml(file)
        data['Girvan-Newman Method Communities'] = sum(1 for community in nx.algorithms.community.girvan_newman(G))
        
        df = df.append(data, ignore_index = True)
        
        df.to_csv('Communities.csv')