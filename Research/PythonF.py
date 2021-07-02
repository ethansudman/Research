# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 20:37:12 2021

@author: Ethan_000
"""

import sys
import re
import networkx as nx
import pandas as pd

print("abc")

file = sys.argv[0]

print(file)

m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.edges$', file).groups()
    
with open('Colony ' + m[0] + ' Day ' + m[1] + ' Log.txt', 'w') as f:  
    data = {'Colony': m[0], 'Day': m[1]}
    
    G = nx.read_edgelist(file)
        
    f.write('Opened graph\n')
    data['Degree Assortativity Coefficient'] = nx.degree_assortativity_coefficient(G)
    data['Nodes'] = nx.number_of_nodes(G)
    data['Edges'] = nx.number_of_edges(G)
    data['Density'] = nx.density(G)
    data['Diameter'] = nx.diameter(G)    
    data['Average Clustering'] = nx.algorithms.cluster.average_clustering(G)
    data['Betweenness Centrality'] = sum(nx.betweenness_centrality(G).values()) / nx.number_of_nodes(G)
    data['Average Degree'] = sum(dict(G.degree()).values()) / nx.number_of_nodes(G)
    data['Connected Components'] = sum(1 for component in nx.connected_components(G))
        
    f.write('About to start community processing\n')
    data['Girvan-Newman Method Communities'] = sum(1 for community in nx.algorithms.community.girvan_newman(G))
        #df = pd.DataFrame(data, columns = ['Colony', 'Day', 'Degree Assortativity Coefficient', 'Average Clustering', 'Nodes', 'Edges', 'Density', 'Diameter', 'Betweenness Centrality', 'Connected Components', 'Girvan-Newman Method Communities', 'Local Efficiency', 'Global Efficiency', 'Small-World Coefficient (Sigma)', 'Small-World Coefficient (Omega)'])
        
    f.write('About to start data frame\n')
    df = pd.DataFrame(data)
    f.write('About to do to_csv\n')
    df.to_csv('Graph Descriptive Statistics - Colony ' + m[0] + ' Day ' + m[1] + '.csv')
    f.write('Did to_csv')