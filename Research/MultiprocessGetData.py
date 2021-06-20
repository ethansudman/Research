# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 00:35:49 2021

@author: Ethan_000
"""

import networkx as nx
import pandas as pd
import glob
import re
from multiprocessing import Process

    #data['Small-World Coefficient (Sigma)'] = sigma(G)
    #data['Small-World Coefficient (Omega)'] = nx.algorithms.smallworld.omega(G)
    
    #df = df.append(data, ignore_index = True)
    #data[colony][day]['Average neighbor degree'] = nx.average_neighbor_degree(G)
    
#m = np.max(df['Nodes'])

#df['Normalized Degree Average'] = np.multiply(m, df['Average Degree'])
#df['Normalized Degree Average'] = np.divide(df['Normalized Degree Average'], df['Nodes'])

#df.to_csv('Graph Descriptive Statistics.csv')

def doCalculations(file):
    m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', file).groups()
    
    with open('Colony ' + m[0] + ' Day ' + m[1] + ' Log.txt', 'w') as f:  
        data = {'Colony': m[0], 'Day': m[1]}
    
        G = nx.read_graphml(file)
        
        f.write('opened graph')
        data['Degree Assortativity Coefficient'] = nx.degree_assortativity_coefficient(G)
        data['Nodes'] = nx.number_of_nodes(G)
        data['Edges'] = nx.number_of_edges(G)
        data['Density'] = nx.density(G)
        data['Diameter'] = nx.diameter(G)    
        data['Average Clustering'] = nx.algorithms.cluster.average_clustering(G)
        data['Betweenness Centrality'] = sum(nx.betweenness_centrality(G).values()) / nx.number_of_nodes(G)
        data['Average Degree'] = sum(dict(G.degree()).values()) / nx.number_of_nodes(G)
        data['Connected Components'] = sum(1 for component in nx.connected_components(G))
        
        f.write('About to start community processing')
        data['Girvan-Newman Method Communities'] = sum(1 for community in nx.algorithms.community.girvan_newman(G))
        #df = pd.DataFrame(data, columns = ['Colony', 'Day', 'Degree Assortativity Coefficient', 'Average Clustering', 'Nodes', 'Edges', 'Density', 'Diameter', 'Betweenness Centrality', 'Connected Components', 'Girvan-Newman Method Communities', 'Local Efficiency', 'Global Efficiency', 'Small-World Coefficient (Sigma)', 'Small-World Coefficient (Omega)'])
        
        f.write('About to start data frame')
        df = pd.DataFrame(data)
        f.write('About to do to_csv')
        df.to_csv('Graph Descriptive Statistics - Colony ' + m[0] + ' Day ' + m[1] + '.csv')
        f.write('Did to_csv')
    
processes = []

for file in glob.glob("*.graphml"):
    p = Process(target = doCalculations, args = (file))
    p.start()
    
    processes.append(p)
    
for p in processes:
    p.join()