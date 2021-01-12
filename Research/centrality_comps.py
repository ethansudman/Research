#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:13:31 2020

@author: szczurpi
"""

import networkx as nx
from networkx import read_graphml
#import pydot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_centralities_df(G, file_name):
    # Compute prestige/centrality scores
    pr = nx.pagerank(G)
    degcentr = nx.degree_centrality(G)
    closecentr = nx.closeness_centrality(G)
    evcentr = nx.eigenvector_centrality(G)
    katz = nx.katz_centrality(G,max_iter=10000,tol=1e-03)
    load = nx.load_centrality(G)
    between = nx.betweenness_centrality(G)
    h,a = nx.hits(G,max_iter=300)
    
    # Put results in a dataframe
    df=pd.DataFrame(index=list(pr.keys()))
    df['pagerank']=pr.values()
    df['hubs']=h.values()
    df['auth']=a.values()
    df['degcentr']=degcentr.values()
    df['between']=between.values()
    df['closecentr'] = closecentr.values()
    df['evcentr'] = evcentr.values()
    df['katz'] = katz.values()
    df['load'] = load.values()
    df['hasum']=df.hubs+df.auth
    
    if file_name:
        df.to_csv(file_name)
    
    return df

def graph(A):
    #graph network from adjacency matrix
    rows,cols = np.where(A == 1)
    links= zip(rows.tolist(),cols.tolist())
    G = nx.Graph(A)
    G.add_edges_from(links)
    layout = nx.spring_layout(G,seed = 95)

    nx.draw(G,pos = layout,arrowsize = 20 ,node_size = 500, with_labels=True)
    return G   
    # Save in a csv file



# Adjacency matrix for an example directed graph
A = np.array([[0,1,1,1],[0,0,1,1],[0,1,0,0],[0,0,0,0]])

# Adjacency matrix for an undirected version of A
S = np.array([[0,0,0,0,0,1],[1,0,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]])

# Make networkx graph structures
#GA = nx.DiGraph(A)
#GS = nx.Graph(S)

#df_GA = get_centralities_df(GA, 'GA_centrality_scores.csv')
GS = graph(S)
df_GS = get_centralities_df(GS, 'GS_centrality_scores.csv')


