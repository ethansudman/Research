#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:24:47 2017

@author: marker
"""

import networkx as nx
#import pydot
from networkx import read_graphml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hits_custom(G, max_iter=100, tol=1.0e-8, nstart=None, normalized=True):
    """Return HITS hubs and authorities values for nodes.
    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.
    Parameters
    ----------
    G : graph
      A NetworkX graph
    max_iter : interger, optional
      Maximum number of iterations in power method.
    tol : float, optional
      Error tolerance used to check convergence in power method iteration.
    nstart : dictionary, optional
      Starting value of each node for power method iteration.
    normalized : bool (default=True)
       Normalize results by the sum of all of the values.
    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.
    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.
    Examples
    --------
    >>> G=nx.path_graph(4)
    >>> h,a=nx.hits(G)
    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.
    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.
    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("hits() not defined for graphs with multiedges.")
    if len(G) == 0:
        return {}, {}
    # choose fixed starting vector if not given
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        # normalize starting vector
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):  # power iteration: make up to max_iter iterations
        hlast = h
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        # this "matrix multiply" looks odd because it is
        # doing a left multiply a^T=hlast^T*G
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr].get('weight', 1)
        # now multiply h=Ga
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr].get('weight', 1)
        # normalize vector
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        # normalize vector
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        # check convergence, l1 norm
        err = sum([abs(h[n] - hlast[n]) for n in h])
        if err < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return h, a

#graph = read_graphml(open("c.elegans.herm_pharynx_1.graphml"))

#graph2 = read_graphml(open("c.elegans_neural.male_1.graphml"))

graph3 = nx.DiGraph()

cells = pd.read_excel("cells.xls")
cell_array = cells.values
names = list(cell_array[:,1])


data = pd.read_excel("NeuronConnect-cElegans.xls")
dataA = data.values

for t in dataA[:,1]:
    graph3.add_node(t)
    
for t in dataA:
    if (t[2] == 'Sp' or t[2] == 'S'):
        graph3.add_edge(t[0],t[1],weight=t[3])
    if(t[2] == 'EJ' or t[2] == 'NMJ'):
        graph3.add_edge(t[0],t[1],weight=t[3])

#dicts = nx.to_dict_of_dicts(graph2)

hits = nx.hits(graph3,max_iter=300)
pr = nx.pagerank(graph3)
degcentr = nx.degree_centrality(graph3)
indegcentr = nx.in_degree_centrality(graph3)
outdegcentr = nx.out_degree_centrality(graph3)
closecentr = nx.closeness_centrality(graph3)
#curclosecentr = nx.current_flow_closeness_centrality(graph3)
#curbetween = nx.current_flow_betweenness_centrality(graph3)
evcentr = nx.eigenvector_centrality(graph3)
#katz = nx.katz_centrality(graph3,max_iter=10000,tol=1e-03)
#comm = nx.communicability_centrality(graph3)
load = nx.load_centrality(graph3)
between = nx.betweenness_centrality(graph3)

sensoryPR = []
motorPR = []
interPR = []

sensoryHA = []
motorHA = []
interHA = []

sensoryHH = []
motorHH = []
interHH = []

for k in graph3.nodes():
    if cell_array[names.index(k)][5] == 'I':
        interPR.append([k,pr[k]])
        interHH.append([k,hits[0][k]])
        interHA.append([k,hits[1][k]])
    elif cell_array[names.index(k)][5] == 'S':
        sensoryPR.append([k,pr[k]])
        sensoryHH.append([k,hits[0][k]])
        sensoryHA.append([k,hits[1][k]])
    elif cell_array[names.index(k)][5] == 'M':
        motorPR.append([k,pr[k]])
        motorHH.append([k,hits[0][k]])
        motorHA.append([k,hits[1][k]])
    else:
        print("error")
        
        
spr = np.array(sensoryPR)[:,1].astype(float)
mpr = np.array(motorPR)[:,1].astype(float)
ipr = np.array(interPR)[:,1].astype(float)

sha = np.array(sensoryHA)[:,1].astype(float)
mha = np.array(motorHA)[:,1].astype(float)
iha = np.array(interHA)[:,1].astype(float)

shh = np.array(sensoryHH)[:,1].astype(float)
mhh = np.array(motorHH)[:,1].astype(float)
ihh = np.array(interHH)[:,1].astype(float)

print("Average")
print(np.average(spr), np.average(mpr), np.average(ipr))
print(np.average(sha), np.average(mha), np.average(iha))
print(np.average(shh), np.average(mhh), np.average(ihh))
print("Median")
print(np.median(spr), np.median(mpr), np.median(ipr))
print(np.median(sha), np.median(mha), np.median(iha))
print(np.median(shh), np.median(mhh), np.median(ihh))
print("Std")
print(np.std(spr), np.std(mpr), np.std(ipr))
print(np.std(sha), np.std(mha), np.std(iha))
print(np.std(shh), np.std(mhh), np.std(ihh))
print("Min")
print(np.min(spr), np.min(mpr), np.min(ipr))
print(np.min(sha), np.min(mha), np.min(iha))
print(np.min(shh), np.min(mhh), np.min(ihh))
print("Max")
print(np.max(spr), np.max(mpr), np.max(ipr))
print(np.max(sha), np.max(mha), np.max(iha))
print(np.max(shh), np.max(mhh), np.max(ihh))
print("Sum")
print(np.sum(spr), np.sum(mpr), np.sum(ipr))
print(np.sum(sha), np.sum(mha), np.sum(iha))
print(np.sum(shh), np.sum(mhh), np.sum(ihh))

role_map=dict(zip(cells['cell_name'].values,cells['role'].values))
df=pd.DataFrame(index=list(pr.keys())) #['cell_name'])
df['pagerank']=pr.values()
df['hubs']=hits[0].values()
df['auth']=hits[1].values()
df['degcentr']=degcentr.values()
df['between']=between.values()

df['indegcentr'] = indegcentr.values()
df['outdegcentr'] = outdegcentr.values()
df['closecentr'] = closecentr.values()
#df['curclosecentr'] = curclosecentr.values()
#df['curbetween'] = curbetween.values()
df['evcentr'] = evcentr.values()
#df['katz'] = katz.values()
#df['comm'] = comm.values()
df['load'] = load.values()

#df['hadiff']=df.hubs-df.auth
df['haratio']=df.hubs/(df.auth+0.00001)
df['paratio']=df.pagerank/(df.auth+0.00001)
df['phratio']=df.pagerank/(df.hubs+0.00001)
df['haprod']=df.hubs*df.auth
df['paprod']=df.pagerank*df.auth
df['phprod']=df.pagerank*df.hubs
df['hadiff']=df.hubs-df.auth
df['padiff']=df.pagerank-df.auth
df['phsum']=df.pagerank-df.hubs
df['hasum']=df.hubs+df.auth
df['pasum']=df.pagerank+df.auth
df['phsum']=df.pagerank+df.hubs
df['ha_harmonic_mean']= 2*df.auth*df.hubs/(df.auth+df.hubs)
#df['loghasum']=np.log1p(df.hubs+df.auth)
#df['loghaprod']=np.log1p(df.hubs*df.auth)
#df['loghloga']=np.log1p(df.hubs)+np.log1p(df.auth)

# Compute the number of nodes in each strongly connected
# componenet for the given node
num_comps_node = {}
#for Gc in nx.strongly_connected_component_subgraphs(graph3):
for Gc in nx.strongly_connected_components(graph3):
    #print(Gc.nodes())
    print(len(Gc))
    #for n in Gc.nodes():
    for n in Gc:
        num_comps_node[n] = len(Gc)

df['hasum_normed'] = df.hasum * pd.Series(num_comps_node) / len(graph3)

df['role']=pd.Series(role_map) #df['cell_name'].map(role_map) 
 
#writer = pd.ExcelWriter('output_072519.xlsx')
#df.to_excel(writer,'Sheet1')
#writer.save()

df.to_csv('output_harmmean_062120.csv')

#[len(Gc) for Gc in sorted(nx.strongly_connected_component_subgraphs(graph3),key=len, reverse=True)]

