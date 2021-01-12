
"""
Spyder Editor

This is a temporary script file.
"""

import networkx as nx
import numpy as np
import pandas as pd
import glob
import re

#data = {}
#graphs = {}

evo_over_time = None

base = 'C:\\Users\\Ethan_000\\Source\\Repos\\Research\\Research\\'

trackingDataBase = 'E:\\School stuff\\Lewis\\Research\\tracking_data\\tracking_data\\'

behavior = pd.read_csv(trackingDataBase + 'behavior.csv')

# Annoyingly enough, they're called colony 1, 2, 3, 4... in one place and 4, 18, 21, 29... in another
# TODO: Make sure that this is true - I've only checked the first one, and that wasn't 100% thorough
behavior['colony'] = behavior['colony'].replace(4, 1)
behavior['colony'] = behavior['colony'].replace(18, 2)
behavior['colony'] = behavior['colony'].replace(21, 3)
behavior['colony'] = behavior['colony'].replace(29, 4)
behavior['colony'] = behavior['colony'].replace(58, 5)
behavior['colony'] = behavior['colony'].replace(78, 6)

for graphml in glob.glob(base + "*.graphml"):
    m = re.match('.*ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', graphml).groups()
    
    colony = int(m[0])
    day = int(m[1])

    #if m[0] not in data.keys():
    #    data[m[0]] = []
    #    graphs[m[0]] = {}

    #data[m[0]].append(m[1])
    
    graph = nx.read_graphml(graphml)

    #graphs[m[0]][m[1]] = graph
    
    h, a = nx.hits(graph)

    # TODO: Add role in colony here
    df_h = pd.DataFrame(h, index=['HubScore'])
    df_h = df_h.swapaxes(0,1)
    df_a = pd.DataFrame(a, index=['AuthorityScore'])
    df_a = df_a.swapaxes(0,1)
    df_comb = df_h.join(df_a)
    
    pr = nx.pagerank(graph)
    df_pr = pd.DataFrame(pr, index=['PageRank'])
    df_pr = df_pr.swapaxes(0,1)
    df_comb = df_pr.join(df_comb)
    
    btc = nx.betweenness_centrality(graph)
    
    btwns = pd.DataFrame(btc, index=['Betweenness'])
    btwns = btwns.swapaxes(0,1)
    df_comb = df_comb.join(btwns)
    
    #deg = nx.degree(graph)
    
    #df_deg = pd.DataFrame(deg, index=['Degree'])
    #df_deg = df_deg.swapaxes(0,1)
    #df_comb = df_comb.join(df_deg)
    
    df_comb['Colony'] = [m[0]] * len(df_comb)
    df_comb['Day'] = [m[1]] * len(df_comb)
    
    df_comb['Role'] = ''
        
    for ant in h.keys():
        # A tag ID of, for example, 5, would be called "Ant5" in the other file
        # Also, annoyingly enough, they re-use the same tags in multiple colonies, so we need both conditions
        rows = behavior.loc[(behavior['colony'] == colony) & (behavior['tag_id'] == int(ant[3:]))]
        if not rows.empty:
            row = rows.iloc[0]
            
            # Yes, I do realize that this could be shortened, but it's clearer this way
            role = None
            if day >= 31:
                role = row['group_period4']
            elif day >= 21:
                role = row['group_period3']
            elif day > 11:
                role = row['group_period2']
            else:
                role = row['group_period1']
            
            if str(role).strip() == '':
                role = 'U'
                
            df_comb.loc[ant, 'Role'] = role
        if str(df_comb.loc[ant, 'Role']).strip() == '':
            df_comb.loc[ant, 'Role'] = 'U'
                
            #df_comb.loc[df_comb['Unnamed: 0'] == ant].iloc[0]['Role'] = role
                
            #df_comb['Role'] = role
    
    df_comb.fillna(0, inplace = True)
     
    if evo_over_time is None:
        evo_over_time = df_comb
    else:
        evo_over_time = pd.concat([evo_over_time, df_comb])
    
    df_comb.to_csv('{baseDir}Colony {colony} day {day}.csv'.format(baseDir = base, colony = str(m[0]), day = str(m[1])))
    
evo_over_time.to_csv(base + 'evo_over_time.csv')

evo_over_time.loc[(evo_over_time['Betweenness'] < 0.0001), 'Betweenness'] = 0.0001
evo_over_time.loc[(evo_over_time['HubScore'] < 0.0001), 'HubScore'] = 0.0001
evo_over_time.loc[(evo_over_time['AuthorityScore'] < 0.0001), 'AuthorityScore'] = 0.0001

evo_over_time.to_csv(base + 'evo_over_time_orange.csv')

#nx.read_graphml(r'C:\Users\Ethan_000\Source\Repos\Research\Research\ant_mersch_col1_day01_attribute.graphml')

#print('x')

#def getGraph():
#    graph = nx.Graph()
#    
#   with open('C:\\Users\\Ethan_000\\Source\\Repos\\Research\\Research\\karate-mirrored.edgelist', 'r') as file:
#        for line in file:
#            edge = line.replace("\n", "").split(' ')
#    
#            graph.add_edge(edge[0], edge[1])
    
#    print(graph)
    
#    return graph

#gr = getGraph()
#print('Done')