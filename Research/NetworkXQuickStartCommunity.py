# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 17:29:15 2021

@author: Ethan_000
"""

import networkx as nx

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            print(v)
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1

def set_edge_community(G):
    for v, w, in G.edges:
            if G.nodes[v]['community'] == G.nodes[w]['community']:
                # Internal edge, mark with community
                G.edges[v, w]['community'] = G.nodes[v]['community']
            else:
                # External edge, mark as 0
                G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)

G_karate = nx.read_graphml('ant_mersch_col1_day15_attribute.graphml')

communities = sorted(nx.algorithms.community.girvan_newman(G_karate), key=len, reverse=True)

# Set node and edge communities
set_node_community(G_karate, communities)
set_edge_community(G_karate)
 # Set community color for nodes
node_color = [
    get_color(G_karate.nodes[v]['community'])
    for v in G_karate.nodes]
 # Set community color for internal edges
external = [
    (v, w) for v, w in G_karate.edges
    if G_karate.edges[v, w]['community'] == 0]
internal = [
    (v, w) for v, w in G_karate.edges
    if G_karate.edges[v, w]['community'] > 0]
internal_color = [
    get_color(G_karate.edges[e]['community'])
    for e in internal]

karate_pos = nx.spring_layout(G_karate)
# Draw external edges
nx.draw_networkx(
    G_karate, pos=karate_pos, node_size=0,
    edgelist=external, edge_color="#333333")
# Draw nodes and internal edges
nx.draw_networkx(
    G_karate, pos=karate_pos, node_color=node_color,
    edgelist=internal, edge_color=internal_color)