import networkx as nx
import pandas as pd
import glob
import re
import numpy as np

def random_reference(G, niter=1, connectivity=True, seed=None):
    if G.is_directed():
        msg = "random_reference() not defined for directed graphs."
        raise nx.NetworkXError(msg)
    if len(G) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")

    from networkx.utils import cumulative_distribution, discrete_sequence

    local_conn = nx.connectivity.local_edge_connectivity

    G = G.copy()
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = cumulative_distribution(degrees)  # cdf of degree
    nnodes = len(G)
    nedges = nx.number_of_edges(G)
    niter = niter * nedges
    ntries = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    swapcount = 0

    for i in range(niter):
        n = 0
        while n < ntries:
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ai, ci) = discrete_sequence(2, cdistribution=cdf, seed=seed)
            if ai == ci:
                continue  # same source, skip
            a = keys[ai]  # convert index to label
            c = keys[ci]
            # choose target uniformly from neighbors
            b = seed.choice(list(G.neighbors(a)))
            d = seed.choice(list(G.neighbors(c)))
            bi = keys.index(b)
            di = keys.index(d)
            if b in [a, c, d] or d in [a, b, c]:
                continue  # all vertices should be different

            # don't create parallel edges
            if (d not in G[a]) and (b not in G[c]):
                G.add_edge(a, d)
                G.add_edge(c, b)
                G.remove_edge(a, b)
                G.remove_edge(c, d)

                # Check if the graph is still connected
                if connectivity and local_conn(G, a, b) == 0:
                    # Not connected, revert the swap
                    G.remove_edge(a, d)
                    G.remove_edge(c, b)
                    G.add_edge(a, b)
                    G.add_edge(c, d)
                else:
                    swapcount += 1
                    break
            n += 1
    return G

def sigma(G, niter=100, nrand=10, seed=None):
    # Compute the mean clustering coefficient and average shortest path length
    # for an equivalent random graph
    randMetrics = {"C": [], "L": []}
    for i in range(nrand):
        Gr = random_reference(G, niter=niter, seed=seed)
        randMetrics["C"].append(nx.transitivity(Gr))
        randMetrics["L"].append(nx.average_shortest_path_length(Gr))

    C = nx.transitivity(G)
    L = nx.average_shortest_path_length(G)
    Cr = np.mean(randMetrics["C"])
    Lr = np.mean(randMetrics["L"])

    sigma = (C / Cr) / (L / Lr)

    return sigma

df = pd.DataFrame(columns = ['Colony', 'Day', 'Degree Assortativity Coefficient', 'Average Clustering', 'Nodes', 'Edges', 'Density', 'Diameter', 'Betweenness Centrality', 'Connected Components', 'Local Efficiency', 'Global Efficiency'])

for file in glob.glob("*.graphml"):
    m = re.match('^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', file).groups()
    
    data = {'Colony': m[0], 'Day': m[1]}

    G = nx.read_graphml(file)
    data['Degree Assortativity Coefficient'] = nx.degree_assortativity_coefficient(G)
    data['Nodes'] = nx.number_of_nodes(G)
    data['Edges'] = nx.number_of_edges(G)
    data['Density'] = nx.density(G)
    data['Diameter'] = nx.diameter(G)    
    data['Average Clustering'] = nx.algorithms.cluster.average_clustering(G)
    data['Betweenness Centrality'] = sum(nx.betweenness_centrality(G).values()) / nx.number_of_nodes(G)
    data['Average Degree'] = sum(dict(G.degree()).values()) / nx.number_of_nodes(G)
    data['Connected Components'] = sum(1 for component in nx.connected_components(G))
    data['Local Efficiency'] = nx.algorithms.local_efficiency(G)
    data['Global Efficiency'] = nx.algorithms.global_efficiency(G)
    #data['Girvan-Newman Method Communities'] = sum(1 for community in nx.algorithms.community.girvan_newman(G))
    #data['Small-World Coefficient (Sigma)'] = sigma(G)
    #data['Small-World Coefficient (Omega)'] = nx.algorithms.smallworld.omega(G)
    
    df = df.append(data, ignore_index = True)
    #data[colony][day]['Average neighbor degree'] = nx.average_neighbor_degree(G)
    
m = np.max(df['Nodes'])

df['Normalized Degree Average'] = np.multiply(m, df['Average Degree'])
df['Normalized Degree Average'] = np.divide(df['Normalized Degree Average'], df['Nodes'])

df.to_csv('Graph Descriptive Statistics.csv')


