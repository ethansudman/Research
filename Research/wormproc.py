import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import itertools

# Load the worm data as a dataframe
df = pd.read_csv('NeuronConnect-cElegans-Gephi.csv')
#df = pd.read_csv('NeuronConnect.csv')
#G_fly = nx.read_graphml('drosophila_medulla_1.graphml')
#G_mouse = nx.read_graphml('mouse_brain_1.graphml')
#G_rat = nx.read_graphml('rattus.norvegicus_brain_1.graphml')

df.rename(columns={'Weight':'weight'},inplace=True)

# Remove Rp edges
df = df[df.SynapseType!='Rp']
df = df[df.SynapseType!='R']

#df = df[df.SynapseType!='Sp']
#df = df[df.SynapseType!='S']
df = df[df.SynapseType!='NMJ']
#df = df[df.SynapseType!='EJ']

# Convert to graph format
G=nx.from_pandas_edgelist(df, 'Source', 'Target', ['weight', 'SynapseType'], nx.DiGraph())
#G = nx.DiGraph(G_rat)
# Run HITS
h,a = nx.hits(G,max_iter=300)
df_h = pd.DataFrame(h, index=['HubScore'])
df_h = df_h.swapaxes(0,1)
df_a = pd.DataFrame(a, index=['AuthorityScore'])
df_a = df_a.swapaxes(0,1)
df_comb = df_h.join(df_a)

# Run PageRank
pr = nx.pagerank(G)
df_pr = pd.DataFrame(pr, index=['PageRank'])
df_pr = df_pr.swapaxes(0,1)
df_comb = df_pr.join(df_comb)

# Get degree centrality
#degs = pd.DataFrame(G.degree(), index=['Degree'])
#degs = degs.swapaxes(0,1)
#df_comb = df_comb.join(degs)

# Get betweeness centrality
#btwns = pd.DataFrame(nx.betweenness_centrality(G), index=['Betweeness'])
#btwns = btwns.swapaxes(0,1)
#df_comb = df_comb.join(btwns)

# Save as csv
#df_comb.to_csv('hits_data_new.csv')

# Paired t-tests
#stats.ttest_rel(rvs1,rvs2)
#list(itertools.combinations(df_comb.columns,2))
n=0
print('{:>2} {:16} {:16} {:>12} {:>10}'.format('#','Type1','Type2','T-Test p-val','Correl'))
m_p = np.ones([5,5])
corr_list = []
for s1,s2 in itertools.combinations(range(len(df_comb.columns)),2):
    n = n + 1
    sl1 = df_comb.iloc[:,s1]
    sl2 = df_comb.iloc[:,s2]
    tt = stats.ttest_rel(sl1,sl2)
    r = stats.pearsonr(sl1,sl2)
    print('{:2} {:16} {:16} {:12f} {:10f}'.format(n,sl1.name,sl2.name,tt.pvalue, r[0]))
    #print tt.pvalue, r[0]
    m_p[s1][s2] = r[0]
    m_p[s2][s1] = r[0]    
    corr_list.append(r[0])

print(corr_list[0], corr_list[1], corr_list[2])



# Get top 10
for c in range(len(df_comb.columns)):
    print(df_comb.iloc[:,c].sort_values(ascending=False)[:25])

# Get ranks
df_ranks = df_comb.rank(ascending=False)

# Paired t-tests on ranks
#stats.ttest_rel(rvs1,rvs2)
#list(itertools.combinations(df_comb.columns,2))
n=0
print('{:>2} {:16} {:16} {:>12} {:>10}'.format('#','Type1','Type2','T-Test p-val','Correl(rho)'))
m_s = np.ones([5,5])
corr_list = []
for s1,s2 in itertools.combinations(range(len(df_ranks.columns)),2):
    n = n + 1
    sl1 = df_ranks.iloc[:,s1]
    sl2 = df_ranks.iloc[:,s2]
    tt = stats.ttest_rel(sl1,sl2)
    r = stats.pearsonr(sl1,sl2)
    print('{:2} {:16} {:16} {:12f} {:10f}'.format(n,sl1.name,sl2.name,r[1], r[0]))
    #print r[1], r[0]
    m_s[s1][s2] = r[0]
    m_s[s2][s1] = r[0]    
    corr_list.append(r[0])

print(df)
print(corr_list[0], corr_list[1], corr_list[2])

# Visualize Pearson correlations
names = ['Hub', 'Auth', 'PageRank', 'Degree', 'Btws']
# plot correlation matrix
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(m_p, vmin=-1, vmax=1, cmap='PiYG')
#cax = ax.matshow(m_p)
#fig.colorbar(cax)
#ticks = np.arange(0,5,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
#plt.show()
