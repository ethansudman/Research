import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import itertools
from scipy.stats import truncnorm

mph_corr = []
mpa_corr = []
mha_corr = []
mph_rho = []
mpa_rho = []
mha_rho = []

graph = nx.karate_club_graph()

#graph = nx.Graph()

#with open('karate-mirrored.edgelist', 'r') as file:
#    for line in file:
#        edge = line.replace("\n", "").split(' ')

#        graph.add_edge(edge[0], edge[1])

h, a = nx.hits(graph)

df_h = pd.DataFrame(h, index=['HubScore'])
df_h = df_h.swapaxes(0,1)
df_a = pd.DataFrame(a, index=['AuthorityScore'])
df_a = df_a.swapaxes(0,1)
df_comb = df_h.join(df_a)

pr = nx.pagerank(graph)
df_pr = pd.DataFrame(pr, index=['PageRank'])
df_pr = df_pr.swapaxes(0,1)
df_comb = df_pr.join(df_comb)

btwns = pd.DataFrame(nx.betweenness_centrality(graph), index=['Betweeness'])
btwns = btwns.swapaxes(0,1)
df_comb = df_comb.join(btwns)

df_comb.to_csv('hits_data_replicated.csv')

ph_corr = []
pa_corr = []
ha_corr = []
ph_rho = []
pa_rho = []
ha_rho = []

n = 0
#print '{:>2} {:16} {:16} {:>12} {:>10}'.format('#','Type1','Type2','T-Test p-val','Correl')
m_p = np.ones([5,5])
corr_list = []
for s1,s2 in itertools.combinations(range(len(df_comb.columns)),2):
    n = n + 1
    sl1 = df_comb.iloc[:,s1]
    sl2 = df_comb.iloc[:,s2]
    tt = stats.ttest_rel(sl1,sl2)
    r = stats.pearsonr(sl1,sl2)
    #print '{:2} {:16} {:16} {:12f}
    ##{:10f}'.format(n,sl1.name,sl2.name,tt.pvalue, r[0])
    ##print tt.pvalue, r[0]
    m_p[s1][s2] = r[0]
    m_p[s2][s1] = r[0]    
    corr_list.append(r[0])
        
    #print corr_list[0], corr_list[1], corr_list[2]
        
ph_corr.append(corr_list[0])
pa_corr.append(corr_list[1])
ha_corr.append(corr_list[2])
        
        
        # Get top 10
        #for c in range(len(df_comb.columns)):
        #    print df_comb.iloc[:,c].sort_values(ascending=False)[:25]
        
        # Get ranks
df_ranks = df_comb.rank(ascending=False)
        
        # Paired t-tests on ranks
        #stats.ttest_rel(rvs1,rvs2)
        #list(itertools.combinations(df_comb.columns,2))
n = 0
        #print '{:>2} {:16} {:16} {:>12}
        #{:>10}'.format('#','Type1','Type2','T-Test p-val','Correl(rho)')
m_s = np.ones([5,5])
corr_list = []
for s1,s2 in itertools.combinations(range(len(df_ranks.columns)),2):
    n = n + 1
    sl1 = df_ranks.iloc[:,s1]
    sl2 = df_ranks.iloc[:,s2]
    tt = stats.ttest_rel(sl1,sl2)
    r = stats.pearsonr(sl1,sl2)
            #print '{:2} {:16} {:16} {:12f}
            #{:10f}'.format(n,sl1.name,sl2.name,r[1], r[0])
            #print r[1], r[0]
    m_s[s1][s2] = r[0]
    m_s[s2][s1] = r[0]    
    corr_list.append(r[0])
        
        #print corr_list[0], corr_list[1], corr_list[2]
    
ph_rho.append(corr_list[0])
pa_rho.append(corr_list[1])
ha_rho.append(corr_list[2])

epsilon = 10**-5

def test(row):
    hs = row.HubScore
    aus = row.AuthorityScore
    res = row.HubScore / (row.AuthorityScore + epsilon)
    return res

#df_comb['HA_Ratio'] = df_comb.apply(lambda row: row.HubScore / (row.AuthorityScore + epsilon), axis = 1)
df_comb['HA_Ratio'] = df_comb.apply(test, axis = 1)
    
        # Visualize Pearson correlations
names = ['Hub', 'Auth', 'PageRank', 'Degree', 'Btws']
    #plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(m_p, vmin=-1, vmax=1, cmap='PiYG')
cax = ax.matshow(m_p)
fig.colorbar(cax)
ticks = np.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

fig2 = plt.figure()        
print(np.mean(ph_corr), np.mean(pa_corr), np.mean(ha_corr))
print(np.mean(ph_rho), np.mean(pa_rho), np.mean(ha_rho))
mph_corr.append(ph_corr)
mpa_corr.append(pa_corr)
mha_corr.append(ha_corr)
mph_rho.append(ph_rho)
mpa_rho.append(pa_rho)
mha_rho.append(ha_rho)
nx.draw_networkx(graph)

#fig3 = plt.figure()
#nx.draw_networkx(nx.karate_club_graph())
input()