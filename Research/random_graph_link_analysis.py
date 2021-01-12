# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:07:38 2017

@author: lajkonik
"""
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import itertools
import numpy as np
from scipy.stats import truncnorm

mu = 22.120996
sigma = 17.704104
lower = 1
upper = 137
a, b = (lower - mu) / sigma, (upper - mu) / sigma
#fig, ax = plt.subplots(1, 1)
#r = truncnorm.rvs(a, b, size=1000, loc=22.120996, scale=17.704104)
#ax.hist(r, normed=True, histtype='stepfilled', alpha=0.5)
#ax.legend(loc='best', frameon=False)
#plt.show()

#for i in range(10):
#    r = truncnorm.rvs(a, b, size=1000, loc=mu, scale=sigma)
#    print pd.Series(r).describe()
    #print pd.Series(r).max()

g = list()

# Generate a random graph
mph_corr = []
mpa_corr = []
mha_corr = []
mph_rho = []
mpa_rho = []
mha_rho = []
#for p in [2**-12, 2**-11, 2**-10, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0]: #np.arange(10)/10.+.1:
for p in [2**-4.3875761371228759]: #np.arange(10)/10.+.1:
    print("p=",p)
    ph_corr = []
    pa_corr = []
    ha_corr = []
    ph_rho = []
    pa_rho = []
    ha_rho = []
    for i in range(100):
        print("i=",i)
        G = nx.fast_gnp_random_graph(n=281, p=p, directed=True)
        for u,v in G.edges():
            #a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            #a, b = (1 - 22.120996) / 17.704104, (137 - 22.120996) / 17.704104
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            G[u][v]['weight'] = truncnorm.rvs(a, b, loc=mu, scale=sigma)
        
        g.append(G)
        # Run HITS
        try:
            h,a = nx.hits(G,max_iter=300000)
        except:
            print("error in hits")
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
        df_comb.to_csv('hits_data_new.csv')
        
        # Paired t-tests
        #stats.ttest_rel(rvs1,rvs2)
        #list(itertools.combinations(df_comb.columns,2))
        n=0
        #print '{:>2} {:16} {:16} {:>12} {:>10}'.format('#','Type1','Type2','T-Test p-val','Correl')
        m_p = np.ones([5,5])
        corr_list = []
        for s1,s2 in itertools.combinations(range(len(df_comb.columns)),2):
            n = n + 1
            sl1 = df_comb.iloc[:,s1]
            sl2 = df_comb.iloc[:,s2]
            tt = stats.ttest_rel(sl1,sl2)
            r = stats.pearsonr(sl1,sl2)
            #print '{:2} {:16} {:16} {:12f} {:10f}'.format(n,sl1.name,sl2.name,tt.pvalue, r[0])
            #print tt.pvalue, r[0]
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
        n=0
        #print '{:>2} {:16} {:16} {:>12} {:>10}'.format('#','Type1','Type2','T-Test p-val','Correl(rho)')
        m_s = np.ones([5,5])
        corr_list = []
        for s1,s2 in itertools.combinations(range(len(df_ranks.columns)),2):
            n = n + 1
            sl1 = df_ranks.iloc[:,s1]
            sl2 = df_ranks.iloc[:,s2]
            tt = stats.ttest_rel(sl1,sl2)
            r = stats.pearsonr(sl1,sl2)
            #print '{:2} {:16} {:16} {:12f} {:10f}'.format(n,sl1.name,sl2.name,r[1], r[0])
            #print r[1], r[0]
            m_s[s1][s2] = r[0]
            m_s[s2][s1] = r[0]    
            corr_list.append(r[0])
        
        #print corr_list[0], corr_list[1], corr_list[2]
    
        ph_rho.append(corr_list[0])
        pa_rho.append(corr_list[1])
        ha_rho.append(corr_list[2])
    
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
        
    print(np.mean(ph_corr), np.mean(pa_corr), np.mean(ha_corr))
    print(np.mean(ph_rho), np.mean(pa_rho), np.mean(ha_rho))
    mph_corr.append(ph_corr)
    mpa_corr.append(pa_corr)
    mha_corr.append(ha_corr)
    mph_rho.append(ph_rho)
    mpa_rho.append(pa_rho)
    mha_rho.append(ha_rho)
    
    nx.draw(G)
pd.DataFrame(np.array(mpa_rho).T).describe()