#import graphml as gp
import networkx as nx
import glob
import re
from GrabFiles import DownloadAntFiles

data = {}
graphs = {}

if len(glob.glob("*.graphml")) < (41 * 4):
    DownloadAntFiles()

for graphml in glob.glob("*.graphml"):
    m = re.match(r'^ant_mersch_col([0-9]+)_day([0-9]+)_attribute.graphml$', graphml).groups()

    if m[0] not in data.keys():
        data[m[0]] = []
        graphs[m[0]] = {}

    data[m[0]].append(m[1])

    nx.read_graphml(graphml)

    graphs[m[0]][m[1]] = nx.read_graphml(graphml)

print('Done')
    # ant_mersch_col1_day01_attribute.graphml
#nx.read_graphml()