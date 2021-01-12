import networkx as nx
import pandas as pd
import glob
import re
import os
import os.path

def getGraph(adjlist):
    df = pd.read_csv(adjlist)

    G = nx.Graph()

    ants = df.columns.tolist()

    # TODO: Is there a shortcut for this?
    for ant in ants:
        G.add_node(ant)

    for index, row in df.iterrows():
        from_ant = ants[index]

        # I don't think it matters much which is the "from_ant" and which is the "to_ant"
        for to_ant in ants:
            encounters = row[to_ant]

            # If there were no encounters, there is no edge
            # This also "filters out" self loops
            if encounters > 0:
                G.add_edge(from_ant, to_ant, weight=encounters)

    return G

base = 'E:\\School stuff\\Lewis\\Research\\tracking_data\\tracking_data\\'

behavior = pd.read_csv(base + 'behavior.csv')

# Annoyingly enough, they're called colony 1, 2, 3, 4... in one place and 4, 18, 21, 29... in another
# TODO: Make sure that this is true - I've only checked the first one, and that wasn't 100% thorough
behavior['colony'] = behavior['colony'].replace(4, 1)
behavior['colony'] = behavior['colony'].replace(18, 2)
behavior['colony'] = behavior['colony'].replace(21, 3)
behavior['colony'] = behavior['colony'].replace(29, 4)
behavior['colony'] = behavior['colony'].replace(58, 5)
behavior['colony'] = behavior['colony'].replace(78, 6)

toWriteTo = pd.DataFrame(columns = ['Colony', 'Day', 'Ant', 'Hubness', 'Authority', 'HA Ratio', 'Classification'])

for adjlist in glob.glob(base + "*.txt"):
    matches = re.match(r'^network_col([0-9]+)_day([0-9]+).txt$', os.path.basename(adjlist))

    if matches is not None:
        m = matches.groups()

        colony = int(m[0])
        day = int(m[1])

        G = getGraph(adjlist)
        hubness, authority = nx.hits(G)

        for ant in hubness.keys():
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

                toWriteTo = toWriteTo.append({ 'Colony': colony, 'Day': day, 'Ant': ant, 'Hubness': hubness[ant], 'Authority': authority[ant], 'HA Ratio': hubness[ant] / authority[ant], 'Classification': role}, ignore_index = True)

toWriteTo.to_csv(base + 'AntsClassified.csv')