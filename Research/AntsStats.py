import pandas as pd
from scipy import stats
import glob
import re

data = {}

for file in glob.glob("Colony*.csv"):
    m = re.match('^Colony ([0-9]+) day ([0-9]+).csv$', file).groups()
    colony = m[0]
    day = m[1]

    if colony not in data.keys():
        data[colony] = {}

    if day not in data[colony].keys():
        data[colony][day] = {}

    df = pd.read_csv(file)

    for index, row in df.iterrows():
        ant = row['Unnamed: 0']

        hubness = row['HubScore']

        authority = row['Authority']

        haRatio = float(hubness) / float(authority)

        data[colony][day][ant] = haRatio
