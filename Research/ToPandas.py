import json
import pandas as pd

d = json.loads('Json.json')
df = pd.read_json('Json.json')

df.to_csv('Test.csv')