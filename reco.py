import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sys, os

os.chdir('/Users/saad/Documents/YourFashion/')

def dummy_data():
    data = {
        0: {
            'id' : 0,
            'category': 'round neck t-shirt', 
            'color' : 'white', 
            'material' : 'cotton', 
            'source' : 'Zara', 
            'price' : 6.99
        }
    }

    for i in range(100):
        c = {
            'id' : i+1,
            'category' : random.choice(['round neck t-shirt', 'half-sleeve button-down shirt', 'v-neck t-shirt', 'button-down shirt', 'polo shirt', 'joggers', 'cargo pants', 'chinos', 'jeans', 'boxers', 'underwear', 'long socks', 'short socks', 'ankle socks']),
            'color' : random.choice(['white','black', 'grey', 'dark grey', 'red', 'blue', 'light blue', 'olive green', 'salmon pink', 'light grey', 'orange', 'cream']),
            'material' : random.choice(['cotton','elastene', 'polyester', 'rayon', 'nylon', 'acrylic', 'linen', 'wool']),
            'source' : random.choice(['Zara', 'HnM', 'PullNBear', 'Bershka', 'Nike', 'Adidas', 'Reebok', 'Puma']),
            'price' : float(f'{random.randrange(3,19)}.99')
        }
        data[i+1] = c
    
    return data

df = pd.DataFrame(dummy_data()).T
idf = pd.DataFrame(columns=list(dummy_data().keys()))
idf.loc[0] = [0] * len(idf.columns)

r = True
if r:
    for i in range(len(idf.columns)):
        idf.iloc[0][i] = int(round(np.random.normal(loc=-1,scale=3),0))
df['interaction'] = idf.T

class recommendation_engine:

    def __init__(self, df, idf):
        self.df = df
        self.idf = idf
        
        self.n = 30
        self.encoded_data = None
        self.svd = None
        self.interactions_m = None
        self.similarity_m = None
        self.top_n_items = None
        self.reco()
        
    def reco(self):
        columns = self.df.drop(columns=['id']).columns.tolist()
        transformer = ColumnTransformer(
            transformers=[('onehot', OneHotEncoder(), columns)],
            remainder='passthrough'
        )
        ed = transformer.fit_transform(df) # encoded data
        ### SVD
        svd = TruncatedSVD(n_components=50)
        edsvd = svd.fit_transform(ed)
        ### Similarity Scores
        im = idf.values[0] # interactions matrix
        self.interactions_m = im
        sm = cosine_similarity(edsvd) # similarity matrix
        self.similarity_m = sm
        user_item_scores = im.dot(sm)
        user_item_ranks = (-user_item_scores).argsort()
        top_n_items = self.df.iloc[user_item_ranks[:self.n]]
        self.top_n_items = top_n_items
    
    def update(self, item_id, score):
        if score == 1:
            self.idf[item_id][0] += score
            self.df['interactions'] = self.idf
            self.reco()
        elif score == 0:
            self.idf[item_id][0] -= 1
            self.df['interactions'] = self.idf
            self.reco()

##testing environment

engine = recommendation_engine(df, idf)
top_n_dict = engine.top_n_items.T.to_dict()
for i in top_n_dict.keys():
    top_n_dict = engine.top_n_items.T.to_dict()
    curr = top_n_dict[i]
    print(f"{curr['id']}: {curr['source']} {curr['color']} {curr['category']} for {curr['price']} ({idf[curr['id']][0]})")
    user_input = int(input("1: yes | 0: no"))
    if user_input == 1:
        engine.update(curr['id'], 1)
        engine.reco()
    if user_input == 0:
        engine.update(curr['id'], -1)
        engine.reco()