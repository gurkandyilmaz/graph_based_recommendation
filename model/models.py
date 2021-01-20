import time
import math
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


class ProcessData():
    def __init__(self):
        pass

    def read_file(self, file: str):
        """Takes a filename(excel or csv), Returns a pandas dataframe object."""
        if file.name.rsplit(sep=".")[-1] in ["xlsx", "xls"]:
            df = pd.read_excel(file)
        elif file.name.rsplit(sep=".")[-1] in ["csv"]:
            df = pd.read_csv(file)
    
        return df

    def transform_text_data(self, corpus: list):
        """Takes a list of tokens, Returns tf-idf vectors and the transformer object used in vectorization."""
        pipe = Pipeline([
                    ("count", CountVectorizer(
                                ngram_range=(1,1), 
                                token_pattern=r"[a-zA-Z0-9ıIiİğĞçÇöÖüÜşŞ]+", 
                                strip_accents="unicode",
                                lowercase=True)),
                    ("tfidf", TfidfTransformer())
                ])
        tfidf = pipe.fit_transform(corpus) 
        
        return tfidf

    def compute_similarity(self, first: list, second: list):
        """Takes two array of numbers, Returns cosine similarity score between these two arrays."""
        from numpy.linalg import norm
        similarity_score = np.dot(first, second)/(norm(first)*norm(second))

        return similarity_score


class BuildGraph():
    def __init__(self):
        pass
    
    def build_graph(self, dataframe, tfidf):
        """Takes a dataframe and a tfidf matrix, Returns a Graph object."""        
        graph = networkx.Graph(label="item")
        start_time = time.time()
        for i, rowi in dataframe.iterrows():
            if (i%1000 == 0):
                print(" iter {} -- {} seconds --".format(i, time.time()-start_time))
            graph.add_node(rowi["ITEM_NAME"], key=rowi["ITEM"], label="ITEM_NAME")
            graph.add_node(rowi["USER"], label="USER")
            graph.add_edge(rowi["ITEM_NAME"], rowi["USER"], label="Purchased")

            # graph.add_node(rowi["Clusters"], label="CLUSTERS")    
            # graph.add_edge(rowi["ITEM_NAME"], rowi["Clusters"], label="Belongs_To")

            s_node = "Similar ({})".format(rowi["ITEM_NAME"][:25].strip())
            graph.add_node(s_node, label="SIMILAR")
            graph.add_edge(rowi["ITEM_NAME"], s_node, label="Similarity")
            indices = self._find_similar(tfidf, i, top_n=25)
            for idx in indices:
                graph.add_edge(s_node, dataframe["ITEM_NAME"].loc[idx], label="SIMILARITY")
        print(" finish -- {} seconds --".format(time.time() - start_time))

        return graph
    
    def _find_similar(self, tfidf_matrix, idx: int, top_n=25):
        """Takes a 2D array and an index, Returns similar arrays for that index."""
        from sklearn.metrics.pairwise import linear_kernel
        cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        #sort the results in an ascending order, reverse it then ignore the first element at idx, take the rest top_n
        top_n_similar = cosine_similarities.argsort()[-2:-(top_n+2):-1].tolist()

        return top_n_similar

         
def get_recommendation(graph, node: str):
    """"Takes a Graph object and a root node, Returns a Series object containin Recommendations for that node."""
    common_items = dict(list())
    for n1 in graph.neighbors(node):
        for n2 in graph.neighbors(n1):
            if n2 == node:
                continue
            if graph.nodes[n2]['label']=="ITEM_NAME":
                if common_items.get(n2, -1) == -1:
                    common_items.update({n2: [n1]})
                elif common_items.get(n2, -1):
                    common_items[n2].append(n1)
    items=[]
    weight=[]
    for key, values in common_items.items():
        w=0.0
        for e in values:
            # Adamic Adar measure
            w += 1/math.log(graph.degree(e))
        items.append(key) 
        weight.append(w)
    
    result = pd.Series(data=np.array(weight),index=items)
    result.sort_values(inplace=True, ascending=False)

    return result

def recommend(graph, node: list):
    import random
    # random.seed(100)
    item_and_similar = {}
    results = []
    for item in node:
        item_and_similar.update({item: get_recommendation(graph, item)})

    for similar_items in item_and_similar.values():
        results.extend(similar_items.index[:10].values)
    
    return random.sample(set(results), 10)

def optional_clustering(dataframe, data_to_be_clustered, n_cluster=200):
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_cluster, random_state=10)
    kmeans.fit(data_to_be_clustered)
    dataframe["Clusters"] = kmeans.predict(data_to_be_clustered)

    return dataframe