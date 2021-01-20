from pathlib import Path
from datetime import datetime
import pickle

from flask import Flask, render_template, request, redirect, url_for
from . import app

from model.models import ProcessData, BuildGraph, get_recommendation, recommend
from utils.paths import get_project_folder

DATA_DIR = get_project_folder().parent / "data"
PICKLE_DIR = get_project_folder() / "model"

@app.route("/", methods=["GET", "POST"])
def main():
    with open(PICKLE_DIR / "graph.pkl", "rb") as graph_pickle:
        graph = pickle.load(graph_pickle)

    if request.method == "POST":
        user_query = request.form.to_dict().get("user_query")
        # for query in user_query.split("*"): 
        try:
            results = recommend(graph, user_query.split("*"))
            message = ""
        except:
            message = "The node {} is NOT in the graph.".format(user_query)
            results = []

    elif request.method == "GET":
        user_query = ""
        results = []
        message = "Make a prediction"

    return render_template("base.html", data=user_query, results=results, message=message, status_recommend="active")

@app.route("/train", methods=["GET", "POST"])
def train():
    if not Path(PICKLE_DIR / "graph.pkl").exists():
        message = "Train Finished"
        file_name = "generic_rec_dataset_Ekim_New_data.xlsx"
        file_path = DATA_DIR / file_name
        df = processor.read_file(file_path)

        item_title = df.loc[:, "ITEM_NAME"]
        tfidf = processor.transform_text_data(item_title)
        graph = graph_builder.build_graph(df, tfidf)
        
        with open(PICKLE_DIR / "graph.pkl", "wb") as graph_file:
            pickle.dump(graph, graph_file, pickle.HIGHEST_PROTOCOL)
    else:
        message = "Model was trained. Pickle file is at --> {}".format(PICKLE_DIR / "graph.pkl")
        with open(PICKLE_DIR / "graph.pkl", "rb") as graph_file:
            graph = pickle.load(graph_file)

    print("Graph object: ", type(graph))
    return render_template("train.html", status_train="active", message=message)

processor = ProcessData()
graph_builder = BuildGraph()

if __name__ == "__main__":
    app.run(debug=True)