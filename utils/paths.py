from pathlib import Path

def get_project_folder():
    """Returns the project directory '../graph_based_recommendation'"""
    # print("ROOT FOLDER in paths.py:", Path(__file__).parent.parent)
    return Path(__file__).parent.parent

# def get_visualization_folder():
#     # print("Visualization FOLDER in paths.py:, ", get_root_folder() / "visualization")
#     return get_root_folder() / "visualization"

# def get_data_folder():
#     return get_root_folder() / "visualization/data"

# if __name__ == "__main__":
#     print("Path: ", Path(__file__).parent)