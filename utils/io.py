
from pathlib import Path
import pandas as pd
from utils.data_index import DataIndex

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT/ "outputs"

def load_data(max_movies):
    data = DataIndex()
    src_files = sorted([x for x in DATA.iterdir() if x.name.startswith("combined_data")])
    current_movie_id = None
    num_movies = 0
    for file in src_files:
        with open(file, "r") as in_f:
            in_lines = in_f.readlines()
        for line in in_lines:
            trimmed_line = line.strip()
            if trimmed_line.endswith(":"):
                current_movie_id = int(trimmed_line[:-1])
                num_movies += 1
            else:
                info = trimmed_line.split(",")
                user_id, rating = int(info[0]), int(info[1])
                data.add_rating(user_id, current_movie_id, rating)
            if num_movies > max_movies:
                break
        else:
            continue
        break
    return data.normalise()


def save_params(mean, user_params, movie_params):
    save_dict_params(OUTPUTS / "user_params.csv", user_params)
    save_dict_params(OUTPUTS / "movie_params.csv", movie_params)
    save_number_to_file(OUTPUTS / "mean.txt", mean)

def save_dict_params(path, array):
    num_params = len(next(iter(array.values())))
    pd.DataFrame.from_dict(array, orient="index", columns=[f"w_{i}" for i in range(num_params - 1)] + ["bias"]).to_csv(path, index=False)

def save_number_to_file(path, data):
    save_to_file(path, str(data))


def save_to_file(path, contents):
    with open(path, "w") as f:
        f.write(contents)
