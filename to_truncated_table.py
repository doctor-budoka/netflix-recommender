from pathlib import Path
from pprint import pprint
import polars as pl

ROOT = Path(__file__).parent
DATA = ROOT / "data"
NUM_MOVIES = 100

def generate_csv():
    src_files = sorted([x for x in DATA.iterdir() if x.name.startswith("combined_data")])
    ratings = {}
    user_ids = set()
    this_movie_id = None
    for file in src_files:
        with open(file, "r") as in_f:
            in_lines = in_f.readlines()
        for line in in_lines:
            trimmed_line = line.strip()
            if trimmed_line.endswith(":"):
                this_movie_id = int(trimmed_line[:-1])
                ratings[this_movie_id] = []
            else:
                data = trimmed_line.split(",")
                user_id, rating = int(data[0]), int(data[1])
                ratings[this_movie_id].append((user_id, rating))
                user_ids.add(user_id)
            if len(ratings.keys()) > NUM_MOVIES:
                break
        else:
            continue
        break
    
    print(len(user_ids))


if __name__ == "__main__":
    generate_csv()
