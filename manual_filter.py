from numpy import random
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
NUM_MOVIES = 100

NUM_PARAMS = 3
LEARNING_RATE = 0.1
EPSILON = 0.1


class DataIndex:
    def __init__(self):
        self.ratings = {}
        self.movie_users = {}
        self.user_movies = {}
        self._num_users = None
        self._num_movies = None
        self._num_ratings = None
    
    def add_rating(self, user, movie, rating):
        self.ratings[(user, movie)] = rating
        if movie not in self.movie_users:
            self.movie_users[movie] = set()
        self.movie_users[movie].add(user)
        if user not in self.user_movies:
            self.user_movies[user] = set()
        self.user_movies[user].add(movie)
        self._num_users = None
        self._num_movies = None
        self._num_ratings = None


    def get_users_for_movie(self, movie):
        return self.movie_users[movie]
    
    def get_movies_for_user(self, user):
        return self.user_movies[user]
    
    def get_ratings(self, user, movie):
        return self.ratings[(user, movie)]
    
    @property
    def num_users(self):
        if self._num_users is None:
            self._num_users = len(self.user_movies.keys())
        return self._num_users
    
    @property
    def num_movies(self):
        if self._num_movies is None:
            self._num_movies = len(self.movie_users.keys())
        return self._num_movies
    
    @property
    def num_ratings(self):
        if self._num_ratings is None:
            self._num_ratings = len(self.ratings.keys())
        return self._num_ratings


def main():
    data = load_data()
    print(data.num_users, data.num_movies, data.num_ratings)
    # user_params, movie_params = initialise_params(data)
    # current_cost = cost(data, user_params, movie_params)
    # while True:
    #     user_params, movie_params = update(data, user_params, movie_params)
    #     new_cost = cost(data, user_params, movie_params)
    #     if abs(new_cost - current_cost) < EPSILON:
    #         break
    #     current_cost = new_cost


def load_data():
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
            if num_movies > NUM_MOVIES:
                break
        else:
            continue
        break
    return data



def initialise_params(users, movies):
    return {x: [0, 0, 0] for x in users}, {x: [0, 0, 0] for x in movies}


def cost(ratings, user_params, movie_params):
    return 0

def update(ratings,user_params, movie_params):
    return user_params,  movie_params

def dcdb(j, ratings, user_params, movie_params):
    return 0

def dcdw(i, j, ratings, user_params, movie_params):
    return 0

def dcdx(i, k, ratings, user_params, movie_params):
    return 0


if __name__ == "__main__":
    main()
