from numpy.random import uniform
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
NUM_MOVIES = 100

NUM_PARAMS = 3
LEARNING_RATE = 0.1
REGULARISATION = 0.1
EPSILON = 0.1


class DataIndex:
    def __init__(self):
        self.ratings = {}
        self.movie_users = {}
        self.user_movies = {}
        self._users = None
        self._movies = None
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
    
    @property
    def users(self):
        if self._users is None:
            self._users = set(self.user_movies.keys())
        return self._users
    
    @property
    def movies(self):
        if self._movies is None:
            self._movies = set(self.movie_users.keys())
        return self._movies



def main():
    data = load_data()
    print("Data loaded")
    user_params, movie_params = initialise_params(data)
    print("Parameters initialised")

    current_cost = cost(data, user_params, movie_params)
    print(current_cost)

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



def initialise_params(data: DataIndex):
    user_init = uniform(low=-1, high=1, size=(data.num_users, NUM_PARAMS + 1))
    user_params = {x: row for x, row in zip(data.users, user_init)}

    movie_init = np.append(uniform(low=-1, high=1, size=(data.num_movies, NUM_PARAMS)), np.ones((data.num_movies, 1)), axis=1)
    movie_params = {x: row for x, row in zip(data.movies, movie_init)}
    return user_params, movie_params


def cost(data: DataIndex, user_params, movie_params, reg=REGULARISATION):
    pure_cost = calculate_pure_cost(data, user_params, movie_params)
    regularisation_term = calculate_regularisation_terms(user_params, movie_params, reg=reg)
    return pure_cost + regularisation_term

def calculate_pure_cost(data: DataIndex, user_params, movie_params):
    errors = [
        (np.dot(user_params[user], movie_params[movie]) - rating)**2
        for (user, movie), rating in data.ratings.items()
    ]
    return 0.5 * sum(errors)

def calculate_regularisation_terms(user_params, movie_params, reg=REGULARISATION):
    user_reg = sum([np.dot(params, params) for params in user_params.values()])
    movie_reg = sum([np.dot(params, params) for params in movie_params.values()])
    return 0.5 * reg * (user_reg + movie_reg)

def update(ratings, user_params, movie_params):
    return user_params,  movie_params

def dcdb(j, ratings, user_params, movie_params):
    return 0

def dcdw(i, j, ratings, user_params, movie_params):
    return 0

def dcdx(i, k, ratings, user_params, movie_params):
    return 0


if __name__ == "__main__":
    main()
