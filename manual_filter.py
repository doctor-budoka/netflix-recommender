from numpy import random

NUM_PARAMS = 3
LEARNING_RATE = 0.1
EPSILON = 0.1

def main():
    ratings, users, movies = load_data()
    user_params, movie_params = initialise_params(users, movies)
    current_cost = cost(ratings, user_params, movie_params)
    while True:
        user_params, movie_params = update(ratings, user_params, movie_params)
        new_cost = cost(ratings, user_params, movie_params)
        if abs(new_cost - current_cost) < EPSILON:
            break
        current_cost = new_cost



def load_data():
    return None


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
