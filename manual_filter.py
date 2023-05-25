from numpy.random import uniform
import numpy as np
from utils import DataIndex, load_data, save_params, Updater

NUM_MOVIES = 50

NUM_PARAMS = 3
LEARNING_RATE = 0.00004
REGULARISATION = 0.1
EPSILON = 1.0
MOMENTUM = 0.7


def main():
    data = load_data(NUM_MOVIES)
    print("Data loaded")
    user_params, movie_params = initialise_params(data)
    print("Parameters initialised")

    initial_cost = cost(data, user_params, movie_params)
    current_cost = initial_cost
    print(f"Initial cost: {current_cost}")

    updater = Updater(LEARNING_RATE, MOMENTUM, REGULARISATION, NUM_PARAMS)
    while True:
        user_params, movie_params = updater.update(data, user_params, movie_params)
        new_cost = cost(data, user_params, movie_params)
        cost_reduction = current_cost - new_cost
        assert cost_reduction > - (initial_cost * 10), "Cost should be reducing"
        if cost_reduction < EPSILON:
            break
        current_cost = new_cost
        print(current_cost)
    print("Training done. Saving params...")
    save_params(data.ratings_mean, user_params, movie_params)


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


if __name__ == "__main__":
    main()
