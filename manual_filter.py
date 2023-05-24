from numpy.random import uniform
import numpy as np
from utils import DataIndex, load_data, save_params

NUM_MOVIES = 10

NUM_PARAMS = 3
LEARNING_RATE = 0.1/(NUM_MOVIES**2)
REGULARISATION = 0.1
EPSILON = 1.0


def main():
    data = load_data(NUM_MOVIES)
    print("Data loaded")
    user_params, movie_params = initialise_params(data)
    print("Parameters initialised")

    initial_cost = cost(data, user_params, movie_params)
    current_cost = initial_cost
    print(f"Initial cost: {current_cost}")

    while True:
        user_params, movie_params = update(data, user_params, movie_params)
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

def update(data, user_params, movie_params):
    new_user_params = update_user_params(data, user_params, movie_params)
    new_movie_params = update_movie_params(data, user_params, movie_params)
    return new_user_params,  new_movie_params


def update_user_params(data, user_params, movie_params):
    return {
        user: current_params - LEARNING_RATE * user_jacobian(data, user, current_params, movie_params)
        for user, current_params in user_params.items()
    }


def user_jacobian(data, user, current_params, movie_params, reg=REGULARISATION):
    diff = sum([
        (np.dot(current_params, movie_params[movie]) - data.ratings[(user, movie)]) * movie_params[movie]
        for movie in data.user_movies[user]
    ])
    reg_diff = reg * current_params
    return diff + reg_diff


def update_movie_params(data, user_params, movie_params):
    return {
        movie: current_params - LEARNING_RATE * movie_jacobian(data, movie, current_params, user_params)
        for movie, current_params in movie_params.items()
    }


def movie_jacobian(data, movie, current_params, user_params, reg=REGULARISATION):
    zero_bias_change = np.array([1]*NUM_PARAMS + [0])
    diff = sum([
        (np.dot(current_params, user_params[user]) - data.ratings[(user, movie)]) * (user_params[user] * zero_bias_change)
        for user in data.movie_users[movie]
    ])
    reg_diff = reg * current_params
    return diff + reg_diff


if __name__ == "__main__":
    main()
