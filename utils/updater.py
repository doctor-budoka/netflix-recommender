import numpy as np

class Updater:
    def __init__(self, learning_rate, momentum, regularisation, num_params):
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._regularisation = regularisation
        self.num_params = num_params
        self._previous_user_update = 0
        self._previous_movie_update = 0

    def update(self, data, user_params, movie_params):
        new_user_params = self.update_user_params(data, user_params, movie_params)
        new_movie_params = self.update_movie_params(data, user_params, movie_params)
        return new_user_params,  new_movie_params


    def update_user_params(self, data, user_params, movie_params):
        return {
            user: (
                current_params - self._learning_rate * self.user_jacobian(data, user, current_params, movie_params) 
                - self._momentum * self._previous_user_update
            )
            for user, current_params in user_params.items()
        }


    def user_jacobian(self, data, user, current_params, movie_params):
        user_movies = data.user_movies[user]
        diff = sum([
            (np.dot(current_params, movie_params[movie]) - data.ratings[(user, movie)]) * movie_params[movie]
            for movie in user_movies
        ])
        reg_diff = self._regularisation * current_params
        return diff + reg_diff


    def update_movie_params(self, data, user_params, movie_params):
        return {
            movie: (
                current_params - self._learning_rate * self.movie_jacobian(data, movie, current_params, user_params) 
                - self._momentum * self._previous_user_update
            )
            for movie, current_params in movie_params.items()
        }


    def movie_jacobian(self, data, movie, current_params, user_params):
        movie_users = data.movie_users[movie]
        zero_bias_change = np.array([1]*self.num_params + [0])
        diff = sum([
            (np.dot(current_params, user_params[user]) - data.ratings[(user, movie)]) * (user_params[user] * zero_bias_change)
            for user in movie_users
        ])
        reg_diff = self._regularisation * current_params
        return diff + reg_diff
