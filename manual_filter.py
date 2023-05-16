from numpy import random

NUM_PARAMS = 3
LEARNING_RATE = 0.1
EPSILON = 0.1


class DataIndex:
    def __init__(self):
        self.ratings = {}
        self.movie_users = {}
        self.user_movies = {}
    
    def add_rating(self, user, movie, rating):
        self.ratings[(user, movie)] = rating
        if movie not in self.movie_users:
            self.movie_users[movie] = set()
        self.movie_users[movie].add(user)
        if user not in self.user_movies:
            self.user_movies[user] = set()
        self.user_movies[user].add(movie)

    def get_users_for_movie(self, movie):
        return self.movie_users[movie]
    
    def get_movies_for_user(self, user):
        return self.user_movies[user]
    
    def get_ratings(self, user, movie):
        return self.ratings[(user, movie)]
    

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
