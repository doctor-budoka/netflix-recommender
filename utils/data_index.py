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
        self.ratings_mean = None
    
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

    def normalise(self):
        self.ratings_mean = sum(self.ratings.values())/len(self.ratings.values())
        self.translate_ratings(-self.ratings_mean)
        return self

    def denormalise(self):
        if self.ratings_mean is None:
            raise ValueError("Can't denormalise because the data hasn't been normalised")
        self.translate_ratings(self.ratings_mean)
        return self
    
    def translate_ratings(self, distance):
        for key, value in self.ratings.items():
            self.ratings[key] = value - distance
