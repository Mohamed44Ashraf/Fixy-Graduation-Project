import os

original_features = [ 'UserAvgRating', 'WorkerAvgRating',
       'UserRatingStd', 'UserWorkerAvg', 'FeedbackCount', 'WorkerRatingVar',
       'RatingDeviation', 'predictedRating']


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")