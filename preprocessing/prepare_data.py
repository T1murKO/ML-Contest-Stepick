import preprocessing.data_operations as do

import sys
import os

sys.path.append("..")
from config import POINTS_TO_PASS_COURSE, DAYS_THRESHOLD_TO_PREDICT_CHURN


def get_x_y(events, submissions):
    """

    :param events: pd.DataFrame
                Users events on course
    :param submissions: pd.DataFrame
                Users submissions on practical tasks
    :return: X features and y labels to fit the model
    """

    events_train = do.preprocess_timestamp(events)
    events_train = do.trunc_data_by_nday(events_train, DAYS_THRESHOLD_TO_PREDICT_CHURN)

    submissions_train = do.preprocess_timestamp(submissions)
    submissions_train = do.trunc_data_by_nday(submissions_train, DAYS_THRESHOLD_TO_PREDICT_CHURN)

    X = do.create_user_data(events_train, submissions_train)
    y = do.get_y(events, submissions, POINTS_TO_PASS_COURSE)

    X = X.sort_index()
    y = y.sort_index()

    return X, y
