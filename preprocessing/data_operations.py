import numpy as np
import pandas as pd


def preprocess_timestamp(data):
    """
    Converting timestamp to day

    :param data: pd.DataFrame
                users actions data set contains column with timestamp
    :return: pd.DataFrame
            column dataframe to day formate yyyy-mm-dd
    """

    data['date'] = pd.to_datetime(data.timestamp, unit='s')
    data['day'] = data.date.dt.date
    return data


def create_interaction(events, submissions):
    """
    Unity all data interaction

    :param events: pd.DataFrame
                Users events on course
    :param submissions: pd.DataFrame
                Users submissions on practical tasks
    :return: pd.DataFrame
    """

    interact_train = pd.concat([events, submissions.rename(columns={'submission_status': 'action'})])
    interact_train.action = pd.Categorical(interact_train.action,
                                           ['discovered', 'viewed', 'started_attempt',
                                            'wrong', 'passed', 'correct'], ordered=True)
    interact_train = interact_train.sort_values(['user_id', 'timestamp', 'action'])
    return interact_train


def create_user_data(events, submissions):
    """
    Creating final users data for fitting to model

    :param events: pd.DataFrame
                Users events on course
    :param submissions: pd.DataFrame
                Users submissions on practical tasks
    :return: pd.DataFrame
            Users data with all counted actions
    """

    users_data = events.groupby('user_id', as_index=False) \
        .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})

    # Number of correct and wrong answers
    users_scores = submissions.pivot_table(index='user_id',
                                           columns='submission_status',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0).reset_index()
    users_data = users_data.merge(users_scores, on='user_id', how='outer')
    users_data = users_data.fillna(0)

    # Number of actions for each user
    users_events_data = events.pivot_table(index='user_id',
                                           columns='action',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0).reset_index()
    users_data = users_data.merge(users_events_data, how='outer')

    # Number of days at course
    users_days = events.groupby('user_id').day.nunique().to_frame().reset_index()
    users_data = users_data.merge(users_days, how='outer')
    user_data = users_data.set_index('user_id')
    users_data = users_data.drop(columns=['last_timestamp'])
    users_data = users_data.set_index('user_id')

    return users_data


def get_y(events, submissions, points_threshold):
    """
    Creating final labels for fitting to model

    :param events: pd.DataFrame
                Users events on course
    :param submissions: pd.DataFrame
                Users submissions on practical tasks
    :param points_threshold: Number of points to pass the course
    :return: pd.Series with labels for X
    """
    interactions = create_interaction(events, submissions)
    users_data = interactions[['user_id']].drop_duplicates()

    passed_steps = (interactions[interactions['action'] == 'correct']
                    .groupby('user_id', as_index=False)['step_id']
                    .apply(lambda a: len(np.unique(a)))
                    .rename(columns={'step_id': 'correct'}))
    users_data = users_data.merge(passed_steps, how='outer')

    # New column is course passed
    users_data['is_gone'] = users_data['correct'] > points_threshold

    users_data = (users_data.drop('correct', axis=1)
                  .set_index('user_id'))
    return users_data['is_gone']


def trunc_data_by_nday(data, n_day):
    """
    Take n first days of each user activity

    :param data: pd.DataFrame with users data
    :param n_day: number of days to take
    :return: pd.DataFrame
            n first days for each user
    """

    users_min_time = data.groupby('user_id', as_index=False).agg({'timestamp': 'min'}).rename(
        {'timestamp': 'min_timestamp'}, axis=1)
    users_min_time['min_timestamp'] += 60 * 60 * 24 * n_day

    events_data_d = pd.merge(data, users_min_time, how='inner', on='user_id')
    cond = events_data_d['timestamp'] <= events_data_d['min_timestamp']
    events_data_d = events_data_d[cond]

    return events_data_d.drop(['min_timestamp'], axis=1)


