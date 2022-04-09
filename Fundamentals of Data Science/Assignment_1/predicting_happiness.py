import os
from textwrap import wrap

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, RandomizedSearchCV
from tqdm import tqdm

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


def k_fold_evaluation(speeches_df, model, n_splits, features, predictable):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    i = 0

    r2_scores = []
    mse_scores = []

    for train, test in kf.split(speeches_df):
        X_train = speeches_df.iloc[train][features]
        y_train = speeches_df.iloc[train][predictable]

        X_test = speeches_df.iloc[test][features]
        y_test = speeches_df.iloc[test][predictable]

        model.fit(X_train, y_train, eval_metric='r2', eval_set=[(speeches_df.iloc[test][features],
                                                                speeches_df.iloc[test][predictable]),
                             (speeches_df.iloc[train][features], speeches_df.iloc[train][predictable])],
                   early_stopping_rounds=100, verbose=100)

        expected_y = y_test
        predicted_y = lgbm_model.predict(X_test)

        r2_scores.append(metrics.r2_score(expected_y, predicted_y))
        mse_scores.append(metrics.mean_squared_error(expected_y, predicted_y))

        i += 1

    print(r'The average r2 score obtained through k-fold validation +- std: {} +- {}'.format(np.mean(r2_scores),
                                                                                  np.std(r2_scores)))
    print(r'The average mse score obtained through k-fold validation +- std: {} +- {}'.format(np.mean(mse_scores), np.std(mse_scores)))



def model_gridsearch(speeches_df, model, n_splits, gridParams, features, predictable):
    grid = RandomizedSearchCV(model, gridParams, cv=n_splits, n_jobs=-1, n_iter=5000, verbose=0)

    grid.fit(speeches_df[features], speeches_df[predictable])
    print('Best parameters: ', grid.best_params_)

    return grid.best_params_

def test_set_evaluation(speeches_df_train, speeches_df_test, model_test, features, predictable):

    # we split the training set so that we can prevent overfitting of the model
    speeches_df_train, speeches_df_validation = train_test_split(speeches_df_train, test_size=0.2, shuffle=True,
                                                                 random_state=42)

    model_test.fit(speeches_df_train[features], speeches_df_train[predictable],
                   eval_set=[(speeches_df_validation[features], speeches_df_validation[predictable]),
                             (speeches_df_train[features], speeches_df_train[predictable])],
                   early_stopping_rounds=300, eval_metric='r2', verbose=100)


    expected_y = speeches_df_test[predictable]
    predicted_y = model_test.predict(speeches_df_test[features])


    print('Plot metrics during training...')
    ax = lgb.plot_metric(model_test.evals_result_, metric='l2', figsize=(10, 5), title=None)
    ax.lines[0].set_linewidth(5)
    ax.lines[1].set_linewidth(5)
    ax.set_ylabel('L2', fontsize=25)
    ax.set_xlabel('Iterations', fontsize=25)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.legend(['Validation set', 'Training set'], fontsize=25)
    plt.tight_layout()
    plt.savefig('metric_over_training.png', dpi=300)
    plt.show()

    plt.figure()
    ind = np.arange(0.5, 5.5, 0.5)
    feature_imp = pd.DataFrame(sorted(zip(model_test.feature_importances_, speeches_df_train[features].columns)), columns=['Value', 'Feature'])
    print(feature_imp)

    # ax1 = lgb.plot_importance(model_test, max_num_features=10, figsize=(15,20), title=None, annotate=False)
    ax1 = feature_imp.iloc[-10:].plot.barh(x="Feature", y="Value")
    ticklabels = feature_imp.iloc[-10:]['Feature'].tolist()
    ticklabels = ['\n'.join(wrap(l, 13)) for l in ticklabels]
    ax1.get_legend().remove()
    ax1.set_yticklabels(ticklabels)
    ax1.set_ylabel('Features', fontsize=20)
    ax1.set_xlabel('Feature importance', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()
    print('the feature importances: ', model_test.feature_importances_)


    print('The R2 score on the test set: ', metrics.r2_score(expected_y, predicted_y))
    print('The MSE score on the test set: ', metrics.mean_squared_error(expected_y, predicted_y))


# load the dataframe that was previously preprocessed
speeches_df = pd.read_pickle('preprocessed_dataframe_top10.pkl')
speeches_df.dropna(subset=['Life Ladder'], inplace=True)

most_used_words = set()
for i, row in speeches_df.iterrows():
    most_used_words |= set(row['top10words'])

features_most_used_words = [word.encode("ascii", "ignore").decode() for word in list(most_used_words)]

print(features_most_used_words)
# print(most_used_words)
speeches_df[features_most_used_words] = 0

for i, row in tqdm(speeches_df.iterrows()):
    for word in row['top10words']:
        speeches_df.loc[i, word.encode("ascii", "ignore").decode()] = 1

print(speeches_df.head())

speeches_df.to_csv('speeches_df_1hotencoded_top10words.csv')

speeches_df_train, speeches_df_test = train_test_split(speeches_df, test_size=0.2, shuffle=True, random_state=42)

# "Log GDP per capita", 'Social support', 'Freedom to make '
#                  'life choices', 'Generosity', 'Perceptions of corruption'

selected_words = []
for possible_feature in features_most_used_words:
    if speeches_df[possible_feature].sum(axis=0) > 20:
        selected_words.append(possible_feature)

print('the number of words we use in the end:', len(selected_words))

# set the list of features to use as well as the predictable
features = ['year', 'word_count', 'pos_sentiment', 'neg_sentiment', 'neu_sentiment', 'average_sentence_length'] + \
           selected_words
predictable = ['Life Ladder']

# set the parameters that will be tested in the grid search
lgbm_grid_params = {
    'learning_rate': [0.005, 0.01, 0.1],
    'n_estimators': [16, 24, 28, 500, 1000, 2000, 3000],
    'min_child_samples': [1, 10, 50, 100],
    'num_leaves': [4, 8, 12, 16, 20],  # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type': ['gbdt'],  # for better accuracy -> try dart
    'max_bin': [255, 510, 1020],  # large max_bin helps improve accuracy but might slow down training progress
    'random_state': [42],
    'colsample_bytree': [0.64, 0.65, 0.8, 1],
    'subsample': [0.5, 0.7, 0.75, 0.8, 0.9, 1],
    'reg_alpha': [0, 0.25, 0.5, 0.75, 1, 1.2],
    'reg_lambda': [0, 0.25, 0.5, 0.75, 1, 1.2, 1.4],
    'verbose': [-1]
}


# create the models
lgbm_model = lgb.LGBMRegressor()
n_splits = 5

# evaluation before optimisation
k_fold_evaluation(speeches_df_train, lgbm_model, n_splits, features, predictable)

test_set_evaluation(speeches_df_train, speeches_df_test, lgbm_model, features, predictable)

# grid search for the best parameters
# best_parameters = model_gridsearch(speeches_df_train, lgbm_model, n_splits, lgbm_grid_params, features, predictable)
best_parameters =  {'verbose': -1, 'subsample': 1, 'reg_lambda': 0, 'reg_alpha': 0, 'random_state': 42, 'num_leaves':
    12, 'n_estimators': 2000, 'min_child_samples': 1, 'max_bin': 255, 'learning_rate': 0.01, 'colsample_bytree': 0.64,
                    'boosting_type': 'gbdt'}
# {'verbose': -1, 'subsample': 1, 'reg_lambda': 0, 'reg_alpha': 0, 'random_state': 42, 'num_leaves': 12, 'n_estimators': 500, 'min_child_samples': 1, 'max_bin': 255, 'learning_rate': 0.01, 'colsample_bytree': 0.64, 'boosting_type': 'gbdt'}

# look at increase of scores
lgbm_model_optimised = lgb.LGBMRegressor(**best_parameters)
k_fold_evaluation(speeches_df_train, lgbm_model_optimised, n_splits, features, predictable)

# final evalutation on unseen test set
lgbm_model_optimised_test = lgb.LGBMRegressor(**best_parameters)
test_set_evaluation(speeches_df_train, speeches_df_test, lgbm_model_optimised_test, features, predictable)



