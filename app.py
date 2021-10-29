from flask import Flask, request, jsonify, send_file
from flask_restx import Api, Resource
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, plot_confusion_matrix
import pickle
import os
import shutil
import warnings
from config import HOST, PORT

matplotlib.use('Agg')
warnings.filterwarnings('ignore')


app = Flask(__name__)
app.config['ERROR_404_HELP'] = False
api = Api(app)

shutil.rmtree('./data')
shutil.rmtree('./models')
os.makedirs('./data')
os.makedirs('./models')

model2task = {'LogisticRegression': 'classification',
              'SVC': 'classification',
              'DecisionTreeClassifier': 'classification',
              'RandomForestClassifier': 'classification',
              'Ridge': 'regression',
              'SVR': 'regression',
              'DecisionTreeRegressor': 'regression',
              'RandomForestRegressor': 'regression'}

model2grid = {'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100],
                                     'solver': ['newton-cg', 'lbfgs', 'sag'],
                                     'penalty': ['l2', 'none']},
              'SVC': {'C': [0.01, 0.1, 1, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'degree': [3, 5, 8]},
              'DecisionTreeClassifier': {'max_depth': [5, 10, 20, None],
                                         'min_samples_split': [2, 5, 10],
                                         'min_samples_leaf': [1, 2, 5],
                                         'max_features': ['sqrt', 'log2', None]},
              'RandomForestClassifier': {'n_estimators': [100, 200, 300, 400, 500],
                                         'max_depth': [5, 10, 20, None],
                                         'min_samples_split': [2, 5, 10],
                                         'min_samples_leaf': [1, 2, 5],
                                         'max_features': ['sqrt', 'log2', None]},
              'Ridge': {'alpha': np.linspace(0, 1, 11),
                        'fit_intercept': [True, False],
                        'solver': ['svd', 'lsqr', 'sag']},
              'SVR': {'C': [0.01, 0.1, 1, 10, 100],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'degree': [3, 5, 8]},
              'DecisionTreeRegressor': {'max_depth': [5, 10, 20, None],
                                        'min_samples_split': [2, 5, 10],
                                        'min_samples_leaf': [1, 2, 5],
                                        'max_features': ['sqrt', 'log2', None]},
              'RandomForestRegressor': {'n_estimators': [100, 200, 300, 400, 500],
                                        'max_depth': [5, 10, 20, None],
                                        'min_samples_split': [2, 5, 10],
                                        'min_samples_leaf': [1, 2, 5],
                                        'max_features': ['sqrt', 'log2', None]}}


def calculate_metrics(model, estimator, X_train, X_test, y_train, y_test):
    metrics = {}
    if model2task[model] == 'classification':
        for x in ['train', 'test']:
            y_true = eval('y_' + x)
            y_pred = estimator.predict(eval('X_' + x))
            y_pred_proba = estimator.predict_proba(eval('X_' + x))
            metrics[x] = {}
            metrics[x]['accuracy'] = np.round(accuracy_score(y_true, y_pred), 4)
            average = 'binary' if len(set(y_true)) == 2 else 'macro'
            metrics[x]['precision'] = np.round(precision_score(y_true, y_pred, average=average), 4)
            metrics[x]['recall'] = np.round(recall_score(y_true, y_pred, average=average), 4)
            metrics[x]['f1_score'] = np.round(f1_score(y_true, y_pred, average=average), 4)
            metrics[x]['auc_roc'] = np.round(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'), 4)
    else:
        for x in ['train', 'test']:
            y_true = eval('y_' + x)
            y_pred = estimator.predict(eval('X_' + x))
            metrics[x] = {}
            metrics[x]['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
            metrics[x]['mae'] = mean_absolute_error(y_true, y_pred)
    return metrics


class MLModelsDAO:
    def __init__(self):
        self.ml_models = ['LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
                          'Ridge', 'SVR', 'DecisionTreeRegressor', 'RandomForestRegressor']
        self.ml_models_all = {}
        self.counter = 0  # total number of fitted models

    def get(self, model_id):
        """
        Return predictions of the given model for the train and test set.
        Abort if the model was deleted.
        """
        if model_id not in self.ml_models_all:
            api.abort(404, f'Model {model_id} does not exist')
        if self.ml_models_all[model_id]['deleted']:
            api.abort(404, f'Model {model_id} was deleted')

        with open(f'models/{model_id}.pkl', 'rb') as f:
            estimator = pickle.load(f)
        with open(f'data/{model_id}_train.pkl', 'rb') as f:
            train = pickle.load(f)
        with open(f'data/{model_id}_test.pkl', 'rb') as f:
            test = pickle.load(f)

        train_pred = list(np.round(estimator.predict(train.iloc[:, :-1]), 2))
        test_pred = list(np.round(estimator.predict(test.iloc[:, :-1]), 2))

        return jsonify({'train_predictions': str(train_pred),
                        'test_predictions': str(test_pred)})

    def create(self, data, model, params, grid_search, param_grid):
        """
        Train the given model with given parameters on the given data (json).
        If parameters is not set, use default values.
        Once the model is trained, append model name, parameters and performance metrics
        to models_dao.ml_models_all dictionary.
        """
        if model not in self.ml_models:
            api.abort(404, f'Can only train one of {self.ml_models} models')

        if params == 'default':
            params = {}
        if param_grid == 'default':
            param_grid = model2grid[model]

        df = pd.read_json(data)
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        if any(X.dtypes == object):
            api.abort(400, 'Could not support categorical features')
        if y.dtype == object and model2task[model] == 'regression':
            api.abort(400, f'{model} can only be used for regression tasks')
        elif y.dtype == float and model2task[model] == 'classification':
            api.abort(400, f'{model} can only be used for classification tasks')

        if model == 'SVR' or model == 'SVC':
            params['probability'] = True

        try:
            estimator = eval(model + '(**params)')
        except TypeError as err:
            api.abort(400, f'{model} got an unexpected keyword argument {str(err).split()[-1]}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if grid_search:
            try:
                gs = GridSearchCV(estimator, param_grid, cv=3, n_jobs=-1, verbose=2)
                gs.fit(X_train, y_train)
                estimator = gs.best_estimator_
            except ValueError as err:
                api.abort(400, err)

        estimator.fit(X_train, y_train)
        metrics = calculate_metrics(model, estimator, X_train, X_test, y_train, y_test)

        self.counter += 1
        with open(f'models/{self.counter}.pkl', 'wb') as f:
            pickle.dump(estimator, f)
        with open(f'data/{self.counter}_train.pkl', 'wb') as f:
            pickle.dump(df.loc[X_train.index], f)
        with open(f'data/{self.counter}_test.pkl', 'wb') as f:
            pickle.dump(df.loc[X_test.index], f)

        self.ml_models_all[self.counter] = {'model': model, 'params': estimator.get_params(), 'metrics': metrics,
                                            'retrained': False, 'deleted': False}

    def update(self, model_id, data):
        """
        Return predictions for the given model.
        Abort if the model was deleted or a new data consists of another columns.
        """
        if model_id not in self.ml_models_all:
            api.abort(404, f'Model {model_id} does not exist')
        if self.ml_models_all[model_id]['deleted']:
            api.abort(404, f'Model {model_id} was deleted')

        train_new = pd.read_json(data)
        with open(f'data/{model_id}_train.pkl', 'rb') as f:
            train_old = pickle.load(f)
        with open(f'data/{model_id}_test.pkl', 'rb') as f:
            test = pickle.load(f)
        if any(train_new.columns != train_old.columns) or any(train_new.dtypes != train_old.dtypes):
            api.abort(400, f'New data must include the same columns')

        X_train, y_train = train_new.iloc[:, :-1], train_new.iloc[:, -1]
        X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

        with open(f'models/{model_id}.pkl', 'rb') as f:
            estimator = pickle.load(f)
        estimator.fit(X_train, y_train)
        with open(f'models/{model_id}.pkl', 'wb') as f:
            pickle.dump(estimator, f)

        self.ml_models_all[model_id]['retrained'] = True
        self.ml_models_all[model_id]['metrics'] = calculate_metrics(self.ml_models_all[model_id]['model'],
                                                                    estimator, X_train, X_test, y_train, y_test)

    def delete(self, model_id):
        """
        Delete pkl file of the given model.
        """
        if model_id not in self.ml_models_all:
            api.abort(404, f'Model {model_id} does not exist')
        if self.ml_models_all[model_id]['deleted']:
            api.abort(404, f'Model {model_id} was already deleted')
        os.remove(f'models/{model_id}.pkl')
        self.ml_models_all[model_id]['deleted'] = True

    def plot(self, model_id):
        """
        Return byte plots for the given model for both train and test set:
           - residual plot for a regression model;
           - confusion matrix for classification model.
        """
        if model_id not in self.ml_models_all:
            api.abort(404, f'Model {model_id} does not exist')
        if self.ml_models_all[model_id]['deleted']:
            api.abort(404, f'Model {model_id} was deleted')

        with open(f'models/{model_id}.pkl', 'rb') as f:
            estimator = pickle.load(f)
        with open(f'data/{model_id}_train.pkl', 'rb') as f:
            train = pickle.load(f)
        with open(f'data/{model_id}_test.pkl', 'rb') as f:
            test = pickle.load(f)

        X_train, X_test = train.iloc[:, :-1].values, test.iloc[:, :-1].values
        y_train, y_test = train.iloc[:, -1].values, test.iloc[:, -1].values
        y_train_pred, y_test_pred = estimator.predict(train.iloc[:, :-1]), estimator.predict(test.iloc[:, :-1])

        if model2task[self.ml_models_all[model_id]['model']] == 'regression':
            plt.figure(figsize=(20, 5))
            plt.subplot(121)
            plt.scatter(np.arange(len(y_train)), y_train - y_train_pred)
            plt.axhline(xmin=0, xmax=len(y_train), color='red', linestyle='--')
            plt.xlabel('Observation')
            plt.ylabel('Residual')
            plt.title(f'Train set\n MAE={self.ml_models_all[model_id]["metrics"]["train"]["mae"].round(4)}, '
                      f'RMSE={self.ml_models_all[model_id]["metrics"]["train"]["rmse"].round(4)}')
            plt.subplot(122)
            plt.scatter(np.arange(len(y_test)), y_test - y_test_pred)
            plt.axhline(xmin=0, xmax=len(y_test), color='red', linestyle='--')
            plt.xlabel('Observation')
            plt.ylabel('Residual')
            plt.title(f'Test set\n MAE={self.ml_models_all[model_id]["metrics"]["test"]["mae"].round(4)}, '
                      f'RMSE={self.ml_models_all[model_id]["metrics"]["test"]["rmse"].round(4)}')
            plt.suptitle('Residual Plot')
        else:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            plot_confusion_matrix(estimator, X_train, y_train, cmap='Blues', ax=ax1)
            ax1.set_title('Train set')
            plot_confusion_matrix(estimator, X_test, y_test, cmap='Blues', ax=ax2)
            ax2.set_title('Test set')
            plt.suptitle('Confusion Matrix')

        bytes_image = BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image

    def plot_feature_importance(self, model_id):
        """
        Return a feature importance plot for the given tree-based model.
        """
        if model_id not in self.ml_models_all:
            api.abort(404, f'Model {model_id} does not exist')
        if self.ml_models_all[model_id]['deleted']:
            api.abort(404, f'Model {model_id} was deleted')

        model_name = self.ml_models_all[model_id]['model']
        if 'Tree' in model_name or 'Forest' in model_name:
            with open(f'models/{model_id}.pkl', 'rb') as f:
                estimator = pickle.load(f)
            with open(f'data/{model_id}_train.pkl', 'rb') as f:
                train = pickle.load(f)

            feature_importance = np.array(estimator.feature_importances_)
            feature_names = np.array(train.columns[:-1])
            fi_df = pd.DataFrame({'feature_names': feature_names, 'feature_importance': feature_importance})
            fi_df.sort_values(by='feature_importance', ascending=False, inplace=True)

            plt.figure(figsize=(20, 7))
            sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')

            bytes_image = BytesIO()
            plt.savefig(bytes_image, format='png')
            bytes_image.seek(0)
            return bytes_image
        else:
            api.abort(400, f'Can only plot feature importance for a tree-based model')


models_dao = MLModelsDAO()


@api.route('/ml_api')
class MLModels(Resource):

    def get(self):
        """
        Return a list of models available for training.
        """
        return models_dao.ml_models

    def post(self):
        """
        Train the given model with given parameters on the given data (json)
        where the last element is target.
        If parameters is not set, use default values.
        Once the model is trained, append model name, parameters and performance metrics
        to models_dao.ml_models_all dictionary.
        """
        json_ = request.json
        data = json_['data']
        model = json_['model']
        params = json_.get('params', 'default')
        grid_search = json_.get('grid_search', False)
        param_grid = json_.get('param_grid', 'default')
        models_dao.create(data, model, params, grid_search, param_grid)


@api.route('/ml_api/<int:model_id>')
class MLModelsID(Resource):

    def put(self, model_id):
        """
        Retrain the given model on a new training set.
        """
        data = request.json
        models_dao.update(model_id, data)

    def delete(self, model_id):
        """
        Delete pkl file of the given model.
        """
        models_dao.delete(model_id)
        return '', 204

    def get(self, model_id):
        """
        Return predictions for the given model.
        """
        return models_dao.get(model_id)


@api.route('/ml_api/all_models')
class MLModelsAll(Resource):

    def get(self):
        """
        Return a dictionary of all fitted models, their parameters and performances.
        """
        return models_dao.ml_models_all


@api.route('/ml_api/plot/<int:model_id>')
class MLModelsPlot(Resource):

    def get(self, model_id):
        """
        Return byte plots for the given model for both train and test set:
           - residual plot for a regression model;
           - confusion matrix for classification model.
        """
        return send_file(models_dao.plot(model_id),
                         download_name='plot.png',
                         mimetype='image/png')


@api.route('/ml_api/plot_feature_importance/<int:model_id>')
class MLModelsPlotFI(Resource):

    def get(self, model_id):
        """
        Return a feature importance plot for the given tree-based model.
        """
        return send_file(models_dao.plot_feature_importance(model_id),
                         download_name='plot.png',
                         mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, host=HOST, port=PORT)
