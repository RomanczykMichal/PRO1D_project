import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import random
from sklearn.linear_model import LinearRegression


def neural_network(X_train, X_test, y_train, y_test, X, y):
    """
    Sources:
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
        https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    """
    model_r = MLPRegressor(hidden_layer_sizes=(90, 40, 10), activation='logistic', alpha=0.001,
                           learning_rate_init=0.001,
                           learning_rate='adaptive', solver='adam', max_iter=10000, early_stopping=True, verbose=False)
    model_r.fit(X_train, y_train)
    print('Neural network prediction: ', model_r.score(X_test, y_test))

    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='r2')

    print('Neural network cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Neural network cross-validation mean score r2 is: ', scores2.mean(), scores2.std())


def decision_tree_regressor(X_train, X_test, y_train, y_test, X, y):
    """
    Sources:
        https://www.statsoft.pl/textbook/stathome_stat.html?https%3A%2F%2Fwww.statsoft.pl%2Ftextbook%2Fstcart.html
    """
    model_r = DecisionTreeRegressor()
    model_r.fit(X_train, y_train)
    print('Decision tree prediction: ', model_r.score(X_test, y_test))

    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='r2')
    print('Decision tree cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Decision tree cross-validation mean score r2 is: ', scores2.mean(), scores2.std())


def linear_regression(X_train, X_test, y_train, y_test, X, y):
    """
    Sources:
        https://satishgunjal.com/multivariate_lr_scikit/
    """

    model_r = LinearRegression()

    model_r.fit(X_train, y_train)
    print('Linear regression prediction: ', model_r.score(X_test, y_test))

    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='r2')
    print('Linear regression cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Linear regression cross-validation mean score r2 is: ', scores2.mean(), scores2.std())


def non_linear_regression(X_train, X_test, y_train, y_test, X, y):
    """
    Sources:
        https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
        https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/
    """
    model_r = SVR()

    model_r.fit(X_train, y_train)
    print('Non-linear regression prediction: ', model_r.score(X_test, y_test))

    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='r2')
    print('Non-Linear regression cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Non-Linear regression cross-validation mean score r2 is: ', scores2.mean(), scores2.std())


def random_forest_regressor(X_train, X_test, y_train, y_test, X, y):
    """
        Sources:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        """
    model_r = RandomForestRegressor(n_estimators=2000, min_samples_split=2, min_samples_leaf=2, max_features='sqrt',
                                    max_depth=20, bootstrap=True)

    model_r.fit(X_train, y_train)
    print('Random Forest Regression prediction: ', model_r.score(X_test, y_test))

    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='neg_mean_squared_error')
    scores2 = cross_val_score(model_r, X.values, y.values.ravel(), cv=10, scoring='r2')
    print('Random Forest Regression cross-validation mean score mse is: ', scores.mean(), scores.std())
    print('Random Forest Regression cross-validation mean score r2 is: ', scores2.mean(), scores2.std())


def neural_network_parameter_search(X_train, y_train):
    """
    Sources
        https://stackoverflow.com/questions/54735717/how-to-define-a-mlpr-with-two-hidden-layers-for-randomsearchcv
    """
    model_r = MLPRegressor(learning_rate='adaptive', max_iter=5000, random_state=42, early_stopping=True)
    params = {
        'activation': ['logistic', 'relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'hidden_layer_sizes': [(random.randrange(20, 100), random.randrange(10, 50), random.randrange(5, 20)) for i in
                               range(40)],
        'solver': ['sgd', 'adam']
    }

    model_r_random = RandomizedSearchCV(estimator=model_r, param_distributions=params, n_iter=120, cv=4, verbose=5,
                                        n_jobs=-1, random_state=42)
    model_r_random.fit(X_train, y_train)
    print(model_r_random.best_params_)
    print(model_r_random.best_estimator_)


def random_forest_parameter_search(X_train, y_train):
    """
    Sources:
        https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
