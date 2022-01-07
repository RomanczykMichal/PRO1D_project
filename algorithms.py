from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def neural_network(X_train, X_test, y_train, y_test, X, y):
    """
    Sources:
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
        https://scikit-learn.org/stable/modules/neural_networks_supervised.html

        scores - tablica wyników z cross-validation.
        cross_val_model - cv=5 to tyle razy odpalane jest uczenie.
    """
    model_r = MLPRegressor(hidden_layer_sizes=(22,), activation='logistic', alpha=0.0001, learning_rate_init=0.0001,
                           solver='adam', max_iter=2000, early_stopping=True, verbose=False)
    model_r.fit(X_train, y_train)
    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=5, scoring='r2')

    print('\nNeural network score:', model_r.score(X_test, y_test))
    metrics.plot_roc_curve(model_r, X_test, y_test)
    print('Neural network cross-validation mean score is: ', scores.mean(), scores.std())


def decision_tree_regressor(X_train, X_test, y_train, y_test, X, y):
    """
    Sources:
        https://www.statsoft.pl/textbook/stathome_stat.html?https%3A%2F%2Fwww.statsoft.pl%2Ftextbook%2Fstcart.html

        TODO Dziwnie duze wyniki wychodza :)
    """
    model_r = DecisionTreeRegressor()
    reg = model_r.fit(X_train, y_train)
    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=5, scoring='r2')

    print('\nDecision tree regressor score:', model_r.score(X_test, y_test))
    metrics.plot_roc_curve(model_r, X_test, y_test)
    print('Decision tree cross-validation mean score is: ', scores.mean(), scores.std())

    # plt.figure()
    # plot_tree(reg, filled=True)
    # plt.savefig('fig.png', dpi=600)
    # plt.show()


def linear_regression(X_train, X_test, y_train, y_test, X, y):
    """
    Sources:
        https://satishgunjal.com/multivariate_lr_scikit/

        TODO dlaczego alpha 35?
    """
    model_r = linear_model.Ridge(alpha=1)
    model_r.fit(X_train, y_train)
    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=5, scoring='r2')

    print('\nLinear regression score:', model_r.score(X_test, y_test))
    metrics.plot_roc_curve(model_r, X_test, y_test)
    print('Linear regression cross-validation mean score is: ', scores.mean(), scores.std())


def non_linear_regression(X_train, X_test, y_train, y_test, X, y):
    """
    Sources:
        https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
        https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/

        TODO dodalem to, ale nie mam pojecia czy to jest w ogole to XD Z tego co przeczytalem
             w tych artykulach to niby powinno sie zgadzac
    """
    model_r = SVR()
    model_r.fit(X_train, y_train)
    scores = cross_val_score(model_r, X.values, y.values.ravel(), cv=5, scoring='r2')

    print('\nNon-Linear regression score:', model_r.score(X_test, y_test))
    metrics.plot_roc_curve(model_r, X_test, y_test)
    print('Non-Linear regression cross-validation mean score is: ', scores.mean(), scores.std())
