from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


def neural_network(X_train, X_test, y_train, y_test):
    """
    Sources:
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
        https://scikit-learn.org/stable/modules/neural_networks_supervised.html

        TODO Chyba naprawilem ta siec. podejrzewam, ze problem leżał tutaj w wielkości sieci, alphie i w tym
             jak dzielilismy dane. Wyrzucało błąd bo do metody train_test_split wrzucalismy DataFrame, a to
             chcialo ndarray.

        TODO Dla max_iter=1000 wartość się zwiększa do ~0.99
             Dla max_iter=500  wartosc wynosi ~0.83
             Trzeba się zastanowić jak bardzo chcemy szkolić naszą sieć, żeby nie była przetrenowana
    """
    model_r = MLPRegressor(random_state=5, hidden_layer_sizes=(22,), activation='logistic', alpha=0.0001,
                           learning_rate_init=0.0001, solver='adam', max_iter=500, early_stopping=True, verbose=False)
    model_r.fit(X_train, y_train)
    print('Neural network score:', model_r.score(X_test, y_test))


def decision_tree_regressor(X_train, X_test, y_train, y_test):
    """
    Sources:
        https://www.statsoft.pl/textbook/stathome_stat.html?https%3A%2F%2Fwww.statsoft.pl%2Ftextbook%2Fstcart.html

        TODO Dziwnie duze wyniki wychodza :)
    """
    model_r = DecisionTreeRegressor(random_state=0)
    model_r.fit(X_train, y_train)
    print('Decision tree regressor score:', model_r.score(X_test, y_test))


def linear_regression(X_train, X_test, y_train, y_test):
    """
    Sources:
        https://satishgunjal.com/multivariate_lr_scikit/

        TODO dlaczego alpha 35?
    """
    model_r = linear_model.Ridge(alpha=35)
    model_r.fit(X_train, y_train)
    print('Linear regression score:', model_r.score(X_test, y_test))


def non_linear_regression(X_train, X_test, y_train, y_test):
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
    print('Non-Linear regression score:', model_r.score(X_test, y_test))
