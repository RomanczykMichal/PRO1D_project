import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split


def main():
    # zbiory danych
    df_original, df_cleaned, df_mean, df_med = prep_data()

    # utworzenie zmiennych
    X, y = prep_dataset(df_cleaned)
    column_names = future_names(X, y)
    # przefiltrowanie kolumn na te najważniejsze, z metody future_names
    df_cleaned_filtered = df_cleaned[column_names]

    # podział na zbiory testowe i treningowe
    # Tutaj można podmienić wartość inputs na dowolny dataset
    inputs = df_mean
    X_train, X_test, y_train, y_test = train_test_split(inputs.values, y.values, test_size=0.25, random_state=5)

    # siec neuronowa
    alg.neural_network(X_train, X_test, y_train, y_test)

    # Drzewo regresyjne
    alg.decision_tree_regressor(X_train, X_test, y_train, y_test)

    # regresja liniowa
    alg.linear_regression(X_train, X_test, y_train, y_test)

    # regresja nieliniowa


def prep_data():
    """
    Metoda odpowiedzialna za pobranie i wstępne przygotowanie danych.

    :return
        df_original -> DataFrame z oryginalnymi wartościami z pliku źródłowego
        df_cleaned  -> DataFrame bez usuniętymi kolumnami, w których znajdowały
                       się puste wartości
        df_mean     -> DataFrame z podmienionymi wartościami na średnią dla danej
                       wartości, gdzie tych danych brakowało.
        df_med      -> DataFrame z podmienionymi wartościami na medianę dla danej
                       wartości, gdzie tych danych brakowało.
    :
    """

    # Dane surowe
    df_original = pd.read_csv('data/data.csv', sep=';')
    df_original = df_original.iloc[:, 5:]

    # Dane oczyszczone przez nas
    df_cleaned = pd.read_csv('data/data_cleaned.csv', sep=';')
    df_cleaned = df_cleaned.iloc[:, 5:]

    # Dane surowe z warotściami wypełonymi średnią
    df_mean = pd.read_csv('data/data.csv', sep=';')
    df_mean = df_mean.iloc[:, 5:]
    df_mean = df_mean.replace('?', np.NaN)
    df_mean = df_mean.apply(pd.to_numeric)
    df_mean = df_mean.fillna(df_mean.mean())

    # Dane surowe z warotściami wypełonymi medianą
    df_med = pd.read_csv('data/data.csv', sep=';')
    df_med = df_med.iloc[:, 5:]
    df_med = df_med.replace('?', np.NaN)
    df_med = df_med.apply(pd.to_numeric)
    df_med = df_med.fillna(df_med.median())

    return df_original, df_cleaned, df_mean, df_med


def prep_dataset(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    return X, y


def future_selector(X, y):
    """
    Metoda odpowiedzialna za głosowanie za kolumnami, które posiadają
    odpowiednią korelację pomiędzy sobą.

    :param X
        Kolumny z wartosciami input
    :

    :param y
        Kolumna z wartoscią decyzyjną
    :

    :return
        Nazwy kolumn wybrane w wyniku głosowań.
    :

    TODO połączyć f_selector i f_selector2 w jedną funkcję, zmienić po prostu score_func.
         Sprawdzic czy jest mozliwosc po zrobionym juz fit_transform

    TODO ogarnac w ogole co robi fit_transform
    """

    f_selector = SelectKBest(score_func=f_regression, k=80)
    X_selected = f_selector.fit_transform(X.values, y.values.ravel())
    X_selected_df = pd.DataFrame(X_selected, columns=[X.columns[i] for i in range(len(X.columns)) if
                                                      f_selector.get_support()[i]])

    f_selector2 = SelectKBest(score_func=mutual_info_regression, k=80)
    X_selected2 = f_selector2.fit_transform(X.values, y.values.ravel())
    X_selected2_df = pd.DataFrame(X_selected2, columns=[X.columns[i] for i in range(len(X.columns)) if
                                                        f_selector2.get_support()[i]])

    selected_features = np.append(X_selected2_df.columns.to_numpy(), X_selected_df.columns.to_numpy())
    return np.unique(selected_features)


if __name__ == '__main__':
    main()
