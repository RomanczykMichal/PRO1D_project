import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression


def main():
    # zbiory danych
    df_cleaned, df_mean, df_med = prep_data()

    # utworzenie zmiennych
    X, y = prep_dataset(df_cleaned)
    column_names = future_names(X, y)

    # przefiltrowanie kolumn na te najważniejsze, z metody future_names
    df_cleaned_filtered = df_cleaned[column_names]


    # podział na zbiory testowe i treningowe
    X_train, X_test, y_train, y_test = train_test_split(df_cleaned_filtered, y, test_size=0.25, random_state=5)



def prep_data():
    # Dane oczyszczone przez nas
    df = pd.read_csv('data/data_cleaned.csv', sep=';')
    df_cleaned = df.iloc[:, 5:]  # dane z komulnami decyzyjnymi
    df_names = df.iloc[:, :5]  # dane z informacjami o wierszu

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

    return df_cleaned, df_mean, df_med


def prep_dataset(df, target='target'):
    from xverse.feature_subset import SplitXY

    clf = SplitXY([target])  # Split the dataset into X and y
    X, y = clf.fit_transform(df)  # returns features (X) dataset and target(Y) as a numpy array

    return X, y


def future_names(X, y):
    future_selector = SelectKBest(score_func=f_regression, k=80)
    X_selected = future_selector.fit_transform(X, y)
    X_selected_df = pd.DataFrame(X_selected, columns=[X.columns[i] for i in range(len(X.columns)) if
                                                      future_selector.get_support()[i]])

    future_selector2 = SelectKBest(score_func=mutual_info_regression, k=80)
    X_selected2 = future_selector2.fit_transform(X, y)
    X_selected2_df = pd.DataFrame(X_selected2, columns=[X.columns[i] for i in range(len(X.columns)) if
                                                        future_selector2.get_support()[i]])

    selected_features = np.append(X_selected2_df.columns.to_numpy(), X_selected_df.columns.to_numpy())
    return np.unique(selected_features)


if __name__ == '__main__':
    main()
