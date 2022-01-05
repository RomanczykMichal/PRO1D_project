import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

def main():
    # Dane oczyszczone przez nas
    df = pd.read_csv('data/data_cleaned.csv', sep=';')
    df_value_columns = df.iloc[:, 5:]  # dane z komulnami decyzyjnymi
    df_names = df.iloc[:, :5]  # dane z informacjami o wierszu

    # Dane surowe z warotściami wypełonymi średnią
    df_mean = pd.read_csv('data/data.csv', sep=';')
    df_mean = df_mean.iloc[:,5:]
    df_mean = df_mean.replace('?', np.NaN)
    df_mean = df_mean.apply(pd.to_numeric)
    df_mean = df_mean.fillna(df_mean.mean())

    # Dane surowe z warotściami wypełonymi medianą
    df_med = pd.read_csv('data/data.csv', sep=';')
    df_med = df_med.iloc[:, 5:]
    df_med = df_med.replace('?', np.NaN)
    df_med = df_med.apply(pd.to_numeric)
    df_med = df_med.fillna(df_med.median())

    # utworzenie zmiennych
    X, y = prep_dataset(df_value_columns, target='ViolentCrimesPerPop')


def prep_dataset(df, target='target'):
    from xverse.feature_subset import SplitXY

    clf = SplitXY([target])  # Split the dataset into X and y
    X, y = clf.fit_transform(df)  # returns features (X) dataset and target(Y) as a numpy array

    return X, y

def future_names(X, y):
    # define feature selection
    fs = SelectKBest(score_func=f_regression, k=35)
    # apply feature selection
    X_selected = fs.fit_transform(X, y)
    X_selected_df = pd.DataFrame(X_selected, columns=[X.columns[i] for i in range(len(X.columns)) if fs.get_support()[i]])

    fs2 = SelectKBest(score_func=mutual_info_regression, k=35)
    X_selected2 = fs2.fit_transform(X, y)
    X_selected2_df = pd.DataFrame(X_selected2, columns=[X.columns[i] for i in range(len(X.columns)) if fs2.get_support()[i]])
    print(X_selected2_df.columns)

if __name__ == '__main__':
    main()
