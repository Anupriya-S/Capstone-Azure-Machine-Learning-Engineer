import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from azureml.core.run import Run
import argparse
import os

# df = pd.read_csv('nasa.csv')

# df['Hazardous'] = df['Hazardous'].map({True: 1, False: 0})
#
# df=df.drop(columns=['Neo Reference ID', 'Name', 'Close Approach Date', 'Orbit Determination Date', 'Orbiting Body', 'Equinox'])
# df=df.drop(columns=['Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)'])
# df=df.drop(columns=['Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)'])
# df=df.drop(columns=['Est Dia in KM(max)'])

# x = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 42)

# clf = DecisionTreeClassifier()
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)

# print('Accuracy =',accuracy_score(y_test, y_pred))

def clean_data(df):
    df['Hazardous'] = df['Hazardous'].map({True: 1, False: 0})

    df=df.drop(columns=['Neo Reference ID', 'Name', 'Close Approach Date', 'Orbit Determination Date', 'Orbiting Body', 'Equinox'])
    df=df.drop(columns=['Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)', 'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)'])
    df=df.drop(columns=['Relative Velocity km per hr', 'Miles per hour', 'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)', 'Miss Dist.(miles)'])
    df=df.drop(columns=['Est Dia in KM(max)'])

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node")
    parser.add_argument('--min_impurity_decrease', type=float, default=0.0, help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value")
    parser.add_argument('--random_state', type=int, default=42, help="Controls the randomness of the estimator")
    args = parser.parse_args()
    run.log("Maximum Depth:", np.int(args.max_depth))
    run.log("Minimum Samples Required:", np.int(args.min_samples_split))
    run.log("Minimum Impurity Decrement:", np.float(args.min_impurity_decrease))
    run.log("Random State:", np.int(args.random_state))
    model = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_split=args.min_samples_split, min_impurity_decrease=args.min_impurity_decrease, random_state=args.random_state).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy:", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperdrive-model.joblib')

if __name__ == '__main__':
    df = pd.read_csv('https://raw.githubusercontent.com/Anupriya-S/Capstone-Azure-Machine-Learning-Engineer/main/nasa.csv')
    df = clean_data(df)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2 , random_state = 42)
    run = Run.get_context()
    main()
