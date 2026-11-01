import pandas as pd
from DecisionTreeClassifier import MyDecisionTreeClassifier

df = pd.read_csv('./titanic_train.csv')

X = df[['Fare', 'Sex']]
X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
y = df["Survived"]

clf = MyDecisionTreeClassifier(max_depth=3, criterion="entropy")
clf.fit(X, y)

print(clf.predict(X))