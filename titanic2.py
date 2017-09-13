import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

train_df = pd.read_csv("I://m-l/train.csv")
test_df = pd.read_csv("I://m-l/test.csv")

train_df.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)
test_df.drop(["Name","Ticket"],axis=1,inplace=True)

train_df["Embarked"] = train_df["Embarked"].fillna("S")

embark_dummies = pd.get_dummies(train_df["Embarked"])
embark_dummies.drop(["S"],axis=1,inplace=True)

embark_dummies_t = pd.get_dummies(test_df["Embarked"])
embark_dummies_t.drop(["S"],axis=1,inplace=True)

train_df = train_df.join(embark_dummies)
test_df = test_df.join(embark_dummies_t)


train_df.drop(['Embarked'],axis=1,inplace=True)
test_df.drop(["Embarked"],axis=1,inplace=True)

test_df["Fare"].fillna(test_df["Fare"].median(),inplace=True)

train_df["Fare"] = train_df["Fare"].astype(int)
test_df["Fare"] = test_df["Fare"].astype(int)

data = [train_df,test_df]

for df in data:
    age_mean = df["Age"].mean()
    age_std = df["Age"].std()
    age_nullNum = df["Age"].isnull().sum()

    rand = np.random.randint(age_mean-age_std,age_mean+age_std,size=age_nullNum)
    df["Age"][np.isnan(df["Age"])] = rand
    df["Age"] = df["Age"].astype(int)


train_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)

train_df["Family"] = train_df["SibSp"]+train_df["Parch"]
train_df["Family"][train_df["Family"]>0] = 1
train_df["Family"][train_df["Family"]==0] = 0

train_df.drop(["SibSp","Parch"],axis=1,inplace=True)

test_df["Family"] = test_df["SibSp"]+test_df["Parch"]
test_df["Family"][test_df["Family"]>0] = 1
test_df["Family"][test_df["Family"]==0] = 0

test_df.drop(["SibSp","Parch"],axis=1,inplace=True)

def ischild(passenger):
    age,sex = passenger
    return "child" if age<16 else sex

train_df['Person'] = train_df[['Age','Sex']].apply(ischild,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(ischild,axis=1)

train_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

person_dummies = pd.get_dummies(train_df["Person"])
person_dummies.drop(["male"],axis=1,inplace=True)
train_df = train_df.join(person_dummies)
train_df.drop(["Person"],axis=1,inplace=True)

person_dummies_t = pd.get_dummies(test_df["Person"])
person_dummies_t.drop(["male"],axis=1,inplace=True)
test_df = test_df.join(person_dummies_t)
test_df.drop(["Person"],axis=1,inplace=True)

pclass_dummies = pd.get_dummies(train_df["Pclass"])
pclass_dummies.columns = ["class1","class2","class3"]
pclass_dummies.drop(["class3"],axis=1,inplace=True)
train_df = train_df.join(pclass_dummies)

pclass_dummies_t = pd.get_dummies(test_df["Pclass"])
pclass_dummies_t.columns = ["class1","class2","class3"]
pclass_dummies_t.drop(["class3"],axis=1,inplace=True)
test_df = test_df.join(pclass_dummies_t)

test_df.drop(["Pclass"],axis=1,inplace=True)
train_df.drop(["Pclass"],axis=1,inplace=True)

train_df.drop(["Cabin"],axis=1,inplace=True)
test_df.drop(["Cabin"],axis=1,inplace=True)

X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()

Y_pred = {}

logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred["Y_logreg"]=logreg.predict(X_test)
print("LogisticRegression`s score on training set:"+str(logreg.score(X_train,Y_train)))

svc = SVC()
svc.fit(X_train,Y_train)
Y_pred["Y_svc"] = svc.predict(X_test)
print("Support Vector Machine`s score on training set:"+str(svc.score(X_train,Y_train)))

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred["Y_randfst"] = random_forest.predict(X_test)
print("Randomforest`s score on training set:"+str(random_forest.score(X_train,Y_train)))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred["Y_knn"] = knn.predict(X_test)
print("Knn`s score on training set:"+str(knn.score(X_train,Y_train)))

nb = GaussianNB()
nb.fit(X_train,Y_train)
Y_pred["Y_nb"] = nb.predict(X_test)
print("Naivy Bay`s score on training set:"+str(nb.score(X_train,Y_train)))

def save_to_csv(Y_pred):
    for m,pred in Y_pred.items():
        r = pd.DataFrame({
            "PassengerId":test_df["PassengerId"],
            "Survived":pred
        })
        r.to_csv(m+".csv",index=False)

