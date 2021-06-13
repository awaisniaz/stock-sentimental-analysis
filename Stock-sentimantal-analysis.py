import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('Data.csv',encoding="ISO-8859-1")
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# Feature Engineering
# Removing Special Character

data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex = True,inplace = True)
list1 = [str(i) for i in range(25)]
data.columns = list1
for index in list1:
    data[index] = data[index].str.lower()

headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

# Implement BAG OF WORDS
counterVectorizer = CountVectorizer(ngram_range=(2,2))
train_Data = counterVectorizer.fit_transform((headlines))

# Implementing Random forest Classifier
randomClassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomClassifier.fit(train_Data,train['Label'])

# predict for the Test Dataset
test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))

test_dateset = counterVectorizer.fit_transform((test_transform))
prediction = randomClassifier.predict(test_dateset)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix = confusion_matrix(test['Label'],prediction)
score = accuracy_score(test['Label'],prediction)
report = classification_report(test['Label'],prediction)

