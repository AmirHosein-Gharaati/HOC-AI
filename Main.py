import csv
from sklearn import tree

data = []
male_or_female = []

with open("data.csv") as csvfile:
    f = csv.reader(csvfile)
    for line in f:
        data.append(line[1:4])
        male_or_female.append(line[4])

clf = tree.DecisionTreeClassifier()
learner = clf.fit(data[1:], male_or_female[1:])

sample1 = [185,75,42]
sample2 = [164, 64, 40]

new_data = [sample2]
answer = learner.predict(new_data)
print(answer)