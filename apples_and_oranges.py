from sklearn import tree
import numpy as np

# Training Data 
# Examples we want to use to train the classifier

# this is supervised learning, we provide examples for the input 'features', and the 'output'
# features =  [[140, 'smooth'], [130, 'smooth'], [150, 'smooth'], [150, 'bumpy'], [170, 'bumpy']]
# Converting to integers: 0 is 'bumpy', 1 is 'smooth'

features =  [[140, 1], [130, 1], [150, 0], [170, 0]]

# 0 is 'apple', 1 is 'orange'
# labels = ["apple", "apple", "orange", "orange"]

labels = [0, 0, 1, 1]

# descision tree, aka 'box of rules'

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

data = [150, 0]
# reshape the data since it is a single sample
data = np.array(data).reshape(1, -1)

print classifier.predict(data)
