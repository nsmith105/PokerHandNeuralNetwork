"""

Nick Smith
CS 445 - Machine Learning
Final Project:
POKER HAND CLASSIFICATION


############ DATASET INFO ##############

10 attributes in Suit, Rank, Suit, ... form
11th attribute to designate class.

## SUITS ##
1 - Hearts
2 - Spades
3 - Diamonds
4 - Clubs

## RANK ##
ACE through 10
JACK  - 11
QUEEN - 12
KING  - 13

## HANDS ##
0: Nothing in hand; not a recognized poker hand
1: One pair; one pair of equal ranks within five cards
2: Two pairs; two pairs of equal ranks within five cards
3: Three of a kind; three equal ranks within five cards
4: Straight; five cards, sequentially ranked with no gaps
5: Flush; five cards with the same suit
6: Full house; pair + different rank three of a kind
7: Four of a kind; four equal ranks within five cards
8: Straight flush; straight + flush
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush

"""
# -------------------- Imports -------------------- #
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np, pandas as pd, os
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import math

# -------------------- Globals and Configs -------------------- #
feature_names = list()
for index in range(1, 6):
    feature_names.extend(["Suit"+str(index), "Rank"+str(index)])

feature_names.append('class')


training_input_file = os.path.abspath('poker-hand-training-true.csv')
testing_input_file = os.path.abspath('poker-hand-testing.csv')

np.random.seed(666)     # seed for reproducible results


# To store configs
class myConfigs:
    features = 0
    classes = 0


config = myConfigs()

# -------------------- Data -------------------- #

train_data = pd.read_csv(training_input_file, names=feature_names)
test_data = pd.read_csv(testing_input_file, names=feature_names)

# Get features of data
config.features = len(train_data.columns) - 1
config.classes = len(set(train_data['class']))

# Shuffle training data
train_data = train_data.sample(frac=1).reset_index(drop=True)

# Seperate data and classes
train_y = np.array(train_data['class'])
train_x = np.array(train_data.drop('class', 1))

test_y = np.array(test_data['class'])
test_x = np.array(test_data.drop('class', 1))

train_y_onehot = list()
for y in range(len(train_y)):
    temp = [0] * config.classes
    temp[train_y[y]] = 1
    train_y_onehot.append(temp)

test_y_onehot = list()
for y in range(len(test_y)):
    temp = [0] * config.classes
    temp[test_y[y]] = 1
    test_y_onehot.append(temp)

train_y_onehot = np.array(train_y_onehot)
test_y_onehot = np.array(test_y_onehot)

tab = PrettyTable(['Config', 'Value'])
configs = vars(config)

for key in configs:
    tab.add_row([key, configs[key]])
print(tab)

print("Instances in training data :", len(train_data))
print("Instances in testing data :", len(test_data))

# -------------------- Model -------------------- #
model = Sequential()

# Input layer
model.add(Dense(10, use_bias=True, bias_initializer='ones', input_shape = (train_x.shape[1],), activation='sigmoid'))

# Hidden layers

# Output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y_onehot, epochs = 10, batch_size = 500, verbose=0)

model.summary()

preds = [list(x) for x in model.predict(train_x)]
for i in range(len(preds)):
    preds[i] = preds[i].index(max(preds[i]))

train_matches = 0
for i in range(len(train_y)):
    if train_y[i] == preds[i]:
        train_matches += 1

print("Matches :", train_matches, '/', len(train_y), '=', train_matches / len(train_y) * 100)
print("Average Error :", sum([math.fabs(x - y) for x, y in zip(train_y, preds)]) / len(train_y))
print("RMSE :", math.sqrt(sum([(x - y) ** 2 for x, y in zip(train_y, preds)]) / len(train_y)))

plt.plot(train_y, 'bo', label='Original')
plt.plot(preds, 'ro', label='Predicted')
plt.legend()
plt.show()

preds = [list(x) for x in model.predict(test_x)]
for i in range(len(preds)):
    preds[i] = preds[i].index(max(preds[i]))

test_matches = 0
for i in range(len(test_y)):
    if test_y[i] == preds[i]:
        test_matches += 1

print("Matches :", test_matches, '/', len(test_y), '=', test_matches / len(test_y) * 100)
print("Average Error :", sum([math.fabs(x - y) for x, y in zip(test_y, preds)]) / len(test_y))
print("RMSE :", math.sqrt(sum([(x - y) ** 2 for x, y in zip(test_y, preds)]) / len(test_y)))

plt.plot(test_y, 'bo', label='Original')
plt.plot(preds, 'ro', label='Predicted')
plt.legend()
plt.show()


scores = model.evaluate(train_x, train_y_onehot)
print("Train =", model.metrics_names[1], scores[1] * 100)

scores = model.evaluate(test_x, test_y_onehot)
print("Test =", model.metrics_names[1], scores[1] * 100)

train_accuracies = list()
test_accuracies = list()

for iterations in range(50, 500, 50):
    model.fit(train_x, train_y_onehot, epochs = iterations, batch_size = 500, verbose=0)
    scores = model.evaluate(train_x, train_y_onehot, verbose=0)
    train_accuracies.append(scores[1]* 100)
    scores = model.evaluate(test_x, test_y_onehot, verbose=0)
    test_accuracies.append(scores[1]* 100)

plt.title('Accuracies for training data')
plt.plot(range(50, 500, 50), train_accuracies)
plt.show()
plt.title('Accuracies for testing data')
plt.plot(range(50, 500, 50), test_accuracies)
plt.show()

confMat = [[0] * config.classes for x in range(config.classes)]

for i in range(len(preds)):
    predLabel = preds[i]
    actLabel = test_y[i]
    confMat[actLabel][predLabel] += 1

header = [""]
for l in range(config.classes):
    header.append("Pred " + str(l))

tab = PrettyTable(header)
l = 0
for c in confMat:
    tab.add_row(["Real " + str(l)] + c)
    l += 1

print(tab)

