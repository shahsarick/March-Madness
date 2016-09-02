# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 01:56:23 2016

@author: Sarick
"""

import pandas as pd
import math
from sklearn import cross_validation, linear_model, datasets, metrics
from sklearn.naive_bayes import GaussianNB
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron



team_1_features = []
base_elo = 1600
team_elos = {}  # Reset each year.
team_stats = {}
X = []
y = []
submission_data = []
submission_data2 = []
submission_data3 = []
submission_data4 = []
folder = 'data-v2'
prediction_year = 2016


def calc_elo(win_team, lose_team, season):
    winner_rank = get_elo(season, win_team)
    loser_rank = get_elo(season, lose_team)

    """
    This is originally from from:
    http://zurb.com/forrst/posts/An_Elo_Rating_function_in_Python_written_for_foo-hQl
    """
    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff
    return new_winner_rank, new_loser_rank


def initialize_data():
    for i in range(1985, 2017):
        team_elos[i] = {}
        team_stats[i] = {}


def get_elo(season, team):
    try:
        return team_elos[season][team]
    except:
        try:
            # Get the previous season's ending value.
            team_elos[season][team] = team_elos[season-1][team]
            return team_elos[season][team]
        except:
            # Get the starter elo.
            team_elos[season][team] = base_elo
            return team_elos[season][team]


def predict_winner(team_1, team_2, model, season, stat_fields):
    features = []

    # Team 1
    features.append(get_elo(season, team_1))
    for stat in stat_fields:
        features.append(get_stat(season, team_1, stat))

    # Team 2
    features.append(get_elo(season, team_2))
    for stat in stat_fields:
        features.append(get_stat(season, team_2, stat))

    return model.predict_proba([features])  
    
def predict_winnerNN(team_1, team_2, model, season, stat_fields):
    features = []

    # Team 1
    features.append(get_elo(season, team_1))
    for stat in stat_fields:
        features.append(get_stat(season, team_1, stat))

    # Team 2
    features.append(get_elo(season, team_2))
    for stat in stat_fields:
        features.append(get_stat(season, team_2, stat))      
    print(len([features]))
    return model.predict([features])  
    

def update_stats(season, team, fields):
    """
    This accepts some stats for a team and udpates the averages.
    First, we check if the team is in the dict yet. If it's not, we add it.
    Then, we try to check if the key has more than 5 values in it.
        If it does, we remove the first one
        Either way, we append the new one.
    If we can't check, then it doesn't exist, so we just add this.
    Later, we'll get the average of these items.
    """
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        # Make sure we have the field.
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []

        if len(team_stats[season][team][key]) >= 6:
            team_stats[season][team][key].pop()
        team_stats[season][team][key].append(value)


def get_stat(season, team, field):
    try:
        l = team_stats[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0


def build_team_dict():
    team_ids = pd.read_csv(folder + '/Teams.csv')
    team_id_map = {}
    for index, row in team_ids.iterrows():
        team_id_map[row['Team_Id']] = row['Team_Name']
    return team_id_map


def build_season_data(all_data):
    # Calculate the elo for every game for every team, each season.
    # Store the elo per season so we can retrieve their end elo
    # later in order to predict the tournaments without having to
    # inject the prediction into this loop.
    print("Building season data.")
    for index, row in all_data.iterrows():
        # Used to skip matchups where we don't have usable stats yet.
        skip = 0

        # Get starter or previous elos.
        team_1_elo = get_elo(row['Season'], row['Wteam'])
        team_2_elo = get_elo(row['Season'], row['Lteam'])
        # Add 100 to the home team (# taken from Nate Silver analysis.)
        if row['Wloc'] == 'H':
            team_1_elo += 100
        elif row['Wloc'] == 'A':
            team_2_elo += 100

        # We'll create some arrays to use later.
        team_1_features = [team_1_elo]
        team_2_features = [team_2_elo]

        # Build arrays out of the stats we're tracking..
        for field in stat_fields:
            team_1_stat = get_stat(row['Season'], row['Wteam'], field)
            team_2_stat = get_stat(row['Season'], row['Lteam'], field)
            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1

        if skip == 0:  # Make sure we have stats.
            # Randomly select left and right and 0 or 1 so we can train
            # for multiple classes.
            if random.random() > 0.5:
                X.append(team_1_features + team_2_features)
                y.append(0)
            else:
                X.append(team_2_features + team_1_features)
                y.append(1)

        # AFTER we add the current stuff to the prediction, update for
        # next time. Order here is key so we don't fit on data from the
        # same game we're trying to predict.
        if row['Wfta'] != 0 and row['Lfta'] != 0:   
               
            stat_1_fields = {      
                'score': row['Wscore'],
                'fgp': row['Wfgm']/row['Wfga']*100,  
                'fga': row['Wfga'],              
                'fga3': row['Wfga3'],
                '3pp': row['Lfgm3'] / row['Lfga3'] * 100,
                'ftp': row['Lftm'] / row['Lfta'] * 100,
                'or': row['Wor'],
                'dr': row['Wdr'],
                'ast': row['Wast'],
                'to': row['Wto'],
                'stl': row['Wstl'],
                'blk': row['Wblk'],
                'pf': row['Wpf'],
            }

            stat_2_fields = {
                'score': row['Lscore'],
                'fgp': row['Lfgm']/row['Lfga']*100,
                'fga': row['Lfga'],
                'fga3': row['Lfga3'],
                '3pp': row['Lfgm3'] / row['Lfga3'] * 100,
                'ftp': row['Lftm'] / row['Lfta'] * 100,
                'or': row['Lor'],
                'dr': row['Ldr'],
                'ast': row['Last'],
                'to': row['Lto'],
                'stl': row['Lstl'],
                'blk': row['Lblk'],
                'pf': row['Lpf'],
            }

            #'fga3'+ row['Wfga3']+'3pp'+ row['Wfgm3'] / row['Wfga3'] * 100) 
            update_stats(row['Season'], row['Wteam'], stat_1_fields)
            update_stats(row['Season'], row['Lteam'], stat_2_fields)

        # Now that we've added them, calc the new elo.
        new_winner_rank, new_loser_rank = calc_elo(
            row['Wteam'], row['Lteam'], row['Season'])
        team_elos[row['Season']][row['Wteam']] = new_winner_rank
        team_elos[row['Season']][row['Lteam']] = new_loser_rank
        
    return X, y

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adadelta
from keras.regularizers import l1l2
from keras.callbacks import EarlyStopping, Callback
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from theano import function
import numpy as np
import logging

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseMLP(BaseEstimator, ClassifierMixin):
    '''
    Model class that wraps Keras model
    Input:
    in_dim: number of variables of the data
    out_dim: number of classes to predict
    n_hidden: number of hidden variables at each layers
    n_deep: number of layers
    l1_norm: penalization coefficient for L1 norm on hidden variables
    drop: dropout percentage at each layer
    verbose: verbosity level (up to 3 levels)
    Methods:
    reset_weigths: re-initiallizes the weights
    save/load: saves/loads weights to a file
    fit: trains on data provided with early stopping
    train_batch: trains on batch provided splitting
                 the data into train and validation
    fit_batches: trains on a sequence of batches with early stopping
    predict: returns prediction on data provided
    auc: returns area under the roc curve on data and true
         labels provided
    '''
    def __init__(self, n_hidden=1000, n_deep=4,
                 l1_norm=0, l2_norm=0, drop=0,
                 early_stop=True, max_epoch=5000,
                 patience=200,
                 learning_rate=1, verbose=0):
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.n_hidden = n_hidden
        self.n_deep = n_deep
        self.l1_norm = l1_norm
        self.l2_norm = l2_norm
        self.drop = drop
        self.patience = patience
        self.verbose = verbose
        self.learning_rate = learning_rate

    def fit(self, X, y, **kwargs):

        # Encoding labels
        self.le = LabelEncoder()
        self.y_ = self.le.fit_transform(y)
        self.n_class = len(self.le.classes_)

        if self.n_class == 2:
            out_dim = 1
        else:
            out_dim = self.n_class

        if hasattr(self, 'model'):
            self.reset_model()
        else:
            self.build_model(X.shape[1], out_dim)
        if self.verbose:
            temp = [layer['output_dim'] for layer in
                    self.model.get_config()['layers']
                    if layer['name'] == 'Dense']
            print('Model:{}'.format(temp))
            print('l1: {}, drop: {}, lr: {}, patience: {}'.format(
                self.l1_norm, self.drop, self.learning_rate,
                self.patience))

        return self

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def build_model(self, in_dim, out_dim):

        self.model = build_model(in_dim, out_dim=out_dim,
                                 n_hidden=self.n_hidden, l1_norm=self.l1_norm,
                                 l2_norm=self.l2_norm,
                                 n_deep=self.n_deep, drop=self.drop,
                                 learning_rate=self.learning_rate)
        self.w0 = self.model.get_weights()
        return self

    def reset_model(self):
        self.model.set_weights(self.w0)

    def feed_forward(self, X):
        # Feeds the model with X and returns the output of
        # each layer
        layer_output = []
        for layer in self.model.layers:
            if layer.get_config()['name'] == 'Dense':
                get_layer = function([self.model.layers[0].input],
                                     layer.get_output(train=False),
                                     allow_input_downcast=True)
                layer_output.append(get_layer(X))
        return layer_output

    def predict_proba(self, X):
        proba = self.model.predict(X, verbose=self.verbose)
        proba = (proba - proba.min())
        proba = proba/proba.max()
        if proba.shape[1] == 1:
            proba = np.array(proba).reshape((X.shape[0], -1))
            temp = (1-proba.sum(axis=1)).reshape(X.shape[0], -1)
            proba = np.hstack((temp, proba))
        return proba

    def predict(self, X):
        prediction = self.model.predict_classes(X, verbose=self.verbose)
        prediction = np.array(prediction).reshape((X.shape[0], -1))
        prediction = np.squeeze(prediction).astype('int')
        return self.le.inverse_transform(prediction)

    def auc(self, X, y):
        prediction = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, prediction)

    def f1(self, X, y):
        prediction = self.predict(X)
        if self.n_class > 2:
            return f1_score(y, prediction, average='weighted')
        else:
            return f1_score(y, prediction)


class TestLossHistory(Callback):

    def __init__(self, X_test, y_test, *args, **kwargs):
        super(TestLossHistory, self).__init__(*args, **kwargs)
        self.X_test = X_test
        self.y_test = y_test

    def on_train_begin(self, logs={}):
        self.test_losses = []

    def on_epoch_end(self, batch, logs={}):
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0,
                                   batch_size=self.X_test.shape[0])
        self.test_losses.append(loss)


class MLP(BaseMLP):

    def fit(self, X, y, X_test=None, y_test=None):
        super(MLP, self).fit(X, y)

        callbacks = []
        test = X_test is not None and y_test is not None
        if test:
            self.test_loss = TestLossHistory(X_test, y_test)
            callbacks.append(self.test_loss)

        if self.n_class > 2:
            y = unroll(self.y_)
        else:
            y = self.y_

        if self.early_stop:
            sss = StratifiedShuffleSplit(self.y_, 1, test_size=0.1,
                                         random_state=0)
            train_index, val_index = next(iter(sss))
            x_train, x_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            stop = EarlyStopping(monitor='val_loss',
                                 patience=self.patience,
                                 verbose=self.verbose)
            callbacks.append(stop)

            self.history = self.model.fit(
                x_train, y_train, nb_epoch=self.max_epoch,
                verbose=self.verbose, callbacks=callbacks,
                validation_data=(x_val, y_val))

        else:
            self.history = self.model.fit(
                X, y, nb_epoch=self.max_epoch, verbose=self.verbose,
                callbacks=callbacks)

        return self


def build_model(in_dim, out_dim=1,
                n_hidden=100, l1_norm=0.0,
                l2_norm=0,
                n_deep=5, drop=0.1,
                learning_rate=0.1):
    model = Sequential()
    # Input layer
    model.add(Dense(
        input_dim=in_dim,
        output_dim=n_hidden,
        init='glorot_normal',
        activation='tanh',
        W_regularizer=l1l2(l1=l1_norm, l2=l2_norm)))

    # do X layers
    for layer in range(n_deep-1):
        model.add(Dropout(drop))
        model.add(Dense(
            output_dim=np.round(n_hidden/2**(layer+1)),
            init='glorot_normal',
            activation='tanh',
            W_regularizer=l1l2(l1=l1_norm, l2=l2_norm)))

    # Output layer
    if out_dim == 1:
        activation = 'tanh'
    else:
        activation = 'softmax'

    model.add(Dense(out_dim,
                    init='glorot_normal',
                    activation=activation))

    # Optimization algorithms
    opt = Adadelta(lr=learning_rate)
    if out_dim == 1:
        model.compile(loss='binary_crossentropy',
                      optimizer=opt)
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt)

    return model


def unroll(y):
    n_class = len(np.unique(y))
    return np.array([np.roll([1] + [0]*(n_class-1), pos) for pos in y])

# using softmax as output layer is recommended for classification where outputs are mutually exclusive
def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)

# using tanh over logistic sigmoid is recommended   
def tanh(x):
    return np.tanh(x)
    
# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y*y
errorlist = []
class MLP_Classifier(object):
    """
    Basic MultiLayer Perceptron (MLP) neural network with regularization and learning rate decay
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm can be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    """
    def __init__(self, input, hidden, output, iterations = 50, learning_rate = 0.01, 
                l2_in = 0, l2_out = 0, momentum = 0, rate_decay = 0, 
                output_layer = 'logistic', verbose = True):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        :param iterations: how many epochs
        :param learning_rate: initial learning rate
        :param l2: L2 regularization term
        :param momentum: momentum
        :param rate_decay: how much to decrease learning rate by on each iteration (epoch)
        :param output_layer: activation (transfer) function of the output layer
        :param verbose: whether to spit out error rates while training
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_in = l2_in
        self.l2_out = l2_out
        self.momentum = momentum
        self.rate_decay = rate_decay
        self.verbose = verbose
        self.output_activation = output_layer
        
        # initialize arrays
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden 
        self.output = output

        # set up array of 1s for activations
        self.ai = np.ones(self.input)
        self.ah = np.ones(self.hidden)
        self.ao = np.ones(self.output)

        # create randomized weights
        # use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
        input_range = 1.0 / self.input ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
        
        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        self.ai[0:self.input -1] = inputs

        # hidden activations
        sum = np.dot(self.wi.T, self.ai)
        self.ah = tanh(sum)
        
        # output activations
        sum = np.dot(self.wo.T, self.ah)
        if self.output_activation == 'logistic':
            self.ao = sigmoid(sum)
        elif self.output_activation == 'softmax':
            self.ao = softmax(sum)
        else:
            raise ValueError('Choose a compatible output layer activation or check your spelling ;-p') 
        
        
        return self.ao

    def backPropagate(self, targets):
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative
        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta (theta) tell you which direction to change the weights
        if self.output_activation == 'logistic':
            output_deltas = dsigmoid(self.ao) * -(targets - self.ao)
        elif self.output_activation == 'softmax':
            output_deltas = -(targets - self.ao)
        else:
            raise ValueError('Choose a compatible output layer activation or check your spelling ;-p') 
        
        # calculate error terms for hidden
        # delta (theta) tells you which direction to change the weights
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = dtanh(self.ah) * error
        
        # update the weights connecting hidden to output, change == partial derivative
        change = output_deltas * np.reshape(self.ah, (self.ah.shape[0],1))
        regularization = self.l2_out * self.wo
        self.wo -= self.learning_rate * (change + regularization) + self.co * self.momentum 
        self.co = change 

        # update the weights connecting input to hidden, change == partial derivative
        change = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        regularization = self.l2_in * self.wi
        self.wi -= self.learning_rate * (change + regularization) + self.ci * self.momentum 
        self.ci = change

        # calculate error
        if self.output_activation == 'softmax':
            error = -sum(targets * np.log(self.ao))
        elif self.output_activation == 'logistic':
            error = sum(0.5 * (targets - self.ao)**2)                    
            
        return error

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))
    import time
    import random
    def fit(self, patterns):
        if self.verbose == True:
            if self.output_activation == 'softmax':
                print ('Using softmax activation in output layer')
            elif self.output_activation == 'logistic':
                print ('Using logistic sigmoid activation in output layer')
                
        num_example = np.shape(patterns)[0]
                
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
                
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
                
            if i % 10 == 0 and self.verbose == True:
                error = error/num_example
                print('Training error %-.5f' % error)
                errorlist.append(error)
            # learning rate decay
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
                
    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions
    
class MLP_NeuralNetwork(object):
    """
    Basic MultiLayer Perceptron (MLP) network, adapted and from the book 'Programming Collective Intelligence' (http://shop.oreilly.com/product/9780596529321.do)
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm has been generalized to be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    """
    def __init__(self, input, hidden, output, iterations, learning_rate, momentum, rate_decay):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay
        
        # initialize arrays
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output

        # create randomized weights
        # use scheme from 'efficient backprop to initialize weights
        input_range = 1.0 / self.input ** (1/2)
        output_range = 1.0 / self.hidden ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.normal(loc = 0, scale = output_range, size = (self.hidden, self.output))
        
        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        for i in range(self.input -1): # -1 is to avoid the bias
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        # output activations
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets):
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative
        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        # update the weights connecting hidden to output
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] -= self.learning_rate * change + self.co[j][k] * self.momentum
                self.co[j][k] = change

        # update the weights connecting input to hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] -= self.learning_rate * change + self.ci[i][j] * self.momentum
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))

    def train(self, patterns):
        # N: learning rate
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
            if i % 10 == 0:
                print('error %-.5f' % error)
            # learning rate decay
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
                
    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions















if __name__ == "__main__":
    stat_fields = ['score','fgp', 'fga', 'fga3','3pp', 'ftp', 'or', 'dr',
                   'ast', 'to', 'stl', 'blk', 'pf']

    initialize_data()
    season_data = pd.read_csv(folder + '/RegularSeasonDetailedResults.csv')
    tourney_data = pd.read_csv(folder + '/TourneyDetailedResults.csv')
    frames = [season_data, tourney_data]
    all_data = pd.concat(frames)
    
    
    # Build the working data.
    X, y = build_season_data(all_data)
    
    from sklearn.preprocessing import normalize
    X1 = normalize(X)
    X2 = np.array(X1)
    y2 = np.array(y)
    cut = int(len(X2) * 0.6)
    Xcut = X1[:cut]
    ycut = y2[:cut] 


    # Fit the model.
    print("Fitting on %d samples." % len(X))

    model = linear_model.LogisticRegression()
    # Check accuracy.
    print("Doing cross-validation.")
    print(cross_validation.cross_val_score(
        model, X1, y, cv=2, scoring='log_loss', n_jobs=1
    ).mean())
    model.fit(X, y)
    
    tuned_parameters = {'penalty': ['l1', 'l2'],
                     'C': [10, 50, 100]}
    from sklearn.grid_search import RandomizedSearchCV            

    logreg = linear_model.LogisticRegression(penalty = 'l2', C = 1000)
   
    print(cross_validation.cross_val_score(
        logreg, X1, y, cv=3, scoring='accuracy', n_jobs=1
    ).mean())    
    logreg.fit(X1, y2)
    from sklearn import svm
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X2, y)
        # Now predict tournament matchups.
    print("Getting teams.")
    seeds = pd.read_csv(folder + '/TourneySeeds.csv')
    # for i in range(2016, 2017):
    tourney_teams = []
    for index, row in seeds.iterrows():
        if row['Season'] == prediction_year:
            tourney_teams.append(row['Team'])

    # Build our prediction of every matchup.
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                prediction = predict_winner(
                    team_1, team_2, clf, prediction_year, stat_fields)
                label = str(prediction_year) + '_' + str(team_1) + '_' + \
                    str(team_2)
                submission_data.append([label, prediction[0][0]])

    # Write the results.
    print("Writing %d results." % len(submission_data))
    with open(folder + '/submission6.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data)
    
    
    
    
    
    
    
    
    
#    print("Best parameters set found on development set:")
#    print()
#    print(grid_log.best_params_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    for params, mean_score, scores in grid_log.grid_scores_:
#        print("%0.3f (+/-%0.03f) for %r"
#                % (mean_score, scores.std() * 2, params))
#    print()
#    
    
    
    model_log = linear_model.LogisticRegression(penalty = 'l1', C = 50)
    print(cross_validation.cross_val_score(
        model_log, X1, y, cv=3, scoring='accuracy', n_jobs=1
    ).mean()) 
    model_log.fit(X,y)
    # Now predict tournament matchups.
    print("Getting teams.")
    seeds = pd.read_csv(folder + '/TourneySeeds.csv')
    # for i in range(2016, 2017):
    tourney_teams = []
    for index, row in seeds.iterrows():
        if row['Season'] == prediction_year:
            tourney_teams.append(row['Team'])

    # Build our prediction of every matchup.
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                prediction = predict_winner(
                    team_1, team_2, model_log, prediction_year, stat_fields)
                label = str(prediction_year) + '_' + str(team_1) + '_' + \
                    str(team_2)
                submission_data.append([label, prediction[0][0]])

    # Write the results.
    print("Writing %d results." % len(submission_data))
    with open(folder + '/submission.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data)
        
        

    # Now so that we can use this to fill out a bracket, create a readable
    # version.
    print("Outputting readable results.")
    team_id_map = build_team_dict()
    readable = []
    less_readable = []  # A version that's easy to look up.
    for pred in submission_data:
        parts = pred[0].split('_')
        less_readable.append(
            [team_id_map[int(parts[1])], team_id_map[int(parts[2])], pred[1]])
        # Order them properly.
        if pred[1] > 0.5:
            winning = int(parts[1])
            losing = int(parts[2])
            proba = pred[1]
        else:
            winning = int(parts[2])
            losing = int(parts[1])
            proba = 1 - pred[1]
        readable.append(
            [
                '%s beats %s: %f' %
                (team_id_map[winning], team_id_map[losing], proba)
            ]
        )
    with open(folder + '/readable-predictions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(readable)
    with open(folder + '/less-readable-predictions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(less_readable)           
    
       
    model_NB = GaussianNB()
    scores = ['accuracy', 'log_loss']
    for score in scores:
        print(cross_validation.cross_val_score(
            model_NB, X1, y2, cv=10, scoring=score, n_jobs=1))
        model_NB.fit(X, y)
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                prediction2 = predict_winner(
                    team_1, team_2, model_NB, prediction_year, stat_fields)
                label = str(prediction_year) + '_' + str(team_1) + '_' + \
                    str(team_2)                
                submission_data2.append([label, prediction2[0][0]])

    
    # Write the results
    print("Writing %d results." % len(submission_data2))
    with open(folder + '/submission2.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data2)

#    from sklearn.grid_search import GridSearchCV            
#    scores = ['accuracy', 'log_loss']
#    
#    from sklearn.calibration import CalibratedClassifierCV
#    a = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
#    iterator = [5, 10, 15, 20, 50]
#    perscores = []
#    for score in scores:
#        for vals in a:
#            for it in iterator:
#                model_per = linear_model.Perceptron(alpha = a, scoring = score, penalty = 'elasticnet', n_iter = iterator)
#                clf_isotonic = CalibratedClassifierCV(model_per, cv=10, method='isotonic')    
#                print(cross_validation.cross_val_score(clf_isotonic, X, y, cv=10, scoring='log_loss', n_jobs=-1))
#                clf_isotonic.fit(X, y)
#                
#
#
#    
#    print("Predicting matchups.")
#    tourney_teams.sort()
#    for team_1 in tourney_teams:
#        for team_2 in tourney_teams:
#            if team_1 < team_2:
#                prediction3 = predict_winner(
#                    team_1, team_2, model_per, prediction_year, stat_fields)
#
#
#                label = str(prediction_year) + '_' + str(team_1) + '_' + \
#                    str(team_2)
#                    #took out one of the brackets bceause numpy is 1d and not 2d
#                submission_data3.append([label, prediction3[0][0]])
#        
#    
#    # Write the results
#    print("Writing %d results." % len(submission_data3))
#    with open(folder + '/submission3.csv', 'w') as f:
#        writer = csv.writer(f)
#        writer.writerow(['id', 'pred'])
#        writer.writerows(submission_data3)  
    
    y3 = y2.reshape(-1, 1)     
    out_train = []
    out_test = []
    from sklearn.cross_validation import train_test_split
    X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(X2, y3, test_size=0.33, random_state=42)
    for i in range(X_nn_train.shape[0]):
        tuppledata = list((X_nn_train[i].tolist(), y_nn_train[i].tolist())) # don't mind this variable name
#        tuppledata = list((X2[i].tolist(), y3[i].tolist()))
        out_train.append(tuppledata)
    for i in range(X_nn_test.shape[0]):
        db = list((X_nn_test[i].tolist(), y_nn_test[i].tolist())) # don't mind this variable nam
        out_test.append(db)
    
    for i in out_train:
        print(len(i[0]))
    k_range = [1, 3]
    for k in k_range:
        NN = MLP_Classifier(28, 10, 1, iterations = 100, learning_rate = 1, momentum = 0.9, rate_decay = 0.01, output_layer = 'logistic')
        errorlist = []  
        NN.fit(out_train)
        
        
    errorlist = []    
    NN.test(out_test)                    
    import matplotlib.pyplot as plt
    plt.scatter(range(1, 10), errorlist)
    
    for i in out_train:
        print(len(i[0]))
    submission_data4=[]
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                prediction4 = predict_winnerNN(
                    team_1, team_2, NN, prediction_year, stat_fields)
                label = str(prediction_year) + '_' + str(team_1) + '_' + \
                    str(team_2)
                submission_data4.append([label, prediction4[0][0]])

    # Write the results
    print("Writing %d results." % len(submission_data4))
    with open(folder + '/submission4.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data4)
        
    
    
    model_log = linear_model.LogisticRegression(penalty = 'l1', C = 50, solver)
    model_NB = GaussianNB()        
    
    scores = ['accuracy', 'log_loss']
    from sklearn.ensemble import VotingClassifier
    
        votingEnsemble = VotingClassifier(estimators=[('lr', model_log), ('gnb', model_NB)], voting = 'soft', weights = [2, 1])
        print(cross_validation.cross_val_score(votingEnsemble, X1, y2, cv=3, scoring='log_loss', n_jobs=1).mean())    
        print()    

    votingEnsemble = VotingClassifier(estimators=[('lr', model_log), ('gnb', model_NB)], voting = 'soft', weights = [2,1])
    votingEnsemble.fit(X1, y2)
    submission_data4=[]
    submission_data5=[]
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                prediction5 = predict_winner(
                    team_1, team_2, votingEnsemble, prediction_year, stat_fields)
                label = str(prediction_year) + '_' + str(team_1) + '_' + \
                    str(team_2)
                submission_data4.append([label, prediction5[0][0]])
    
    # Write the results
    print("Writing %d results." % len(submission_data4))
    with open(folder + '/submission5.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data4)
    
    from sklearn.ensemble1 import VotingClassifier
    votingEnsemble1 = VotingClassifier(estimators=[('lr', model_log), ('gnb', model_NB)], voting = 'soft')
    votingEnsemble1.fit()
    
    

    print("Writing %d results." % len(submission_data5))
    with open(folder + '/submission5.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data5)
    
    from sklearn.ensemble import AdaBoostRegressor
    boostingEnsemble = AdaBoostRegressor(base_estimator=model_log, n_estimators = 200)
    
    
    print(cross_validation.cross_val_score(boostingEnsemble, X1, y2, cv=3, scoring='log_loss', n_jobs=1).mean())    
    print() 
    boostingEnsemble.fit(X1, y2)
    
    

    

    submission_data4=[]
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                prediction4 = predict_winner(
                    team_1, team_2, boostingEnsemble, prediction_year, stat_fields)
                label = str(prediction_year) + '_' + str(team_1) + '_' + \
                    str(team_2)
                submission_data4.append([label, prediction4[0][0]])
    
    # Write the results
    print("Writing %d results." % len(submission_data4))
    with open(folder + '/submission4.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data4)
    
    
    
    
    
    
    
    
    
    
    
    

        
