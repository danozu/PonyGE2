# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:12:55 2020

@author: allan
"""

from utilities.algorithm.general import check_python_version
check_python_version()

import numpy as np
import pandas as pd
import math
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets

#from stats.stats import get_stats
from algorithm.parameters import params, set_params,load_params
from utilities.fitness.math_functions import *
import sys

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import random
__all__ = ['ponyge']

class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 CROSSOVER_PROBABILITY=0.75,
                 #ERROR_METRIC='f1_score',
                 GENERATIONS = 10,
                 POPULATION_SIZE=10,
                 MAX_INIT_TREE_DEPTH=10,
                 MAX_TREE_DEPTH=17,
                 RANDOM_SEED=7,
    #             data_address='Sarcoidose/Train1.csv',
                 ):
        self.CROSSOVER_PROBABILITY = CROSSOVER_PROBABILITY
       # self.ERROR_METRIC = ERROR_METRIC
        self.GENERATIONS = GENERATIONS
        self.POPULATION_SIZE = POPULATION_SIZE
        self.MAX_INIT_TREE_DEPTH = MAX_INIT_TREE_DEPTH
        self.MAX_TREE_DEPTH = MAX_TREE_DEPTH
        self.RANDOM_SEED = RANDOM_SEED
 #       self.data_address = data_address


    def fit(self, X, y):
        """
        """
        if isinstance(self, ClassifierMixin):
            X, y = check_X_y(X, y, y_numeric=False)
            check_classification_targets(y)

            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes = len(self.classes_)

        else:
            X, y = check_X_y(X, y, y_numeric=True)
            
        l, c = np.shape(X)
        self.n_features_ = c

        i = random.randint(0, 1e10)

#        afile = "/home/allan/Dropbox/Mestrado - PEL/Pesquisa/sarcoidose/PonyGE2-master/grammars/supervised_learning/Sarcoidose" + str(i) +".bnf"
#        texto = open(afile,'w')#.readlines()
#        texto.write("<e> ::= <op>(<e>, <e>) | <op>(<e>, <c>) | x[<idx>]" + '\n')
#        texto.write("<op> ::= add | mul | sub | pdiv" + '\n')
##        if c == 4:
#            texto.write("<idx> ::= 0 | 1 | 2 | 3" + '\n')
#        elif c == 8:
#            texto.write("<idx> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7" + '\n')
#        elif c == 12:
#            texto.write("<idx> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11" + '\n')
#        texto.write("<c> ::= 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9")
#        texto.close()

        data = np.empty([l,c+1], dtype=float)
        data[:,0:c] = X
        data[:,c] = y
        
        head = []
        for j in range(c):
            head.append('x'+str(j))
        head.append('class')



        pd.DataFrame(data).to_csv("C:/Users/PICHAU/Dropbox/Mestrado - PEL/Pesquisa/sarcoidose/PonyGE2-master/datasets/Sarcoidose/Train" + str(i) + ".csv", header=head, sep=" ", index=None)
        pd.DataFrame(data).to_csv("C:/Users/PICHAU/Dropbox/Mestrado - PEL/Pesquisa/sarcoidose/PonyGE2-master/datasets/Sarcoidose/Test" + str(i) + ".csv", header=head, sep=" ", index=None)
        #pd.DataFrame(data).to_csv(
        #    "/home/allan/Dropbox/Mestrado - PEL/Pesquisa/sarcoidose/PonyGE2-master/datasets/Sarcoidose/Train" + str(
        #        i) + ".csv", header=head, sep=" ", index=None)
        #pd.DataFrame(data).to_csv(
        #    "/home/allan/Dropbox/Mestrado - PEL/Pesquisa/sarcoidose/PonyGE2-master/datasets/Sarcoidose/Test" + str(
        #        i) + ".csv", header=head, sep=" ", index=None)


        load_params('C:/Users/PICHAU/Dropbox/Mestrado - PEL/Pesquisa/sarcoidose/PonyGE2-master/parameters/classification.txt')
        #load_params(
        #    '/home/allan/Dropbox/Mestrado - PEL/Pesquisa/sarcoidose/PonyGE2-master/parameters/classification.txt')
        params['CROSSOVER_PROBABILITY'] = self.CROSSOVER_PROBABILITY
        params['POPULATION_SIZE'] = self.POPULATION_SIZE
        params['GENERATIONS'] = self.GENERATIONS
        params['MAX_INIT_TREE_DEPTH'] = self.MAX_INIT_TREE_DEPTH
        params['MAX_TREE_DEPTH'] = self.MAX_TREE_DEPTH
        params['RANDOM_SEED'] = self.RANDOM_SEED
        params['DATASET_TRAIN'] = 'Sarcoidose/Train'+ str(i) + '.csv'
        params['DATASET_TEST'] = 'Sarcoidose/Test'+ str(i) + '.csv'
        params['SAVE_PLOTS'] = False
        params['CACHE'] = False
        params['SILENT'] = True
        params['INITIALISATION'] = 'rhh'

#        params['GRAMMAR_FILE'] = 'supervised_learning/Sarcoidose'+ str(i) + '.bnf'

        
      
        
        
        
        set_params (sys.argv[1:])  # exclude the ponyge.py arg itself
        
        self.individuals = params['SEARCH_LOOP']()
        
        self.best_individual = max(self.individuals)
        
        self.phenotype = self.best_individual.phenotype
        
        return self
 
class ponyge(BaseSymbolic, ClassifierMixin):
    """
    """
    def __init__(self,
                 CROSSOVER_PROBABILITY=0.75,
                 #ERROR_METRIC='f1_score',
                 GENERATIONS = 10,
                 POPULATION_SIZE=10,
                 MAX_INIT_TREE_DEPTH=10,
                 MAX_TREE_DEPTH=17,
                 RANDOM_SEED=7,
      #           data_address='Sarcoidose/Train1.csv',
                 ):
         super(ponyge, self).__init__(
             CROSSOVER_PROBABILITY = CROSSOVER_PROBABILITY,
             # self.ERROR_METRIC = ERROR_METRIC,
             GENERATIONS = GENERATIONS,
             POPULATION_SIZE = POPULATION_SIZE,
             MAX_INIT_TREE_DEPTH = MAX_INIT_TREE_DEPTH,
             MAX_TREE_DEPTH = MAX_TREE_DEPTH,
             RANDOM_SEED = RANDOM_SEED)#,
     #        data_address = data_address)

    def predict_proba(self, X):
        """Predict probabilities on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        proba : array, shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.

        """
  #      if not hasattr(self, '_program'):
  #          raise NotFittedError('SymbolicClassifier not fitted.')
            
        #X = check_array(X)
        _, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_, n_features))
            
        l, c = np.shape(X)
        
        yhat = np.zeros([l], dtype=float)

        for i in range(l):
            converter = {
                'add': add,
                'mul': mul,
                'sub': sub,
                'pdiv': pdiv,
                'WA': WA,
                'OWA': OWA,
                'minimum': minimum,
                'maximum': maximum,
                'dilator': dilator,
                'concentrator': concentrator,
                'x': X[i]
            }
            yhat[i] = eval(self.best_individual.phenotype, {}, converter)

        fuzzy = 1
        #if max(yhat) <= 1 and min(yhat) >= 0:
        if fuzzy == 1:
            proba = np.vstack([1 - yhat, yhat]).T
        else:
            sigmoid = np.zeros([len(yhat)], dtype=float)
            for i in range(len(yhat)):
                try:
                    sigmoid[i] = 1 / (1 + math.exp(-yhat[i]))
                except OverflowError:
                    s = np.sign(yhat[i])*float("inf")
                    sigmoid[i] = 1 / (1 + math.exp(-s))
            proba = np.vstack([1 - sigmoid, sigmoid]).T
        #pd.DataFrame(data).to_csv("C:/Users/PICHAU/Dropbox/Mestrado - PEL/Pesquisa/PonyGE2-master/datasets/Sarcoidose/Train1.csv", sep=" ", header=cols, index=None)
            
     #   yhat = eval(self.best_individual.phenotype)

   #     X_built = self._program.construct(X, self.n_new_features)

   #     proba = self.model_fitness.predict_proba(X_built)


        return proba

    def predict(self, X):
        """Predict classes on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples,]
            The predicted classes of the input samples.

        """
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)
        #y_pred = np.zeros([len(X)], dtype=int)
        #for i in range(len(X)):
        #    y_pred[i] = 1 if proba[i,1] > proba[i,0] else 0
        #Sreturn y_pred
    
#    def score(self, y, y_pred, metric):
#        return _fitness_map[metric](y, y_pred)
        
        
        