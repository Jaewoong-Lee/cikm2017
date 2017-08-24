"""
Matrix Factorization with Theano implementation
Theano: http://deeplearning.net/software/theano/
"""

import numpy as np

from util.load_data import build_ml_1m_train_binary4
from util.load_data import build_ml_1m_test_binary4

from util.load_data import build_ml_1m_train_binary3
from util.load_data import build_ml_1m_test_binary3

from util.load_data import build_ml_1m_train_binary2
from util.load_data import build_ml_1m_test_binary2

from util.load_data import build_ml_1m_train_binary1
from util.load_data import build_ml_1m_test_binary1

from util.load_data import build_ml_1m_train_binary5
from util.load_data import build_ml_1m_test_binary5

from util.load_data import build_user_item_matrix

from util.evaluation_metrics import RMSE
from util.evaluation_metrics import PrecAt
from util.evaluation_metrics import AUC
from util.evaluation_metrics import RecallAt
from util.evaluation_metrics import nDCGAt
from util.evaluation_metrics import MapAt

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import time
import theano
from theano import shared
import theano.tensor as T
from _ast import operator
from matplotlib.pyplot import axis


class CDAE_ML_1m():

    def __init__(self, num_user, num_item, train, test, n_hidden, **params):
        super(CDAE_ML_1m, self).__init__()
        self._num_user = num_user
        self._num_item = num_item
        
        # learning rate
        self.learning_rate = float(params.get('learning_rate', 0.2))
        
        self.lam = float(params.get('lam', 0.03))
        
        # About DAE
        n_visible = num_item
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        numpy_rng = np.random.RandomState(123)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        initial_W = np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                size=(n_visible, n_hidden)
            ),
            dtype=theano.config.floatX
        )
        
        W = theano.shared(value=initial_W, name='W', borrow=True)
        
        initial_V = np.asarray(
            numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                size=(num_user, n_hidden)
            ),
            dtype=theano.config.floatX
        )
        
        V = theano.shared(value=initial_V, name='V', borrow=True) # user bias term
        
        bvis = theano.shared(
            value=np.zeros(
                n_visible,
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        bhid = theano.shared(
            value=np.zeros(
                n_hidden,
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        
        self.x = T.matrix(name='input')
        self.o = T.matrix(name='output')
        
        self.W = W
        self.V = V
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        self.theano_rng = theano_rng

        self.train = shared(np.matrix(train, dtype=theano.config.floatX), borrow=True)
        self.test = shared(np.matrix(test, dtype=theano.config.floatX), borrow=True)

        self.params = [self.W, self.V, self.b, self.b_prime]

    @property
    def user(self):
        return self._num_user

    @property
    def items(self):
        return self._num_item

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.V + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_ranking_updates(self, corruption_level):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L = - T.sum( self.o * T.log(z) + (1 - self.o) * T.log(1 - z), axis=1)

        regularization = T.sum( (self.W**2).sum() + (self.V**2).sum() +
                                (self.b**2).sum() + (self.b_prime**2).sum() )
            
        # loss
        loss = L + 0.5 * self.lam * regularization

        cost = T.mean(loss)

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        
        return (cost, updates)


    def estimate(self, iterations=300, corrupt_value=0.0):     
        
        cost, updates = self.get_cost_ranking_updates(
            corruption_level=corrupt_value
        )

        train_da = theano.function(
            [],
            cost,
            updates=updates,
            givens={
                self.x: self.train,
                self.o: self.train
            }
        )
        
        for iteration in range(iterations):
            train_cost = train_da()                    
            print ("iterations: %3d, cost: %.6f"  % \
                (iteration + 1, train_cost))
        
        
        train_preds = self.predict(self.train)
        
        #####################################
        ns = [5, 10, 15]
        precisions = np.zeros(len(ns))
        recalls = np.zeros(len(ns))
        ndcgs = np.zeros(len(ns))
        maps = np.zeros(len(ns))
        auc = 0
        numberOfUsers = 0

        for i in range(self._num_user):
        
            train_data = self.train[i].eval()
            missing_index = np.where(train_data == 0)
            test_data = self.test[i].eval()
            #groundTruth = np.where(test_data == 1)
            groundTruth = np.where(test_data > 0.5)
            
            if len(groundTruth[0]) == 0 :
                continue

            if len(np.where(train_data > 0)) == 0 :
                continue

            numberOfUsers += 1
          
            train_predicted = train_preds[i].eval()

            predicted = train_preds[i].eval()
            missing_index = missing_index[0]
            predicted_missingdata = predicted[missing_index]
                
            index = np.argsort(predicted_missingdata)
            missing_index = np.take(missing_index, index)
                
            rankedList = missing_index[::-1] # reverse


            precisions += PrecAt(rankedList, groundTruth[0], ns)
            maps += MapAt(rankedList, groundTruth[0], ns)
            ndcgs += nDCGAt(rankedList, groundTruth[0], ns)

            numDropped = self._num_item - len(rankedList)
            auc += AUC(rankedList, groundTruth[0], numDropped)

            
        precisions /= float(numberOfUsers)
        ndcgs /= float(numberOfUsers)
        maps /= float(numberOfUsers)
        auc /= float(numberOfUsers)

        print ("iterations: %3d, pre5: %.6f, pre10: %.6f, pre15: %.6f, ndcg5: %.6f, ndcg10: %.6f, ndcg15: %.6f, map5: %.6f, map10: %.6f, map15: %.6f, auc: %.6f" % \
            (iteration + 1, precisions[0], precisions[1], precisions[2], ndcgs[0], ndcgs[1], ndcgs[2], maps[0], maps[1], maps[2], auc))
        
        f = open("./result/resultAE.txt",'a')

        data = ("iterations: %3d, pre5: %.6f, pre10: %.6f, pre15: %.6f, ndcg5: %.6f, ndcg10: %.6f, ndcg15: %.6f, map5: %.6f, map10: %.6f, map15: %.6f, auc: %.6f" % \
            (iteration + 1, precisions[0], precisions[1], precisions[2], ndcgs[0], ndcgs[1], ndcgs[2], maps[0], maps[1], maps[2], auc))
        
        f.write(data)
        f.close()

    def predict(self, input):
        y = self.get_hidden_values(input)
        z = self.get_reconstructed_input(y)
        return z


def example():
    """simple test and performance measure
    """
    
    num_user = 6041
    num_item = 3953
    
    num_hidden = 500
    iterations = 501
    
    train = build_ml_1m_train_binary2()
    test = build_ml_1m_test_binary2()

    train_matrix = build_user_item_matrix(num_user, num_item, train)
    test_matrix = build_user_item_matrix(num_user, num_item, test)

    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()

    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]
    
    num_user = num_user-1
    num_item = num_item-1

    mf_model = CDAE_ML_1m(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, num_hidden)

    mf_model.estimate(iterations,0.2)


    train = build_ml_1m_train_binary4()
    test = build_ml_1m_test_binary4()
    
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()
    
    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]
    
    
    mf_model = CDAE_ML_1m(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, num_hidden)

    mf_model.estimate(iterations,0.2) 
    



    train = build_ml_1m_train_binary5()
    test = build_ml_1m_test_binary5()
    
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)    
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()
    
    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]
    
    
    mf_model = CDAE_ML_1m(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, num_hidden)

    mf_model.estimate(iterations,0.2) 



    train = build_ml_1m_train_binary3()
    test = build_ml_1m_test_binary3()
    
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)    
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()
    
    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]   
    
    
    mf_model = CDAE_ML_1m(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, num_hidden)

    mf_model.estimate(iterations,0.2) 
    


    train = build_ml_1m_train_binary1()
    test = build_ml_1m_test_binary1()
    
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)    
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()
    
    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]
    
    
    mf_model = CDAE_ML_1m(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, num_hidden)

    mf_model.estimate(iterations,0.2)
 

    return mf_model
    
    

if __name__ == "__main__":
    example()
    
    
    
    
