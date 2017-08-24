"""
Matrix Factorization with Theano implementation
Theano: http://deeplearning.net/software/theano/
"""

import numpy as np

import theano.sandbox.cuda
import os

from util.load_data import build_ml_100k_train_binary4
from util.load_data import build_ml_100k_test_binary4

from util.load_data import build_ml_100k_train_binary3
from util.load_data import build_ml_100k_test_binary3

from util.load_data import build_ml_100k_train_binary2
from util.load_data import build_ml_100k_test_binary2

from util.load_data import build_ml_100k_train_binary1
from util.load_data import build_ml_100k_test_binary1

from util.load_data import build_ml_100k_train_binary5
from util.load_data import build_ml_100k_test_binary5

from util.load_data import build_user_item_matrix

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

class IDAE_ML_100k():

    def __init__(self, num_user, num_item, train, test, output, n_hidden, **params):
        super(IDAE_ML_100k, self).__init__()
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
        self.alpha = T.matrix(name='alpha')
                
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        self.theano_rng = theano_rng

        self.train = shared(np.matrix(train, dtype=theano.config.floatX), borrow=True)
        self.test = shared(np.matrix(test, dtype=theano.config.floatX), borrow=True)
        self.output = shared(np.matrix(output, dtype=theano.config.floatX), borrow=True)

        self.params = [self.W, self.b, self.b_prime]



    def user(self):
        return self._num_user


    def items(self):
        return self._num_item

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_ranking_updates(self, corruption_level):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L = - T.sum( self.alpha*(self.o * T.log(z) + (1 - self.o) * T.log(1 - z)), axis=1)

        regularization = T.sum( (self.W**2).sum()  +
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



    def preTrain(self, iterations=200, imputation_ratio=0.02, batch_size=943):
        
        index = T.lscalar()        
        n_train_batches = self._num_user // batch_size        
        
        alpha_matrix  = np.asarray(self.train.eval())
        alpha_matrix = alpha_matrix*0 +1 # all 1
        alpha_matrix = shared(np.matrix(alpha_matrix, dtype=theano.config.floatX), borrow=True)        
        
        
        cost, updates = self.get_cost_ranking_updates(
            corruption_level=0.
        )

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                self.x: self.train[index * batch_size: (index + 1) * batch_size],
                self.o: self.output[index * batch_size: (index + 1) * batch_size],
                self.alpha: alpha_matrix[index * batch_size: (index + 1) * batch_size]
            }
        )

        for iteration in range(iterations):
            train_cost = 0
            for batch_index in range(n_train_batches):
                train_cost = train_da(batch_index)
                    
            print ("iterations: %3d, cost: %.6f"  % \
                (iteration + 1, train_cost))

        predicted_rating = self.predict(self.train)
        train_matrix_ = np.asarray(self.train.eval())
        
        
        # imputation 
        predicted_rating = predicted_rating.eval()
    
        for i in range(self._num_user) :
            missing_index = np.where(train_matrix_[i,:] == 0)
            missing_index = missing_index[0]
            
            predicted_missing = predicted_rating[i,:]
            predicted_missing = predicted_missing[missing_index]
            
            sorted_missing = np.sort(predicted_missing)
            sorted_missing = sorted_missing[::-1] # reverse
            
            
            constrain = sorted_missing[int(round(self._num_item *imputation_ratio ))]
            
            for j in range(self._num_item) :
                if (predicted_rating[i,j] > constrain or train_matrix_[i,j] > 0.5):
                    predicted_rating[i,j] = 1            
                else :
                    predicted_rating[i,j] = 0   
                
                    
        return predicted_rating



    def estimate(self, iterations=300, alpha_value=1, batch_size=943):

        index = T.lscalar()
        n_train_batches = self._num_user // batch_size        
        
        output_matrix  = np.asarray(self.output.eval())
        alpha_matrix  = np.asarray(self.train.eval())
        
        alpha_matrix = output_matrix - alpha_matrix # remains only imputated items
        alpha_matrix = alpha_matrix*(1-alpha_value)*(-1) + 1 # imtutated items change to alpha and the others are 1
        
        alpha_matrix = shared(np.matrix(alpha_matrix, dtype=theano.config.floatX), borrow=True)              
        
        
        cost, updates = self.get_cost_ranking_updates(
            corruption_level=0.
        )

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                self.x: self.train[index * batch_size: (index + 1) * batch_size],
                self.o: self.output[index * batch_size: (index + 1) * batch_size],
                self.alpha: alpha_matrix[index * batch_size: (index + 1) * batch_size]
            }
        )
        
        for iteration in range(iterations):
            train_cost = 0
            for batch_index in range(n_train_batches):
                train_cost = train_da(batch_index)
                    
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
        
    num_user = 943
    num_item = 1682
    
    num_hidden = 200
    iterations = 400
    pre_iterations = 400
    alpha = 0.4
    imputation_ratio = 0.02
    
    train = build_ml_100k_train_binary3()
    test = build_ml_100k_test_binary3()
   
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()

    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]
    
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, train_matrix, num_hidden)
    
    output_matrix = mf_model.preTrain(pre_iterations,imputation_ratio)
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, output_matrix, num_hidden)
    
    mf_model.estimate(iterations, alpha)
    
    
    train = build_ml_100k_train_binary1()
    test = build_ml_100k_test_binary1()
    
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()
    
    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]
    
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, train_matrix, num_hidden)

    output_matrix = mf_model.preTrain(pre_iterations,imputation_ratio)
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, output_matrix, num_hidden)

    mf_model.estimate(iterations, alpha)



    train = build_ml_100k_train_binary2()
    test = build_ml_100k_test_binary2()
    
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)    
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()
    
    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]
    
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, train_matrix, num_hidden)

    output_matrix = mf_model.preTrain(pre_iterations,imputation_ratio)
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, output_matrix, num_hidden)

    mf_model.estimate(iterations, alpha)


    train = build_ml_100k_train_binary4()
    test = build_ml_100k_test_binary4()
    
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)    
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()
    
    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]   
    
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, train_matrix, num_hidden)

    output_matrix = mf_model.preTrain(pre_iterations,imputation_ratio)
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, output_matrix, num_hidden)

    mf_model.estimate(iterations, alpha)
    

    train = build_ml_100k_train_binary5()
    test = build_ml_100k_test_binary5()
    
    train_matrix = build_user_item_matrix(num_user+1, num_item+1, train)
    test_matrix = build_user_item_matrix(num_user+1, num_item+1, test)    
    train_matrix = train_matrix.todense()
    test_matrix = test_matrix.todense()
    
    train_matrix = train_matrix[1:,1:]
    test_matrix = test_matrix[1:,1:]
    
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, train_matrix, num_hidden)

    output_matrix = mf_model.preTrain(pre_iterations,imputation_ratio)
    
    mf_model = IDAE_ML_100k(
        train_matrix.shape[0], num_item, train_matrix, test_matrix, output_matrix, num_hidden)

    mf_model.estimate(iterations, alpha)
    
    return mf_model
    
    

if __name__ == "__main__":
    example()
    
    
    
    
