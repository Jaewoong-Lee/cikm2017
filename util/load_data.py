"""
load data set

"""
import numpy as np
import scipy.sparse as sparse
import os

def build_user_item_matrix(n_users, n_items, ratings):
    """Build user-item matrix
    Return
    ------
        sparse matrix with shape (n_users, n_items)
    """
    
    data = ratings[:, 2]
    row_ind = ratings[:, 0]
    col_ind = ratings[:, 1]
    shape = (n_users, n_items)
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)

def build_ml_1m_train_binary1():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u1.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings


def build_ml_1m_test_binary1():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u1.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings

def build_ml_1m_train_binary2():
    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u2.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings


def build_ml_1m_test_binary2():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u2.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings

def build_ml_1m_train_binary3():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u3.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings


def build_ml_1m_test_binary3():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u3.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings

def build_ml_1m_train_binary4():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u4.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings


def build_ml_1m_test_binary4():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u4.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings

def build_ml_1m_train_binary5():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u5.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings


def build_ml_1m_test_binary5():

    print("\nloadind movie lens 1m data")
    with open("./data/ml-1m/u5.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('::')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings




def build_ml_100k_train_binary5():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u5.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings

def build_ml_100k_test_binary5():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u5.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings






def build_ml_100k_train_binary4():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u4.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings

def build_ml_100k_test_binary4():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u4.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings


def build_ml_100k_train_binary3():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """   

    print("\nloadind movie lens 1m data")
    with open("./data/ml-100k/u3.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            
            line[2] = 1 #
            #if(line[2] > 3 ):
            #    line[2] = 1 # to binary # temporary
            #else :
            #    line[2] = 0 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings



def build_ml_100k_test_binary3():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u3.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 #
            #if(line[2] > 3 ):
            #    line[2] = 1 # to binary # temporary
            #else :
            #    line[2] = 0 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings




def build_ml_100k_train_binary2():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u2.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings

def build_ml_100k_test_binary2():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u2.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings




def build_ml_100k_train_binary1():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u1.base", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings

def build_ml_100k_test_binary1():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """

    print("\nloadind movie lens 100k data")
    with open("./data/ml-100k/u1.test", "r") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line[:-1]
            line = line[:-1]
            line = line.split('\t')[:3]
            #line = line.split(' ')[:4] #to create data
            
            line = [int(l) for l in line]
            line[2] = 1 # to binary # temporary
            ratings.append(line)

            if line_num % 100000 == 0:
                print (line_num)

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)]
    print ("max user id", max(ratings[:, 0]))
    print ("max item id", max(ratings[:, 1]))
    
    num_user = max(ratings[:, 0])+1
    num_item = max(ratings[:, 1])+1
    
    return ratings




