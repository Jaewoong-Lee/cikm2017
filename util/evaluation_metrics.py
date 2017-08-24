import numpy as np
import math

def RMSE(estimation, truth):
    """Root Mean Square Error"""

    #num_sample = len(estimation)

    # sum square error 
    sse = np.sum(np.square(truth - estimation))
    return sse #np.sqrt(np.divide(sse, num_sample - 1.0))


def _exceptSize(list1, list2):
    res =0
    
    for t in list1 :
        if not(t in list2):
            res += 1
    
    return res


def AUC(rankedList, groundTruth, num_dropped_items):
    
    overlap = [val for val in rankedList if val in groundTruth]
    
    num_rele_items = len(overlap)
    num_eval_items = len(rankedList) + num_dropped_items
    num_eval_pairs = (num_eval_items - num_rele_items) * num_rele_items
    
    if(num_eval_pairs < 0) :
        print ('error')
        
    if(num_eval_pairs == 0) :
        return 0.5
        
    num_correct_pairs = 0
    hits = 0
    for item_id in rankedList:
        if not(item_id in groundTruth):
            num_correct_pairs += hits
        else:
            hits += 1
    
    num_miss_items = _exceptSize(groundTruth, rankedList)
    num_correct_pairs += hits *(num_dropped_items - num_miss_items)
    
    return float(num_correct_pairs) / float(num_eval_pairs)


def HitsAt(rankedList, groundTruth, n):
    hits = 0
    
    k = len(rankedList)
    
    for i in range(k):
        item = rankedList[i]
    
        if not(item in groundTruth):
            continue
    
        if(i < n):
            hits += 1
        else:
            break
        
    
    return hits


def PrecAt(rankedList, groundTruth, ns):
    prec_at_n = np.zeros(len(ns))
    
    for n in range(len(ns)) :
        prec_at_n[n] = Prec(rankedList, groundTruth, ns[n])
        
    return prec_at_n


def Prec(rankedList, groundTruth, n):
    return float( HitsAt(rankedList, groundTruth, n) ) / float(n)


def RecallAt(rankedList, groundTruth, ns):
    recall_at_n = np.zeros(len(ns))
    
    for n in range(len(ns)) :
        recall_at_n[n] = Recall(rankedList, groundTruth, ns[n])
        
    return recall_at_n


def Recall(rankedList, groundTruth, n):
    return float( HitsAt(rankedList, groundTruth, n) ) / float(len(groundTruth))


def nDCGAt(rankedList, groundTruth, ns):
    nDCG_at_n = np.zeros(len(ns))
    
    for n in range(len(ns)) :
        nDCG_at_n[n] = nDCG(rankedList, groundTruth, ns[n])

    return nDCG_at_n


def nDCG(rankedList, groundTruth, n):
    if(len(rankedList) < n):
        n = len(rankedList)
        
    dcg = 0
    idcg = IDCG(n)
    
    for i in range(n):
        item_id = rankedList[i]
        
        if not(item_id in groundTruth):
            continue    
        
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)

    return float(dcg) / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg



def MapAt(rankedList, groundTruth, ns):
    prec_at_n = np.zeros(len(ns))
    
    for n in range(len(ns)) :
        prec_at_n[n] = MapHits(rankedList, groundTruth, ns[n])
        
    return prec_at_n


def MapHits(rankedList, groundTruth, n):
    k = len(rankedList)
    if (k < n):
        n = k
        
    hits = 0
    mapN = 0
    
    k = len(rankedList)
    
    for i in range(k):
        item = rankedList[i]
    
        if not(item in groundTruth):
            continue
    
        if(i < n):
            hits += 1
            mapN += float(hits) / float(i+1)
        else:
            break
        
    
    return mapN/float(n)