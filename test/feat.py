  
''' 
Implements the WEAT tests from sent-bias repo and bias-in-vision-and-language repo
https://github.com/W4ngatang/sent-bias/blob/master/sentbias/weat.py
https://github.com/candacelax/bias-in-vision-and-language/blob/703f559b1d81d51817d6fb7251b901efc28505b6/scripts/weat/weat_images_targ_specific.py#L234
'''

import logging as log
import math
import itertools as it
import numpy as np
import scipy.special
import scipy.stats

# 1. get cosine similarity
def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    Returns an array of size (len(XY), len(AB)) containing cosine similarities
    between items in XY and items in AB.
    """

    cossims = np.zeros((len(XY), len(AB)))
    for xy in XY:
        for ab in AB:
            cossims[xy, ab] = cossim(XY[xy], AB[ab])
    return cossims


# 2. Compute p-value

def mean_s_wAB(X, A, B, cossims):
    return np.mean(s_wAB(A, B, cossims[X]))

def stdev_s_wAB(X, A, B, cossims):
    return np.std(s_wAB(A, B, cossims[X]))

def p_val_permutation_test(X, Y, A, B, n_samples, cossims, parametric=False):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''
    
    X,Y = list(X), list(Y)
    A,B = list(A), list(B)

    if len(X) < len(Y):
        Y = Y[:len(X)]
    elif len(X) > len(Y):
        X = X[:len(Y)]
    assert len(X) == len(Y), f'len X {len(X)}, len Y {len(Y)}'
    size = len(X)
    s_wAB_memo = s_wAB(A, B, cossims=cossims)
    XY = X + Y

    if parametric:
        print('Using parametric test')
        s = s_XYAB(X, Y, s_wAB_memo)

        print('Drawing {} samples'.format(n_samples))
        samples = []
        for _ in range(n_samples):
            np.random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            assert len(Xi) == len(Yi)
            si = s_XYAB(Xi, Yi, s_wAB_memo)
            samples.append(si) 

        # Compute sample standard deviation and compute p-value by
        # assuming normality of null distribution
        print('Inferring p-value based on normal distribution')
        (shapiro_test_stat, shapiro_p_val) = scipy.stats.shapiro(samples)
        print('Shapiro-Wilk normality test statistic: {:.2f}, p-value: {:.2f}'.format(
            shapiro_test_stat, shapiro_p_val))
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        print('Sample mean: {:.2f}, sample standard deviation: {:.2f}'.format(
            sample_mean, sample_std))
        p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
        return p_val

    else:
        print('Using non-parametric test')
        s = s_XAB(X, s_wAB_memo)
        total_true = 0
        total_equal = 0
        total = 0

        run_sampling = len(X) > 20 # large to compute num partitions, so sample
        if run_sampling:
            # We only have as much precision as the number of samples drawn;
            # bias the p-value (hallucinate a positive observation) to
            # reflect that.
            total_true += 1
            total += 1
            print('Drawing {} samples (and biasing by 1)'.format(n_samples - total))
            for _ in tqdm(range(n_samples - 1)):
                np.random.shuffle(XY)
                Xi = XY[:size]
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
#                 print(si, s)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        else:
            num_partitions = int(scipy.special.binom(2 * len(X), len(X)))
            print(f'Using exact test ({num_partitions} partitions)')
            for Xi in it.combinations(XY, len(X)):
                #Xi = torch.tensor(Xi, dtype=torch.int)
                assert 2 * len(Xi) == len(XY)
                Xi = list(Xi)               
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        if total_equal:
            print('Equalities contributed {}/{} to p-value'.format(total_equal, total))
        if not isinstance(total_true / total, float):
            raise Exception(f'nan {total_true}, {total}')
        return total_true / total

def s_wAB(A, B, cossims):
    """
    Return vector of s(w, A, B) across w, where
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """  
    return np.mean(cossims[:, A], axis=1) - np.mean(cossims[:, B], axis=1)


def s_XAB(X, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    return slightly more computationally efficient version of WEAT
    statistic for p-value computation.
    Caliskan defines the WEAT statistic s(X, Y, A, B) as
        sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
    where s(w, A, B) is defined as
        mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    The p-value is computed using a permutation test on (X, Y) over all
    partitions (X', Y') of X union Y with |X'| = |Y'|.
    However, for all partitions (X', Y') of X union Y,
        s(X', Y', A, B)
      = sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = C,
    a constant.  Thus
        sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = sum_{x in X'} s(x, A, B) + (C - sum_{x in X'} s(x, A, B))
      = C + 2 sum_{x in X'} s(x, A, B).
    By monotonicity,
        s(X', Y', A, B) > s(X, Y, A, B)
    if and only if
        [s(X', Y', A, B) - C] / 2 > [s(X, Y, A, B) - C] / 2,
    that is,
        sum_{x in X'} s(x, A, B) > sum_{x in X} s(x, A, B).
    Thus we only need use the first component of s(X, Y, A, B) as our
    test statistic.
    """ 
    return s_wAB_memo[X].sum()


def s_XYAB(X, Y, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    the WEAT test statistic for p-value computation.
    """
    return s_XAB(X, s_wAB_memo) - s_XAB(Y, s_wAB_memo)

def effect_size(X, Y, A, B, cossims):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X, Y = list(X), list(Y)
    A, B = list(A), list(B)
    assert X != Y
    numerator = mean_s_wAB(X, A, B, cossims) - mean_s_wAB(Y, A, B, cossims)
    denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
#     print(numerator, denominator)
    return numerator / denominator