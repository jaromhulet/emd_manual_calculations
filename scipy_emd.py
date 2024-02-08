from scipy.stats import wasserstein_distance

if __name__ == '__main__':

    a =  [1, 1, 2, 2, 2, 3, 3, 3]
    b = [6, 6, 7, 8, 8, 8, 9, 9]

    emd = wasserstein_distance(a, b)
    print(emd)