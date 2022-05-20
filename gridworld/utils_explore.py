import bisect


def OWA(w, a):
    """
    Applies OWA to vector a with given weights w
    """
    
    assert len(w) == len(a), "Input vector doesn't have the same dimension as the weights ({} and {})".format(len(a), len(w))

    return [w[i]*x for i, x in enumerate(sorted(a))]


def find_neighbours(A, x):
    """
    Returns the neighbours of x in A
    """

    cA = list(A)

    i = bisect.bisect_left(cA,x)
    return cA[i-1], cA[i] 


def build_capacity(w, p, ndigits=18):
    """
    Builds the capacity for WOWA from the given weights w and p
    """

    phi = {i/len(w): round(sum(w[-i:]), ndigits=ndigits) for i in range(1,len(w)+1)}
    phi[0] = 0

    sp = sorted(p)
    sum_p = [sum(sp[i:]) for i in range(1,len(p))]

    for x in sum_p:
        if x not in phi:
            n1, n2 = find_neighbours(phi.keys(), x)
            adjust = (x - n1) / (n2 - n1)
            phi[x] = adjust * (phi[n2] + phi[n1])

    return phi


def WOWA(w, p, a, phi=None):
    """
    Applies WOWA to vector a with given weights w and p
    """

    if phi is None:
        phi = build_capacity(w, p)
    
    sx, sp = sorted(a), sorted(p)
    return sx[0] + sum([(sx[i] - sx[i-1]) * phi[sum(sp[i:])] for i in range(1, len(w))])


if __name__ == "__main__":

    # STRANGE.. NOT SAME RESULTS AS CHOQUET INTEGRAL
    #   MAYBE MISTAKE IN THE FORMULA ?

    w = [3/6, 2/6, 1/6]
    p = [3/6, 1/6, 2/6]
    x = (10, 5, 15)
    y = (10, 12, 8)

    phi = build_capacity(w, p)
    print(WOWA(w, p, x, phi), WOWA(w, p, y, phi))

    w = [.7, .25, .05]
    phi = build_capacity(w, p)
    print(WOWA(w, p, x, phi), WOWA(w, p, y, phi))