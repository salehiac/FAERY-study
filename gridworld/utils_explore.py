import bisect


class Capacity:
    
    def __init__(self, w) -> None:
        self.phi = {i/len(w): sum(w[-i:]) for i in range(1,len(w))}
        self.phi[0] = 0
        self.phi[1] = 1

    def find_neighbours(self, A, x):
        """
        Returns the neighbours of x in A
        """

        cA = sorted(list(A))

        i = bisect.bisect_left(cA,x)
        return cA[i-1], cA[i]
    
    def interpolate(self, newp):
        """
        Adds a new argument to phi with linear interpolation
        """

        n1, n2 = self.find_neighbours(self.phi.keys(), newp)
        adjust = (newp - n1) / (n2 - n1)
        self.phi[newp] = adjust * (self.phi[n2] - self.phi[n1]) + self.phi[n1]

    def __call__(self, p):
        """
        Calls phi on given vector, compute it if not already
        """
        
        if p not in self.phi:
            self.interpolate(p)
        
        return self.phi[p]

    def __str__(self):
        return str(self.phi)


def OWA(w, a):
    """
    Applies OWA to vector a with given weights w
    """
    
    assert len(w) == len(a), "Input vector doesn't have the same dimension as the weights ({} and {})".format(len(a), len(w))

    return [w[i]*x for i, x in enumerate(sorted(a))]


def WOWA(w, p, a, phi=None):
    """
    Applies WOWA to vector a with given weights w and p
    """

    assert len(w) == len(a), "Input vector doesn't have the same dimension as the weights ({} and {})".format(len(a), len(w))

    if phi is None:
        phi = Capacity(w)
    
    # OREDERING GIVEN X!!
    si, sa = zip(*sorted(enumerate(a), key=lambda x: x[1]))

    s = sa[0]
    for i in range(1, len(w)):
        s += (sa[i] - sa[i-1]) * phi(sum([p[k] for k in si[i:]]))
    
    return s


if __name__ == "__main__":

    # w = [3/6, 2/6, 1/6]
    # p = [3/6, 1/6, 2/6]
    # x = (10, 5, 15)
    # y = (10, 12, 8)

    # phi = Capacity(w)
    
    # print(WOWA(w, p, x, phi), WOWA(w, p, y, phi))
    # w = [.7, .25, .05]
    # phi = Capacity(w)
    # print(WOWA(w, p, x, phi), WOWA(w, p, y, phi))


    vectors = [
        (6, 6, 6), # Uniform
        (9, 6, 3), # Early
        (3, 6, 9) # Late
    ]

    p_to_test = [
        ((1/3, 1/3, 1/3), "uniform"),
        ((4/6, 3/12, 1/12), "early"),
        ((1/12, 3/12, 4/6), "late"),
    ]

    w_to_test = [
        ((.6, .3, .1), "uniform"),
        ((.7, .25, .05), "slow"),
        ((.05, .25, .7), "fast"),
    ]

    print()
    print("p (localization)\t\t\tw (imbalance)")
    for p, w in zip(p_to_test, w_to_test):
        print("\t{}: {}\t\t{}: {}".format(p[1], [round(i, 2) for i in p[0]], w[1], [round(i, 2) for i in w[0]]))

    print()
    print("a (vectors to compare)")
    s = "\t\t\t  "
    for k in range(len(vectors)):
        print("\ta{}: {}".format(k, vectors[k]))
        s+= "a{}    ".format(k)
    print(s)

    for p, comp in p_to_test:
        print("p={}".format(comp))

        for w, comw in w_to_test:
            cap = Capacity(w)
            print("\tw={} : \t {}".format(comw, [round(WOWA(w, p, v, cap),1) for v in vectors]))

    print()