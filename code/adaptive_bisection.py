from __future__ import division, print_function
import numpy as np
from progress import printProgress
import matplotlib.pyplot as plt
from matplotlib import rc
# Computer modern fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

eval_tol_max = 0.5
eval_tol_min = 0.02

LOG = True

def tween(i, a, b):
    assert (0 <= i <= 1), "i needs to be normalized"

    return a + ((1-i) * i**(2) + (i) * i**(1/6)) * (b - a)

# i_space = np.linspace(0,1,1000)
# it = np.vectorize(tween)
# plt.figure()
# plt.title(r"Adaptive Precision $a=0.01, b=0.5$")
# plt.xlabel(r"Bisection Step")
# plt.ylabel(r"RKF Precision")
# plt.plot(i_space, it(i_space, 0.5, 0.01))
# plt.savefig("../figures/bisection_tween.pdf", format="pdf")
# plt.show()

def adaptive_bisection(f, a, b, precision=0.001):
    n_max = np.ceil(np.log2(abs(b-a) / precision))

    o_a = a
    o_b = b

    n = 1
    if LOG: printProgress(0, n_max, "Bisection")
    f_a = f(a, eval_tol_max)
    f_b = f(b, eval_tol_max)
    if f_a * f_b > 0:
        print(f_a, f_b)
        raise Exception("No root in range")
    best_c = None
    best_f = None
    best_tol = None

    # cs = [a, b]
    # fs = [f_a, f_b]
    while n <= n_max:
        c = (a + b)/2
        tol = tween(n/n_max, eval_tol_max, eval_tol_min)
        # print(tol)
        f_c = f(c, tol)
        # print(c,f_c)

        # cs.append(c)
        # fs.append(f_c)

        if best_c is None or (abs(f_c) < abs(best_f)):
            best_c = c
            best_f = f_c
            best_tol = tol

        if (f_c == 0 or (b-a)/2 < precision):
            if LOG: printProgress(n_max, n_max, "Complete")
            # print("Error",  best_f, best_c)
            # plt.figure()
            # plt.title(r"Error in Luminosity Boundary Condition")
            # plt.xlabel(r"$\rho_c$")
            # plt.ylabel(r"$f(\rho_c)$")
            # plt.plot([o_a,o_b], [0,0])
            # plt.plot(cs, fs, 'ro')
            # plt.gca().set_yscale("log")
            # plt.gca().set_xscale("log")

            # # plt.gca().axhline(y=0,c="blue",linewidth=0.5,zorder=0)
            # plt.savefig("../figures/error.pdf", format="pdf")
            # plt.show()
            return (best_c, best_tol)
        if LOG: printProgress(n, n_max, "Bisection")
        n = n+1
        if (f_c * f_a > 0):
            a, f_a = c, f_c
        else:
            b, f_b = c, f_c
    raise Exception("n_max was not large enough.")
