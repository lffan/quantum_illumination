import time
from datetime import datetime
import numpy as np
import pandas as pd

import qillumi.laser2mode as laser
from qillumi.qi import QIExpr
from qillumi.utils import log

from scipy.optimize import minimize

__author__ = 'L. Fan'


def etgl_sym(r, l, n_max):
    # print(r)
    state = laser.PCS(l, n_max, (r[0], r[0]))
    return - state.entanglement


def etgl_asym(r, l, n_max):
    state = laser.PCS(l, n_max, (r[0], r[1]))
    return - state.entanglement


def basic_attributes(n_max, divides):
    cols = ['nmax', 'State', 'sqz', 'lambda', 'Exact_N', 'Aver_N',
            'VN_Entropy', 'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    lasers = {'TMSS': laser.TMSS, 'PS': laser.PS, 'PA': laser.PA,
              'PSA': laser.PSA, 'PAS': laser.PAS, 'PCS': laser.PCS}

    lambdas = {'TMSS': np.linspace(0.001, 0.801, divides),
               'PS': np.linspace(0.001, 0.651, divides),
               'PA': np.linspace(0.001, 0.601, divides),
               'PSA': np.linspace(0.001, 0.501, divides),
               'PAS': np.linspace(0.001, 0.451, divides),
               'PCS': np.linspace(0.3545, 0.701, divides)}

    # names = ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS')
    # names = ('TMSS', 'PS', 'PA')
    names = ('PCS',)

    @log
    def create_laser(s_name, l):
        if s_name != 'PCS':
            s = lasers[s_name](l, n_max)
        else:
            res = minimize(etgl_asym, np.array([0.2, 0.85]), args=(l, n_max),
                           method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
            res2 = minimize(etgl_asym, np.array([0.3, 0.3]), args=(l, n_max),
                            method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
            if res2.fun < res.fun:
                res = res2
            ra, rb = res.x[0], res.x[1]
            if (-1e-6 < ra < 1e-6) and (-1e-6 < rb < 1e-6):
                ra, rb = 1, 1
            s = laser.PCS(l, n_max, rs=(ra, rb))
        return s

    filename = "../output/data/expr_0_basic_pcs_asym_nmax_{:d}_div_{:d}_{}.csv"\
        .format(n_max, divides, datetime.today().strftime('%m-%d'))

    with open(filename, "a") as file:
        file.write("# Quantum Illumination Experiment 0\n")
        file.write('# Entanglement and Average Photon Numbers\n')
        file.write('# Author: L. Fan\n')
        file.write('# Created on: {}\n'.format(str(datetime.now())))
        file.write(','.join(cols) + '\n')

        for name in names:
            for lmd in lambdas[name]:
                state = create_laser(name, lmd)
                attributes = state.get_attrs()
                out_string = ','.join([str(attributes.get(col, 0)) for col in cols]) + '\n'
                file.write(out_string)


def find_optimal_etgl(ns, n_max, l=None):
    if not l:
        l = np.sqrt(ns / (1 + ns))
    res1 = minimize(etgl_asym, np.array([0.2, 0.8]), args=(l, n_max),
                    method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
    res2 = minimize(etgl_asym, np.array([0.3, 0.6]), args=(l, n_max),
                    method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
    res3 = minimize(etgl_asym, np.array([0.9, 0.9]), args=(l, n_max),
                    method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
    print(res1.fun, res1.x)
    print(res2.fun, res2.x)
    print(res3.fun, res3.x)


if __name__ == "__main__":
    start_time = time.time()
    find_optimal_etgl(ns=0.01, n_max=48, l=0.45)
    # basic_attributes(n_max=64, divides=100)
    print("--- %s seconds ---" % (time.time() - start_time))

    # tmss = laser.TMSS(0.0995037, n_max=24)
    # tmss.num()
