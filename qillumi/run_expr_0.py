import time
from datetime import datetime
import numpy as np
import pandas as pd

import qillumi.laser2mode as laser
from qillumi.qi import QIExpr
from qillumi.utils import log

from scipy.optimize import minimize


# def basic_attributes(n_max, divides):
#     cols = ['State', 'lambda', 'Exact_N', 'Aver_N', 'VN_Entropy',
#             'A_aver_N', 'B_aver_N', 'ra', 'rb']
#     df = pd.DataFrame(columns=cols)
#
#     lasers = {'TMSS': laser.TMSS, 'PS': laser.PS, 'PA': laser.PA,
#               'PSA': laser.PSA, 'PAS': laser.PAS, 'PCS': laser.PCS}
#
#     lambdas = {'TMSS': np.linspace(0.001, 0.901, divides),
#                'PS': np.linspace(0.001, 0.751, divides),
#                'PA': np.linspace(0.001, 0.701, divides),
#                'PSA': np.linspace(0.001, 0.601, divides),
#                'PAS': np.linspace(0.001, 0.551, divides),
#                'PCS': np.linspace(0.001, 0.701, divides)}
#     names = ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS')
#
#     @log
#     def create_laser(s_name, l):
#         if s_name != 'PCS':
#             s = lasers[s_name](l, n_max)
#         else:
#             s = laser.PCS(l, n_max, rs=(0.4, 0.4))
#             # s = laser.PCS(l, n_max, rs=(0.2, 0.9))
#         return s
#
#     for name in names:
#         for lmd in lambdas[name]:
#             state = create_laser(name, lmd)
#             new_df = pd.DataFrame.from_dict({'res': state.get_attrs()}, orient='index')
#             df = df.append(new_df)
#
#     df.to_csv('../output/data/basic_attributes.csv', index=False, columns=cols)


def etgl_sym(r, l, n_max):
    # print(r)
    state = laser.PCS(l, n_max, (r[0], r[0]))
    return - state.entanglement


def etgl_asym(r, l, n_max):
    state = laser.PCS(l, n_max, (r[0], r[1]))
    return - state.entanglement


def basic_attributes_2(n_max, divides):
    cols = ['nmax', 'State', 'sqz', 'lambda', 'Exact_N', 'Aver_N',
            'VN_Entropy', 'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    lasers = {'TMSS': laser.TMSS, 'PS': laser.PS, 'PA': laser.PA,
              'PSA': laser.PSA, 'PAS': laser.PAS, 'PCS': laser.PCS}

    # ss = np.linspace(0.001, 1.001, divides)
    # lmd = [np.tanh(e) for e in ss]
    # lambdas = {'TMSS': lmd,
    #            'PS': lmd,
    #            'PA': lmd,
    #            'PSA': lmd,
    #            'PAS': lmd,
    #            'PCS': lmd}

    lambdas = {'TMSS': np.linspace(0.001, 0.901, divides),
               'PS': np.linspace(0.001, 0.751, divides),
               'PA': np.linspace(0.001, 0.701, divides),
               'PSA': np.linspace(0.001, 0.601, divides),
               'PAS': np.linspace(0.001, 0.551, divides),
               'PCS': np.linspace(0.001, 0.701, divides)}

    names = ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS')

    @log
    def create_laser(s_name, l):
        if s_name != 'PCS':
            s = lasers[s_name](l, n_max)
        else:
            res1 = minimize(etgl_asym, np.array([0.2, 0.9]), args=(l, n_max),
                            method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
            res2 = minimize(etgl_asym, np.array([0.3, 0.3]), args=(l, n_max),
                            method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
            res = res1 if res1.fun < res2.fun else res2
            ra, rb = res.x[0], res.x[1]
            if (1 - 1e-6 < ra < 1 + 1e-6) and (1 - 1e-6 < rb < 1 + 1e-6):
                ra, rb = 0, 0
            s = laser.PCS(l, n_max, rs=(ra, rb))
        return s

    for name in names:
        for lmd in lambdas[name]:
            state = create_laser(name, lmd)
            new_df = pd.DataFrame.from_dict({'res': state.get_attrs()}, orient='index')
            df = df.append(new_df)

    df.to_csv("../output/data/expr_0_basic_nmax_{:d}_div_{:d}_{}.csv"
              .format(n_max, divides, datetime.today().strftime('%m-%d')),
              index=False, columns=cols)


if __name__ == "__main__":
    start_time = time.time()
    basic_attributes_2(n_max=64, divides=201)
    print("--- %s seconds ---" % (time.time() - start_time))
