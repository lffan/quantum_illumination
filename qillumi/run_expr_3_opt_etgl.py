import time
from datetime import datetime
import numpy as np
import pandas as pd

import qillumi.laser2mode as laser
from qillumi.qi import QIExpr
from qillumi.utils import log

from scipy.optimize import minimize


def etgl_sym(r, l, n_max):
    state = laser.PCS(l, n_max, (r[0], r[0]))
    return - state.entanglement


def etgl_asym(r, l, n_max):
    state = laser.PCS(l, n_max, (r[0], r[1]))
    return - state.entanglement


def expr_three_qhb_vs_energy(nth, n_max, divides, qcb_approx=True):
    # nss = np.linspace(0.01, 1, nss_divides)
    # lambdas = np.sqrt(nss / (1.0 + nss))
    # divides: divides lambda
    cols = ['nmax', 'Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom_Bound', 'Chernoff_Bound', 'optimal_s',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    lambdas = {'TMSS': np.linspace(0.001, 0.901, divides),
               'PS': np.linspace(0.001, 0.751, divides),
               'PA': np.linspace(0.001, 0.701, divides),
               'PSA': np.linspace(0.001, 0.601, divides),
               'PAS': np.linspace(0.001, 0.551, divides),
               'PCS': np.linspace(0.001, 0.701, divides)}
    names = ('TMSS', 'PS', 'PA', 'PSA', 'PAS')
    # names = ('PCS',)

    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=0.01, nth=nth)

    @log
    def set_laser(s_name, l):
        if s_name != 'PCS':
            expr.set_input_laser(s_name, l)
        else:
            res1 = minimize(etgl_asym, np.array([0.2, 0.9]), args=(l, n_max),
                            method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
            res2 = minimize(etgl_asym, np.array([0.3, 0.3]), args=(l, n_max),
                            method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
            res = res1 if res1.fun < res2.fun else res2
            ra, rb = res.x[0], res.x[1]
            if (1 - 1e-6 < ra < 1 + 1e-6) and (1 - 1e-6 < rb < 1 + 1e-6):
                ra, rb = 0, 0
            expr.set_input_laser('PCS', l, rs=(ra, rb))

    for name in names:
        for lmd in lambdas[name]:
            set_laser(name, lmd)
            try:
                expr.run_expr(qcb_approx=qcb_approx)
            except Exception as e:
                print("%s" % str(e))
                write_data_to_file(df, n_max, nth, divides, cols)
            new_df = pd.DataFrame.from_dict({'res': expr.get_results()}, orient='index')
            df = df.append(new_df)

    write_data_to_file(df, n_max, nth, divides, cols)


def write_data_to_file(df, n_max, nth, nss_divides, cols):
    filename = "../output/data/expr_3_qi_nmax_{}_nth_{}_g_{}_{}_opt_etgl.csv" \
        .format(n_max, nth, nss_divides, datetime.today().strftime('%m-%d'))
    with open(filename, 'w') as file:
        file.write("# Quantum Illumination Experiment\n")
        file.write("# Experiment for all states vs lambda.\n")
        file.write('# Author: L. Fan\n')
        file.write('# Created on: {}\n\n'.format(str(datetime.now())))
    df.to_csv(filename, index=False, columns=cols, mode='a')


def special_pcs(n_max, nth):
    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=0.01, nth=nth)
    expr.set_input_laser('PCS', lmd=0.2985, rs=(0.4472107216, 1.0))
    expr.run_expr(qcb_approx=True)
    print(expr.get_results())


if __name__ == "__main__":
    start_time = time.time()
    # expr_three_qhb_vs_energy(divides=51, n_max=24, nth=0.1, qcb_approx=True)
    expr_three_qhb_vs_energy(divides=11, n_max=32, nth=1.0, qcb_approx=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    # special_pcs(n_max=32, nth=1.0)
