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
    state = laser.PCS(l, n_max, (r[0], r[0]))
    return - state.entanglement


def etgl_asym(r, l, n_max):
    state = laser.PCS(l, n_max, (r[0], r[1]))
    return - state.entanglement


def qcb_sym(r, lmd, expr):
    expr.set_input_laser('PCS', lmd=lmd, rs=(r[0], r[0]))
    expr.run_expr(qcb_approx=True)
    return expr.get_results()['Chernoff_Bound']


def qcb_asym(r, lmd, expr):
    expr.set_input_laser('PCS', lmd=lmd, rs=(r[0], r[1]))
    expr.run_expr(qcb_approx=True)
    return expr.get_results()['Chernoff_Bound']


def expr_three_qhb_vs_energy(method, nth, n_max, div1, div2, qcb_approx=True):
    # nss = np.linspace(0.01, 1, nss_divides)
    # lambdas = np.sqrt(nss / (1.0 + nss))
    # divides: divides lambda
    cols = ['nmax', 'Nth', 'R', 'State', 'sqz', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom_Bound', 'Chernoff_Bound', 'optimal_s',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    lmds = np.append(np.linspace(0.002, 0.452, div1), np.linspace(0.462, 0.602, div2))
    names = ('PCS',)

    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=0.01, nth=nth)

    @log
    def set_laser(s_name, l):
        if s_name != 'PCS':
            expr.set_input_laser(s_name, l)
        else:
            res = minimize(qcb_asym, np.array([0.2, 0.8]), args=(l, expr),
                           method=method, bounds=[(0, 1), (0, 1)])
            if l > 0.3:
                res2 = minimize(qcb_asym, np.array([0.8, 0.8]), args=(l, expr),
                                method=method, bounds=[(0, 1), (0, 1)])
                if res2.fun < res:
                    res = res2
            ra, rb = res.x[0], res.x[1]
            print("ra: {}, rb: {}".format(ra, rb))
            if (- 1e-6 < ra < 1e-6) and (- 1e-6 < rb < 1e-6):
                ra, rb = 1, 1
            expr.set_input_laser('PCS', l, rs=(ra, rb))

    filename = "../output/data/expr_3_qi_nmax_{}_nth_{}_g_{}+{}_{}_opt_qcb_{}.csv"\
        .format(n_max, nth, div1, div2, datetime.today().strftime('%m-%d'), method)

    with open(filename, "a") as file:
        file.write("# Quantum Illumination Experiment 3.1\n")
        file.write('# Author: L. Fan\n')
        file.write('# Created on: {}\n\n'.format(str(datetime.now())))
        file.write(','.join(cols) + '\n')

        for lmd in lmds:
            set_laser('PCS', lmd)
            try:
                expr.run_expr(qcb_approx=qcb_approx)
                # print(expr.results_string())
                file.write(expr.results_string() + '\n')
            except np.linalg.LinAlgError:
                file.write('# Singular Matrix lambda: {}'.format(lmd) + '\n')
                print("Singular Matrix")
                continue


def find_optimal_etgl(n_max, ns, nth):
    l = np.sqrt(ns / (1 + ns))
    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=0.01, nth=nth)

    # # res1 = minimize(qcb_asym, np.array([0.2, 0.8]), args=(l, expr),
    # #                 method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
    # res2 = minimize(qcb_asym, np.array([0.2, 0.2]), args=(l, expr),
    #                 method='L-BFGS-B', bounds=[(0, 1), (0, 1)])
    # res3 = minimize(qcb_sym, np.array([0.2]), args=(l, expr),
    #                 method='L-BFGS-B', bounds=[(0, 1)])
    # # print(res1.fun, res1.x)
    # print(res2.fun, res2.x)

    expr.set_input_laser('PCS', lmd=l, rs=(0.11755415, 0.94959852))
    expr.run_expr()
    print(expr.get_results())
    # print(res3.fun, res3.x)


def special_pcs(n_max, nth):
    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=0.01, nth=nth)
    expr.set_input_laser('PCS', lmd=0.099503719020998915, rs=(1.0, 0.14019005))
    expr.run_expr(qcb_approx=True)
    print(expr.get_results())


if __name__ == "__main__":
    start_time = time.time()
    # expr_three_qhb_vs_energy(method='L-BFGS-B', div1=46, div2=8, n_max=24, nth=1.0, qcb_approx=True)
    # expr_three_qhb_vs_energy(method='TNC', div1=91, div2=25, n_max=24, nth=1.0, qcb_approx=True)
    # find_optimal_etgl(n_max=32, ns=0.01, nth=1.0)
    print("--- %s seconds ---" % (time.time() - start_time))
    special_pcs(n_max=32, nth=1.0)
