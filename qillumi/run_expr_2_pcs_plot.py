import time
from datetime import datetime
import numpy as np
import pandas as pd

from qillumi.qi import QIExpr
from qillumi.utils import log


@log
def expr_two_pcs_vs_rs(n_max, nth, ns, rflct, l_divides, r_divides, qcb_approx=True):
    cols = ['nmax', 'Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom_Bound', 'Chernoff_Bound', 'optimal_s',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']

    points = np.linspace(0, 1, r_divides)
    rss = [(ra, rb) for ra in points for rb in points]
    lmds = np.linspace(0.01, 0.41, l_divides)

    # lmd = np.sqrt(ns / (1 + ns))

    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=rflct, nth=nth)

    @log
    def set_laser(lmd, rs_t):
        expr.set_input_laser('PCS', lmd, rs_t)

    filename = "../output/data/expr_2_pcs_scan_nmax_{:d}_nth_{}_div_{:d}x{:d}_{}.csv" \
        .format(n_max, nth, l_divides, r_divides, datetime.today().strftime('%m-%d'))

    with open(filename, "a") as file:
        file.write("# Quantum Illumination Experiment\n")
        file.write("# PCS state different ra and rb.\n")
        file.write("# n_max: {}, nth: {}, ns: {}, rflct: {}\n".format(n_max, nth, ns, rflct))
        file.write('# Author: L. Fan\n')
        file.write('# Created on: {}\n'.format(str(datetime.now())))
        file.write(','.join(cols) + '\n')

        for lmd in lmds:
            for rs in rss:
                set_laser(lmd, rs)
                try:
                    expr.run_expr(qcb_approx=qcb_approx)
                    file.write(expr.results_string() + '\n')
                except np.linalg.LinAlgError:
                    file.write('# Singular Matrix lambda: {}'.format(lmd) + '\n')
                    print("Singular Matrix")
                    continue


if __name__ == "__main__":
    start_time = time.time()
    expr_two_pcs_vs_rs(n_max=24, nth=1.0, ns=0.01, rflct=0.01, l_divides=6, r_divides=51, qcb_approx=True)
    # expr_two_pcs_vs_rs(n_max=24, nth=1, ns=0.01, rflct=0.01, grid_divides=51, qcb_approx=True)
    print("--- %s seconds ---" % (time.time() - start_time))
