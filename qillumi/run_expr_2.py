import time
from datetime import datetime
import numpy as np
import pandas as pd

from qillumi.qi import QIExpr
from qillumi.utils import log


@log
def expr_two_pcs_vs_rs(n_max, nth, ns, rflct, grid_divides):
    points = np.linspace(0, 1, grid_divides)
    rss = [(ra, rb) for ra in points for rb in points]
    cols = ['Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom_Bound', 'Chernoff_Bound', 'optimal_s',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']
    # cols = ['Nth', 'R', 'State', 'lambda', 'ra', 'rb', 'Aver_N',
    #         'A_N', 'B_N', 'VN_Entropy', 'Helstrom', 'Chernoff', 'S_opt']
    df = pd.DataFrame(columns=cols)

    lmd = np.sqrt(ns / (1 + ns))
    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=rflct, nth=nth)

    @log
    def set_laser(rs_t):
        expr.set_input_laser('PCS', lmd, rs_t)

    for rs in rss:
        set_laser(rs)
        expr.run_expr()
        new_df = pd.DataFrame.from_dict({'res': expr.get_results()}, orient='index')
        df = df.append(new_df)

    write_data_to_file(df, n_max, nth, ns, rflct, grid_divides, cols)


def write_data_to_file(df, n_max, nth, ns, rflct, grid_divides, cols):
    filename = "../output/expr_2_pcs_nth_{}_g_{}_{}.csv"\
        .format(nth, grid_divides, datetime.today().strftime('%m-%d'))
    with open(filename, 'w') as file:
        file.write("# Quantum Illumination Experiment\n")
        file.write("# PCS state with same lambda but different ra and rb.\n")
        file.write("# n_max: {}, nth: {}, ns: {}, rflct: {}\n".format(n_max, nth, ns, rflct))
        file.write('# Author: L. Fan\n')
        file.write('# Created on: {}\n\n'.format(str(datetime.now())))
    df.to_csv(filename, index=False, columns=cols, mode='a')


if __name__ == "__main__":
    start_time = time.time()
    # expr_two_pcs_vs_rs(n_max=8, nth=0.1, ns=0.01, rflct=0.01, grid_divides=101)
    expr_two_pcs_vs_rs(n_max=24, nth=1, ns=0.01, rflct=0.01, grid_divides=3)
    print("--- %s seconds ---" % (time.time() - start_time))
