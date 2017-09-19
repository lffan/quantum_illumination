import time
from datetime import datetime
import numpy as np
import pandas as pd

from qillumi.qi_utils import QIExpr


count = 0


def log(func):
    def wrapper(*args, **kw):
        global count
        count += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        func_name = func.__name__
        paras = args[1:-1] if func_name == 'run_all_states' else args
        print('{}: {:d} {}{}\t'.format(now, count, func_name, paras))
        return func(*args, **kw)
    return wrapper


@log
def run_all_states(state_names, n_max, nth, ns, rflct, rs, df):

    lmd = np.sqrt(ns / (1 + ns))
    expr = QIExpr(n_max)
    expr.set_environment(rflct, nth)

    # print("\nHelstrom Bounds\n")
    for name in state_names:
        if name != 'PCS':
            expr.set_input_laser(name, lmd)
        else:
            expr.set_input_laser('PCS', lmd, rs)
        expr.run_expr()
        new_df = pd.DataFrame.from_dict({'res': expr.get_attrs()}, orient='index')
        df = df.append(new_df)

    return df


def expr_one_basic():
    cols = ['Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom', 'Chernoff', 'S_opt',
            'A_N', 'B_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)
    names = ('TMSS', 'PS', 'PA', 'PAS', 'PSA', 'PCS')

    df = run_all_states(names, 10, 0.1, 0.01, 0.01, (0.4, 0.4), df)
    df = run_all_states(names, 12, 0.33, 0.01, 0.01, (0.4, 0.4), df)
    df = run_all_states(names, 15, 1.0, 0.01, 0.01, (0.4, 0.4), df)
    df = run_all_states(names, 17, 3.3, 0.01, 0.01, (0.4, 0.4), df)
    df = run_all_states(names, 20, 10.0, 0.01, 0.01, (0.4, 0.4), df)

    filename = '../output/expr_1_basic.csv'
    file = open(filename, 'w')
    file.write("# Quantum Illumination Experiment\n")
    file.write("# Several basic results for all kind of states, "
               "with different noise level (Nth).\n")
    file.write('# Author: L. Fan\n')
    file.write('# Created on: {}\n\n'.format(str(datetime.now())))
    file.close()

    df.fillna('')
    df.to_csv(filename, index=False, columns=cols, mode='a')


def expr_two_pcs_same_ns(grid_divides):
    points = np.linspace(0, 1, grid_divides)
    rss = [(ra, rb) for ra in points for rb in points]

    cols = ['Nth', 'R', 'State', 'lambda', 'ra', 'rb', 'Aver_N',
            'A_N', 'B_N', 'VN_Entropy', 'Helstrom', 'Chernoff', 'S_opt']
    df = pd.DataFrame(columns=cols)

    lmd = np.sqrt(0.01 / (1 + 0.01))
    expr = QIExpr(n_max=10)
    expr.set_environment(reflectance=0.01, nth=10)

    # print("\nHelstrom Bounds\n")
    @log
    def set_laser(rs_t):
        expr.set_input_laser('PCS', lmd, rs_t)

    for rs in rss:
        set_laser(rs)
        expr.run_expr()
        new_df = pd.DataFrame.from_dict({'res': expr.get_attrs()}, orient='index')
        df = df.append(new_df)

    filename = "../output/expr_2_pcs_grid{}.csv".format(grid_divides)
    file = open(filename, 'w')
    file.write("# Quantum Illumination Experiment\n")
    file.write("# PCS state with same lambda but different ra and rb.\n")
    file.write('# Author: L. Fan\n')
    file.write('# Created on: {}\n\n'.format(str(datetime.now())))
    file.close()
    df.to_csv(filename, index=False, columns=cols, mode='a')


def expr_three_qhb_vs_e(nss_divides):
    nss = np.linspace(0.01, 3, nss_divides)
    lambdas = np.sqrt(nss / (1.0 + nss))

    cols = ['Nth', 'R', 'State', 'lambda', 'ra', 'rb', 'Aver_N',
            'A_N', 'B_N', 'VN_Entropy', 'Helstrom', 'Chernoff', 'S_opt']
    df = pd.DataFrame(columns=cols)

    expr = QIExpr(n_max=10)
    expr.set_environment(reflectance=0.01, nth=0.1)

    @log
    def set_laser(s_name, l):
        if s_name != 'PCS':
            expr.set_input_laser(s_name, l)
        else:
            expr.set_input_laser('PCS', l, rs=(0.1, 0.9))

    for lmd in lambdas:
        for name in ('TMSS', 'PS', 'PA', 'PAS', 'PSA', 'PCS'):
            set_laser(name, lmd)
            expr.run_expr()
            new_df = pd.DataFrame.from_dict({'res': expr.get_attrs()},
                                            orient='index')
            df = df.append(new_df)

    filename = "../output/expr_3_lmd_p{}.csv".format(nss_divides)
    file = open(filename, 'w')
    file.write("# Quantum Illumination Experiment\n")
    file.write("# Experiment for all states vs lambda.\n")
    file.write('# Author: L. Fan\n')
    file.write('# Created on: {}\n\n'.format(str(datetime.now())))
    file.close()
    df.to_csv(filename, index=False, columns=cols, mode='a')


def expr_four_pcs_same_ns():
    pass


if __name__ == "__main__":
    start_time = time.time()
    # expr_one_basic()
    # expr_two_pcs_same_ns(101)
    expr_three_qhb_vs_e(101)
    print("--- %s seconds ---" % (time.time() - start_time))
