import time
from datetime import datetime
import numpy as np
import pandas as pd

from qillumi.qi import QIExpr


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
        new_df = pd.DataFrame.from_dict({'res': expr.get_results()}, orient='index')
        df = df.append(new_df)

    return df


def expr_two_pcs_vs_rs(n_max, nth, ns, rflct, grid_divides):
    points = np.linspace(0, 1, grid_divides)
    rss = [(ra, rb) for ra in points for rb in points]

    cols = ['Nth', 'R', 'State', 'lambda', 'ra', 'rb', 'Aver_N',
            'A_N', 'B_N', 'VN_Entropy', 'Helstrom', 'Chernoff', 'S_opt']
    df = pd.DataFrame(columns=cols)

    lmd = np.sqrt(ns / (1 + ns))
    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=rflct, nth=nth)

    # print("\nHelstrom Bounds\n")
    @log
    def set_laser(rs_t):
        expr.set_input_laser('PCS', lmd, rs_t)

    for rs in rss:
        set_laser(rs)
        expr.run_expr()
        new_df = pd.DataFrame.from_dict({'res': expr.get_results()}, orient='index')
        df = df.append(new_df)

    filename = "../output/expr_2_pcs_n_max_{}_nth_{}_grid_{}_{}.csv"\
        .format(n_max, nth, grid_divides, datetime.today().strftime('%m-%d'))
    file = open(filename, 'w')
    file.write("# Quantum Illumination Experiment\n")
    file.write("# PCS state with same lambda but different ra and rb.\n")
    file.write("# n_max: {}, nth: {}, ns: {}, rflct: {}\n".format(n_max, nth, ns, rflct))
    file.write('# Author: L. Fan\n')
    file.write('# Created on: {}\n\n'.format(str(datetime.now())))
    file.close()
    df.to_csv(filename, index=False, columns=cols, mode='a')


if __name__ == "__main__":
    start_time = time.time()
    expr_two_pcs_vs_rs(n_max=20, nth=1, ns=0.01, rflct=0.01, grid_divides=26)
    print("--- %s seconds ---" % (time.time() - start_time))
