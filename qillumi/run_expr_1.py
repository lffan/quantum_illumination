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
    """
    run experiments given the provided parameters
    Parameters
    ----------
    state_names: a list of strings of state names
    """
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


def expr_one_basic():
    """
    run experiments on different noise levels
    """
    cols = ['Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom_Bound', 'Chernoff_Bound', 'optimal_s',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)
    names = ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS')

    df = run_all_states(names, 16, 0.5, 0.01, 0.01, (0.4, 0.4), df)
    df = run_all_states(names, 24, 2.0, 0.01, 0.01, (0.4, 0.4), df)
    df = run_all_states(names, 32, 4.0, 0.01, 0.01, (0.4, 0.4), df)

    filename = '../output/expr_1_basic_{}.csv'.\
        format(datetime.today().strftime('%m-%d'))
    file = open(filename, 'w')
    file.write("# Quantum Illumination Experiment\n")
    file.write("# Several basic results for all kind of states, "
               "with different noise level (Nth).\n")
    file.write('# Author: L. Fan\n')
    file.write('# Created on: {}\n#\n'.format(str(datetime.now())))
    file.close()

    df.fillna('')
    df.to_csv(filename, index=False, columns=cols, mode='a')


def check_n_max(Nth, low, high):
    """
    check if results converges on the selected N_max
    """
    cols = ['Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom_Bound', 'Chernoff_Bound', 'optimal_s',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)
    names = ('TMSS',)
    for n_max in range(low, high):
        df = run_all_states(names, n_max, Nth, 0.01, 0.01, (0.4, 0.4), df)
        # print(df.iloc[-1])
        print("Helstrom: {:.11f}".format(df.iloc[-1]['Helstrom_Bound']))
        print("Chernoff: {:.11f}".format(df.iloc[-1]['Chernoff_Bound']))


if __name__ == "__main__":
    start_time = time.time()
    # check_n_max(0.1, 8, 16)
    # check_n_max(1.0, 16, 25)
    # check_n_max(4, 28, 36)
    expr_one_basic()
    print("--- %s seconds ---" % (time.time() - start_time))
