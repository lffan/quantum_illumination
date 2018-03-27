import time
from datetime import datetime
import numpy as np
import pandas as pd

from qillumi.qi import QIExpr
from qillumi.utils import log


@log
def run_all_states(state_names, n_max, nth, ns, rflct, rs, df):
    """
    run experiments given the provided parameters

    Parameters
    ----------
    state_names: tuple of string, indicating input states
    n_max: integer, truncated photon numbers
    nth: double, average thermal noise number
    ns: double, N_s parameter of the two mode squeezed state
    rflct: double, reflectance of the target object
    rs: tuple of double, parameters for PCS states
    df: dataframe, running results

    Returns
    -------
    df: a pandas dataframe recording running results
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


def expr_one_basic(df, names, configs):
    """
    run experiments on different noise levels
    """
    for N_max, Nth in configs:
        df = run_all_states(names, N_max, Nth, 0.01, 0.01, (0.4, 0.4), df)
    return df


def run_and_save_data(names, configs):
    """
    run experiments given the state names and configuration

    Parameters
    ----------
    names: tuple of strings, states to be run
    configs: tuple of double, (N_max, Nth)

    Returns
    -------

    """
    cols = ['Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom_Bound', 'Chernoff_Bound', 'optimal_s',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    df = expr_one_basic(df, names, configs)

    filename = '../output/data/expr_1_basic_{}.csv'.\
        format(datetime.today().strftime('%m-%d'))
    with open(filename, 'w') as file:
        file.write("# Quantum Illumination Experiment\n")
        file.write("# Several basic results for all kind of states, "
                   "with different noise level (Nth).\n")
        file.write('# Author: L. Fan\n')
        file.write('# Created on: {}\n#\n'.format(str(datetime.now())))

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
    names = ('PCS',)
    for n_max in range(low, high):
        df = run_all_states(names, n_max, Nth, 0.01, 0.01, (0.4, 0.4), df)
        # print(df.iloc[-1])
        print("Helstrom: {:.11f}".format(df.iloc[-1]['Helstrom_Bound']))
        print("Chernoff: {:.11f}".format(df.iloc[-1]['Chernoff_Bound']))


if __name__ == "__main__":
    # check if results converge for a given N_max
    # check_n_max(0.1, 8, 16)
    # check_n_max(1.0, 16, 25)
    # check_n_max(4, 28, 36)

    states = ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS')
    # expr_configs = [(8, 0.1), (16, 0.5)]
    expr_configs = [(8, 0.1), (16, 0.5), (24, 1.0), (24, 2.0)]
    start_time = time.time()
    run_and_save_data(states, expr_configs)
    print("--- %s seconds ---" % (time.time() - start_time))
