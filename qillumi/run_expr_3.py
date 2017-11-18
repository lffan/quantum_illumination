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


def expr_three_qhb_vs_energy(nss_divides):
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
            new_df = pd.DataFrame.from_dict({'res': expr.get_results()},
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
    expr_three_qhb_vs_energy(101)
    print("--- %s seconds ---" % (time.time() - start_time))
