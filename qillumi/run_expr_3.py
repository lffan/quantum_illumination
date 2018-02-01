import time
from datetime import datetime
import numpy as np
import pandas as pd

from qillumi.qi import QIExpr
from qillumi.utils import log


def expr_three_qhb_vs_energy(nth, n_max, divides, qcb_approx=True):
    # nss = np.linspace(0.01, 1, nss_divides)
    # lambdas = np.sqrt(nss / (1.0 + nss))
    # divides: divides lambda
    cols = ['Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom_Bound', 'Chernoff_Bound', 'optimal_s',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    lambdas = {'TMSS': np.linspace(0.001, 0.901, divides),
               'PS': np.linspace(0.001, 0.701, divides),
               'PA': np.linspace(0.001, 0.651, divides),
               'PSA': np.linspace(0.001, 0.681, divides),
               'PAS': np.linspace(0.001, 0.601, divides),
               'PCS': np.linspace(0.001, 0.601, divides)}
    names = ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS')
    # names = ('TMSS', 'PS', 'PSA')

    expr = QIExpr(n_max=n_max)
    expr.set_environment(reflectance=0.01, nth=nth)

    @log
    def set_laser(s_name, l):
        if s_name != 'PCS':
            expr.set_input_laser(s_name, l)
        else:
            expr.set_input_laser('PCS', l, rs=(0.2, 0.9))

    for name in names:
        for lmd in lambdas[name]:
            set_laser(name, lmd)
            try:
                expr.run_expr(qcb_approx=qcb_approx)
            except Exception as e:
                print("%s" % str(e))
            new_df = pd.DataFrame.from_dict({'res': expr.get_results()}, orient='index')
            df = df.append(new_df)

    write_data_to_file(df, nth, divides, cols)


def write_data_to_file(df, nth, nss_divides, cols):
    filename = "../output/data/expr_3_nbar_nth_{}_g_{}_{}.csv" \
        .format(nth, nss_divides, datetime.today().strftime('%m-%d'))
    with open(filename, 'w') as file:
        file.write("# Quantum Illumination Experiment\n")
        file.write("# Experiment for all states vs lambda.\n")
        file.write('# Author: L. Fan\n')
        file.write('# Created on: {}\n\n'.format(str(datetime.now())))
    df.to_csv(filename, index=False, columns=cols, mode='a')


if __name__ == "__main__":
    start_time = time.time()
    # expr_three_qhb_vs_energy(divides=11, n_max=24, nth=0.1, qcb_approx=True)
    # expr_three_qhb_vs_energy(divides=51, n_max=24, nth=0.1, qcb_approx=True)
    expr_three_qhb_vs_energy(divides=11, n_max=24, nth=1.0, qcb_approx=True)
    # expr_three_qhb_vs_energy(divides=51, n_max=24, nth=1.0, qcb_approx=True)
    print("--- %s seconds ---" % (time.time() - start_time))
