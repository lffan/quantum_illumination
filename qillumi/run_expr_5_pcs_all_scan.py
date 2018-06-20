import time
from datetime import datetime
import numpy as np
import pandas as pd

import qillumi.laser2mode as laser
from qillumi.utils import log


def basic_pcs_attributes(n_max, l_divides, r_divides):
    cols = ['nmax', 'State', 'sqz', 'lambda', 'Aver_N',
            'VN_Entropy', 'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    # lasers = {'TMSS': laser.TMSS, 'PS': laser.PS, 'PA': laser.PA,
    #           'PSA': laser.PSA, 'PAS': laser.PAS, 'PCS': laser.PCS}

    lambdas = {'PCS': np.linspace(1e-6 + 0.3, 0.6 + 1e-6, l_divides)}
    rss = np.linspace(0.0, 1.0, r_divides)

    @log
    def create_laser(l, rs):
        s = laser.PCS(l, n_max, rs=rs)
        # s = laser.PCS(l, n_max, rs=(0.2, 0.9))
        return s

    filename = "../output/data/expr_5_pcs_nmax_{:d}_{:d}x{:d}x{:d}_{}.csv"\
        .format(n_max, l_divides, r_divides, r_divides, datetime.today().strftime('%m-%d'))

    with open(filename, "a") as file:
        file.write("# Quantum Illumination Experiment 0\n")
        file.write('# Entanglement and Average Photon Numbers\n')
        file.write('# Author: L. Fan\n')
        file.write('# Created on: {}\n\n'.format(str(datetime.now())))
        file.write(','.join(cols) + '\n')
        for lmd in lambdas['PCS']:
            for ra in rss:
                for rb in rss:
                    state = create_laser(lmd, (ra, rb))
                    attributes = state.get_attrs()
                    out_string = ','.join([str(attributes[col]) for col in cols]) + '\n'
                    file.write(out_string)
                    # new_df = pd.DataFrame.from_dict({'res': state.get_attrs()}, orient='index')
                    # df = df.append(new_df)

    # df.to_csv("../output/data/expr_4_pcs_nmax_{:d}_{:d}x{:d}_{}.csv"
    #           .format(n_max, l_divides, r_divides, datetime.today().strftime('%m-%d')),
    #           index=False, columns=cols)


if __name__ == "__main__":
    start_time = time.time()
    basic_pcs_attributes(n_max=32, l_divides=31, r_divides=51)
    print("--- %s seconds ---" % (time.time() - start_time))

