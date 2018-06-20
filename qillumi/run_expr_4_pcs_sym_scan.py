import time
from datetime import datetime
import numpy as np
import pandas as pd

import qillumi.laser2mode as laser
from qillumi.utils import log


def basic_pcs_attributes(n_max, l_divides, r_divides):
    cols = ['nmax', 'State', 'sqz', 'lambda', 'Exact_N', 'Aver_N',
            'VN_Entropy', 'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    lasers = {'TMSS': laser.TMSS, 'PS': laser.PS, 'PA': laser.PA,
              'PSA': laser.PSA, 'PAS': laser.PAS, 'PCS': laser.PCS}

    ss = {'PCS': np.linspace(1e-6, 1.0 + 1e-6, l_divides)}
    lambdas = {'PCS': np.tanh(ss['PCS'])}
    # lambdas = {'PCS': np.linspace(0.001, 0.751, l_divides)}
    rss = np.linspace(0.0, 1.0, r_divides)
    names = ('PCS',)

    @log
    def create_laser(s_name, l, rs):
        if s_name != 'PCS':
            s = lasers[s_name](l, n_max)
        else:
            s = laser.PCS(l, n_max, rs=rs)
            # s = laser.PCS(l, n_max, rs=(0.2, 0.9))
        return s

    for name in names:
        for lmd in lambdas[name]:
            for r in rss:
                state = create_laser(name, lmd, (r, r))
                new_df = pd.DataFrame.from_dict({'res': state.get_attrs()}, orient='index')
                df = df.append(new_df)

    df.to_csv("../output/data/expr_4_pcs_nmax_{:d}_{:d}x{:d}_{}.csv"
              .format(n_max, l_divides, r_divides, datetime.today().strftime('%m-%d')),
              index=False, columns=cols)


if __name__ == "__main__":
    start_time = time.time()
    basic_pcs_attributes(n_max=32, l_divides=10, r_divides=10)
    print("--- %s seconds ---" % (time.time() - start_time))

