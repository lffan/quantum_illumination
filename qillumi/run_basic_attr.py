import time
from datetime import datetime
import numpy as np
import pandas as pd

import qillumi.laser2mode as laser
from qillumi.qi import QIExpr
from qillumi.utils import log


def basic_attributes(n_max, divides):
    cols = ['State', 'lambda', 'Exact_N', 'Aver_N', 'VN_Entropy',
            'A_aver_N', 'B_aver_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    lasers = {'TMSS': laser.TMSS, 'PS': laser.PS, 'PA': laser.PA,
              'PSA': laser.PSA, 'PAS': laser.PAS, 'PCS': laser.PCS}

    lambdas = {'TMSS': np.linspace(0.001, 0.801, divides),
               'PS': np.linspace(0.001, 0.601, divides),
               'PA': np.linspace(0.001, 0.601, divides),
               'PSA': np.linspace(0.001, 0.551, divides),
               'PAS': np.linspace(0.001, 0.501, divides),
               'PCS': np.linspace(0.001, 0.651, divides)}
    names = ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS')

    @log
    def create_laser(s_name, l):
        if s_name != 'PCS':
            s = lasers[s_name](l, n_max)
        else:
            s = laser.PCS(l, n_max, rs=(0.4, 0.4))
            # s = laser.PCS(l, n_max, rs=(0.2, 0.9))
        return s

    for name in names:
        for lmd in lambdas[name]:
            state = create_laser(name, lmd)
            new_df = pd.DataFrame.from_dict({'res': state.get_attrs()}, orient='index')
            df = df.append(new_df)

    df.to_csv('../output/basic_attributes.csv', index=False, columns=cols)


if __name__ == "__main__":
    start_time = time.time()
    basic_attributes(64, 11)
    print("--- %s seconds ---" % (time.time() - start_time))
