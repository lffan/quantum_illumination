import numpy as np
import pandas as pd

from qillumi.qi_utils import QIExpr


def run(n_max, nth, ns, rflct, rs):
    cols = ['State', 'Aver N', 'Nth', 'R', 'Helstrom', 'Chernoff', 'S_opt']
    df = pd.DataFrame(columns=cols)

    lmd = np.sqrt(ns / (1 + ns))
    expr = QIExpr(n_max)
    expr.set_environment(rflct, nth)

    # print("\nHelstrom Bounds\n")
    for name in ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS'):
        if name != 'PCS':
            expr.set_input_laser(name, lmd)
        else:
            expr.set_input_laser('PCS', lmd, rs)
        expr.run_expr()
        df = pd.concat([df, pd.DataFrame.from_dict({'res': expr.get_attr()}, orient='index')])

    # df.reset_index(drop=True, inplace=True)
    df.to_csv('../output/result.csv', index=False, mode='a', header=False)


def expr_one():
    run(10, 0.1, 0.01, 0.01, (0.4, 0.4))
    run(15, 1.0, 0.01, 0.01, (0.4, 0.4))
    run(20, 10.0, 0.01, 0.01, (0.4, 0.4))


if __name__ == "__main__":
    expr_one()