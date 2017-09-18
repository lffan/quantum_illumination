import time
import numpy as np
import pandas as pd

from qillumi.qi_utils import QIExpr


def run(n_max, nth, ns, rflct, rs, df):

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
        new_df = pd.DataFrame.from_dict({'res': expr.get_attrs()}, orient='index')
        df = df.append(new_df)

    return df


def run_pcs_vs_rs(n_max, nth, ns, rflct, rss, df):

    lmd = np.sqrt(ns / (1 + ns))
    expr = QIExpr(n_max)
    expr.set_environment(rflct, nth)

    # print("\nHelstrom Bounds\n")
    cts = 0
    for rs in rss:
        expr.set_input_laser('PCS', lmd, rs)
        expr.run_expr()
        new_df = pd.DataFrame.from_dict({'res': expr.get_attrs()}, orient='index')
        df = df.append(new_df)
        cts += 1
        print(cts)

    return df


def expr_one():
    cols = ['Nth', 'R', 'State', 'lambda', 'Aver_N',
            'VN_Entropy', 'Helstrom', 'Chernoff', 'S_opt',
            'A_N', 'B_N', 'ra', 'rb']
    df = pd.DataFrame(columns=cols)

    df = run(10, 0.1, 0.01, 0.01, (0.4, 0.4), df)
    df = run(15, 1.0, 0.01, 0.01, (0.4, 0.4), df)
    df = run(20, 10.0, 0.01, 0.01, (0.4, 0.4), df)

    print(df)
    df.to_csv('../output/expr_one.csv', index=False, columns=cols)


def expr_two(num_points):
    points = np.sqrt(np.linspace(0, 1, num_points))
    rss = [(ra, rb) for ra in points for rb in points]

    cols = ['Nth', 'R', 'State', 'lambda', 'ra', 'rb', 'Aver_N',
            'A_N', 'B_N', 'VN_Entropy', 'Helstrom', 'Chernoff', 'S_opt']
    df = pd.DataFrame(columns=cols)

    df = run_pcs_vs_rs(n_max=10, nth=0.1, ns=0.01, rflct=0.01, rss=rss, df=df)

    df.to_csv("../output/expr_two_p{}.csv".format(num_points),
              index=False, columns=cols)


def expr_three(num_points):
    points = np.linspace(0, 1, num_points)
    rss = [(ra, rb) for ra in points for rb in points]

    cols = ['Nth', 'R', 'State', 'lambda', 'ra', 'rb', 'Aver_N',
            'A_N', 'B_N', 'VN_Entropy', 'Helstrom', 'Chernoff', 'S_opt']
    df = pd.DataFrame(columns=cols)

    df = run_pcs_vs_rs(n_max=10, nth=0.1, ns=0.01, rflct=0.01, rss=rss, df=df)

    df.to_csv("../output/expr_two_p{}.csv".format(num_points),
              index=False, columns=cols)


if __name__ == "__main__":
    start_time = time.time()
    # expr_one()
    expr_three(51)
    print("--- %s seconds ---" % (time.time() - start_time))
