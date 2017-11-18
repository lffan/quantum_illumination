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


def expr_four_pcs_same_ns():
    pass


if __name__ == "__main__":
    start_time = time.time()
    expr_four_pcs_same_ns(101)
    print("--- %s seconds ---" % (time.time() - start_time))
