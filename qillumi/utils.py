from datetime import datetime


def log(func):
    """
    a decorator function to print running logs

    Parameters
    ----------
    func: functions to be run
    """
    count = 0

    def wrapper(*args, **kw):
        nonlocal count
        count += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        func_name = func.__name__
        paras = args[1:-1] if func_name == 'run_all_states' else args
        print('{}: {:d} {}{}\t'.format(now, count, func_name, paras))
        return func(*args, **kw)

    return wrapper
