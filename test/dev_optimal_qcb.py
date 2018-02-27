# -*- coding: utf-8 -*-
# Created on 2017-07-24
# Last Modified on 2017-09-17
# Latest Modified on 2017-09-17

""" Test if Qobj().sqrtm() gets the same results with those via math"""

import numpy as np
import scipy.sparse as sp
import qutip as qu
from qutip.sparse import sp_eigs
from scipy.optimize import minimize


def power(qstate, power, sparse=False, tol=0, maxiter=100000):
    """power of a quantum operator.
    Operator must be square.
    Parameters
    ----------
    sparse : bool
        Use sparse eigenvalue/vector solver.
    tol : float
        Tolerance used by sparse solver (0 = machine precision).
    maxiter : int
        Maximum number of iterations used by sparse solver.
    Returns
    -------
    oper : qobj
        Matrix square root of operator.
    Raises
    ------
    TypeError
        Quantum object is not square.
    Notes
    -----
    The sparse eigensolver is much slower than the dense version.
    Use sparse only if memory requirements demand it.
    """
    if qstate.dims[0][0] == qstate.dims[1][0]:
        evals, evecs = sp_eigs(qstate.data, qstate.isherm, sparse=sparse,
                               tol=tol, maxiter=maxiter)
        numevals = len(evals)
        dV = sp.spdiags(np.power(evals, power, dtype=complex), 0,
                        numevals, numevals, format='csr')
        if qstate.isherm:
            spDv = dV.dot(evecs.T.conj().T)
        else:
            spDv = dV.dot(np.linalg.inv(evecs.T))

        out = qu.Qobj(evecs.T.dot(spDv), dims=qstate.dims)
        return out.tidyup() if qu.settings.auto_tidyup else out

    else:
        raise TypeError('Invalid operand for matrix square root')


def qcb_s(s, rho0, rho1):
    return (power(rho0, s) * power(rho1, 1 - s)).tr().real


def qcb_opt(rho0, rho1):
    """
    find the optimal s and the corresponding quantum Chernoff bound

    Parameters
    ----------
    rho0: qutip.Qobj()
        quantum state 1
    rho1: qutip.Qobj()
        quantum state 2

    Returns
    -------
    s: float
        optimal value of s
    qcb: float
        optimal value of quantum Chernoff bound
    """
    res = minimize(qcb_s, np.array([0.5]), args=(rho0, rho1, ),
                   method='Nelder-Mead', options={'disp': True})
    print(res.x)

    res = minimize(qcb_s, np.array([0.5]), args=(rho0, rho1,),
                   method='Powell', options={'disp': True})
    print(res.x)

    res = minimize(qcb_s, np.array([0.5]), args=(rho0, rho1,),
                   method='COBYLA', options={'disp': True})
    print(res.x)

    s = res.x
    if 0 <= s <= 1:
        return s
    else:
        raise ValueError("s should be within [0, 1].")


if __name__=="__main__":
    # qstate = qu.tensor(qu.coherent_dm(4, 1), qu.coherent_dm(4, 1))
    # print(qstate.sqrtm())
    # print(power(qstate, 0.5))

    s1 = qu.coherent_dm(10, 1)
    s2 = qu.coherent_dm(10, 3)
    qcb_opt(s1, s2)
