# -*- coding: utf-8 -*-
# Created on 2017-07-24
# Last Modified on 2017-07-24
# Latest Modified on 2017-09-14

""" Quantum Illumination """

import numpy as np
import qutip as qu
from scipy.sparse import spdiags
from scipy.optimize import minimize
from qutip.sparse import sp_eigs

from qillumi import laser2mode as l2m

__author__ = 'Longfei Fan'


class QIExpr(object):
    """
    Automatically numerical quantum illumination experiment
    """
    
    def __init__(self, n_max):
        """
        Initialize the QuIllumin object.
        Set the maximum photon number used for numerical simulation
        """
        self.n_max = n_max          # Fock numbers are in [0, n_max - 1]
        self.laser = None           # laser used for experiment
        self.reflectance = None     # reflectance of the beam splitter
        self.nth = None             # average photon number of the thermal state
        self.thermal_0 = None       # thermal state where an object may be embedded in
        self.thermal_1 = None       # thermal state adjusted by the reflection factor
        self.qhb = None
        self.qcb = [0.5, 0.5]

    def __create_laser(self, name, l, rs=False):
        """
        Setup the entangled laser source for detection
        """
        laser = None

        if name == 'TMSS':
            laser = l2m.TMSS(l, self.n_max)
        elif name == 'PS':
            laser = l2m.PS(l, self.n_max)
        elif name == 'PA':
            laser = l2m.PA(l, self.n_max)
        elif name == 'PSA':
            laser = l2m.PSA(l, self.n_max)
        elif name == 'PAS':
            laser = l2m.PAS(l, self.n_max)
        elif name == "PCS":
            laser = l2m.PCS(l, self.n_max, rs)

        return laser

    def set_input_laser(self, state_name, lmd, rs=False):
        """
        Setup the two-mode entangled laser as input
        Parameters
        ----------
        state_name: string
            to specific what kind of laser as an input
        lmd: float
            lambda parameter for the laser, lmd = (N / (1 + N)) ** 0.5
        rs: 2-elements tuple of float
            used for PCS state

        Returns
        -------
        no return, alternate self.laser inplace

        """
        self.laser = self.__create_laser(state_name, lmd, rs)

    def set_environment(self, reflectance, nth):
        """
        Set up the thermal noise bath

        Parameters
        ----------
        reflectance: float
            reflectance of the possibly-exciting target object
        nth: float
            average photon number of the thermal noise

        Returns
        -------
        no return, alternate self.thermal_0 and self.thermal_1 inplace
        """
        self.reflectance = reflectance
        self.nth = nth
        self.thermal_0 = qu.thermal_dm(self.n_max, nth)
        self.thermal_1 = qu.thermal_dm(self.n_max, nth / (1 - self.reflectance))

    def __evolve_rho0(self):
        """
        Output state rho 0 if an object is absent
        """
        rho_ab = qu.ket2dm(self.laser.state)
        return qu.tensor(rho_ab.ptrace(0), self.thermal_0)  # rho_A (index 0) is kept here.

    def __evolve_rho1(self):
        """
        Output state rho 1 if an object is present
        """
        rho_ab = qu.ket2dm(self.laser.state)
        rho_abc = qu.tensor(rho_ab, self.thermal_1)

        # state A unchanged, tm_mixing acted on state B and thermal
        xi = np.arccos(np.sqrt(self.reflectance))
        op = qu.tensor(qu.qeye(self.n_max), l2m.tm_mix(xi, self.n_max))
        rho_1 = op * rho_abc * op.dag()

        return rho_1.ptrace([0, 1])  # keep A and B (index 0, 1)

    def run_expr(self):
        """
        Run the experiment according to the parameters setup
        """
        rho0 = self.__evolve_rho0()
        rho1 = self.__evolve_rho1()
        self.qhb = qu_helstrom(rho0, rho1)
        qcb = qu_chernoff(rho0, rho1, approx=False)
        self.qcb[0] = qcb[0]
        self.qcb[1] = upper_bound(qcb[1], M=1)
        # print(self.qhb, self.qcb)

    def get_results(self):
        """
        Return some useful information
        """
        laser_attrs = self.laser.get_attrs()
        expr_attrs = {'Nth': self.nth, 'R': self.reflectance,
                      'Helstrom_Bound': self.qhb, 'Chernoff_Bound': self.qcb[1],
                      'optimal_s': np.round(self.qcb[0], 6)}
        return {**laser_attrs, **expr_attrs}


def power(qstate, power, sparse=False, tol=0, maxiter=100000):
    """power of a quantum operator.
    Operator must be square.
    Parameters
    ----------
    qstate: qutip.Qobj()
        quantum state
    power: float
        power
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
        dV = spdiags(np.power(evals, power, dtype=complex), 0,
                     numevals, numevals, format='csr')
        if qstate.isherm:
            spDv = dV.dot(evecs.T.conj().T)
        else:
            spDv = dV.dot(np.linalg.inv(evecs.T))

        out = qu.Qobj(evecs.T.dot(spDv), dims=qstate.dims)
        return out.tidyup() if qu.settings.auto_tidyup else out

    else:
        raise TypeError('Invalid operand for matrix square root')


def qu_helstrom(rho0, rho1, p0=0.5, M=1):
    """ Calculate Helstrom error probability
        which is defined as

            P_e = 0.5 (1 - ||p1 * rho1 - p0 * rho0||)

        pi_0, rho_0: state 1 and its a priori probability pi_0
        pi_1, rho_1: state 2 and its a priori probability pi_1
        M: number of copies
        || rho || is the trace norm
    """
    if M == 1:
        q1 = 0.5 * (1 - ((1 - p0) * rho1 - p0 * rho0).norm())
        return q1
    else:
        pass  # TODO: for those M != 1


def qcb_s(s, rho0, rho1):
    """
    Tr[rho ** s - rho ** (1 - s)]
    """
    return (power(rho0, s) * power(rho1, 1 - s)).tr().real


def qu_chernoff(rho0, rho1, approx=False):
    """ Approximated Q for QCB
        Actually the trace of sqrt(rho_1) * sqrt(rho_2)
    """
    if approx:
        # s = 0.5
        return 0.5, (rho0.sqrtm() * rho1.sqrtm()).tr().real
    else:
        res = minimize(qcb_s, np.array([0.3]), args=(rho0, rho1,),
                       method='Nelder-Mead', options={'disp': False})
        s = res.x[0]
        if 0 <= s <= 1:
            return s, qcb_s(s, rho0, rho1)
        else:
            raise ValueError("s should be within [0, 1].")


def upper_bound(QCB, M):
    """ Upper bound (Quantum Chernoff bound) of the error probability
        using s = 1/2
    """
    return 0.5 * QCB ** M


def lower_bound(tr, M):
    """ Lower bound of the error probability
    """
    return (1 - np.sqrt(1 - tr ** (2 * M))) / 2


def run(n_max, nth, ns, rflct, rs):

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
        print(expr.get_results())


def expr_one():
    run(10, 0.1, 0.01, 0.01, (0.4, 0.4))
    run(15, 1.0, 0.01, 0.01, (0.4, 0.4))


if __name__ == "__main__":
    expr_one()
