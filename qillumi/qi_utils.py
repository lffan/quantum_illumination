# -*- coding: utf-8 -*-
# Created on 2017-07-24
# Last Modified on 2017-07-24
# Latest Modified on 2017-09-14

""" Quantum Illumination """

import numpy as np
import qutip as qu

from qillumi import laser2mode as l2m

__author__ = 'Longfei Fan'


def qu_helstrom(rho0, rho1, p0=0.5, M=1):
    """ Calculate Helstrom error probability
        which is defined as

            P_e = 0.5 (1 - ||p1 * rho1 - p0 * rho0||)

        pi_0, rho_0: state 1 and its a priori probability pi_0
        pi_1, rho_1: state 2 and its a priori probability pi_1
        M: number of copies
    """
    if M == 1:
        return 0.5 * (1 - ((1 - p0) * rho1 - p0 * rho0).norm())
    else:
        pass  # TODO: for those M != 1


def qu_chernoff(rho0, rho1, approx=False):
    """ Approximated Q for QCB
        Actually the trace of sqrt(rho_1) * sqrt(rho_2)
    """
    if approx:
        # s = 0.5
        return (rho0.sqrtm() * rho1.sqrtm()).tr().real
    else:
        # TODO: give the optimal QCB by varying value of s
        pass


def upper_bound(QCB, M):
    """ Upper bound (Quantum Chernoff bound) of the error probability
        using s = 1/2
    """
    return 0.5 * QCB ** M


def lower_bound(tr, M):
    """ Lower bound of the error probability
    """
    return (1 - np.sqrt(1 - tr ** (2 * M))) / 2


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
        self.thermal_0 = None       # thermal state where an object may be embedded in
        self.thermal_1 = None       # thermal state adjusted by the reflection factor

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
        self.thermal_0 = qu.thermal_dm(self.n_max, nth)
        self.thermal_1 = qu.thermal_dm(self.n_max, nth / (1 - self.reflectance))

    # def set_expr(self, state_name, l, reflectance, nth, rs=False):
    #     """
    #     Setup the experiment
    #
    #     """
    #     self.set_input_laser(state_name, l, rs)
    #     self.set_environment(reflectance, nth)

    # def setup_experiment(self, names, ls, nth, rfl, rt=False):
    #     """
    #     Setup the two mode entangled laser, the thermal noise state,
    #     and the reflectance of the beam splitter
    #     """
    #     for name in names:
    #         for l in ls:
    #             if name not in self.lasers:
    #                 self.lasers[name] = [self.create_laser(name, l, rt)]
    #             else:
    #                 self.lasers[name].append(self.create_laser(name, l, rt))
    #     self.thermal_0 = qu.thermal_dm(self.n_max, nth)
    #     self.thermal_1 = qu.thermal_dm(self.n_max, nth / (1 - rfl))
    #     self.rfl = rfl

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
        qhb = qu_helstrom(rho0, rho1)
        qcb = qu_chernoff(rho0, rho1, approx=True)
        qcb1 = upper_bound(qcb, M=1)
        return qhb, qcb, qcb1


def run(n_max, nth, ns, rflct, rs):
    lmd = np.sqrt(ns / (1 + ns))

    expr = QIExpr(n_max)
    expr.set_environment(rflct, nth)

    print("\nHelstrom Bounds")
    for name in ('TMSS', 'PS', 'PA', 'PSA', 'PAS'):
        expr.set_input_laser(name, lmd)
        print("{} QHB: {}".format(name, expr.run_expr()[0]))
        print("{} QCB: {}".format(name, expr.run_expr()[2]))
    expr.set_input_laser('PCS', lmd, rs)
    print("{} QHB: {}".format('PCS', expr.run_expr()[0]))
    print("{} QCB: {}".format('PCS', expr.run_expr()[2]))


def test_run():
    run(11, 0.1, 0.01, 0.01, (0.7071, 0.7071))
    run(16, 1.0, 0.01, 0.01, (0.7071, 0.7071))
    run(21, 10.0, 0.01, 0.01, (0.7071, 0.7071))


if __name__ == "__main__":
    test_run()
