# -*- coding: utf-8 -*-
# Created on 2017-07-24
# Last Modified on 2017-07-24
# Latest Modified on 2017-09-14

""" Quantum Illumination """

import numpy as np
import qutip as qu

from qillumi import laser2mode as l2m

__author__ = 'Longfei Fan'


class QuIllumination(object):
    """
    Automatically numerical quantum illumination experiment
    """
    
    def __init__(self, n_max):
        """
        Initialize the QuIllumin object.
        Set the maximum photon number used for numerical simulation
        """
        self.n_max = n_max      # Fock numbers are in [0, n_max - 1]
        self.lasers = {}        # lasers used for experiment
        self.thermal_0 = None   # thermal state where an object may be embedded in
        self.thermal_1 = None
        self.rfl = None         # reflectance of the beam splitter

    def create_laser(self, name, l, rs=False):
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

    def setup_experiment(self, names, ls, nth, rfl, rt=False):
        """
        Setup the two mode entangled laser, the thermal noise state, 
        and the reflectance of the beam splitter
        """
        for name in names:
            for l in ls:
                if name not in self.lasers:
                    self.lasers[name] = [self.create_laser(name, l, rt)]
                else:
                    self.lasers[name].append(self.create_laser(name, l, rt))
        self.thermal_0 = qu.thermal_dm(self.n_max, nth)
        self.thermal_1 = qu.thermal_dm(self.n_max, nth / (1 - rfl))
        self.rfl = rfl

    def rho_0(self, input_laser):
        """
        Output state rho 0 if an object is absent
        """
        rho_AB = qu.ket2dm(input_laser)
        return qu.tensor(rho_AB.ptrace(0), self.thermal_0)  # rho_A (index 0) is kept here.

    def rho_1(self, input_laser):
        """
        Output state rho 1 if an object is present
        """
        rho_AB = qu.ket2dm(input_laser)

        rho_ABC = qu.tensor(rho_AB, self.thermal_1)

        # state A unchanged, tm_mixing acted on state B and thermal
        xi = np.arccos(np.sqrt(kappa))
        op = qu.tensor(qeye(N), self.__tm_mix(xi))

        rho_1 = op * rho_ABC * op.dag()
        return rho_1.ptrace([0, 1])  # keep A and B (index 0, 1)

    def run_experiment(self):
        """
        Run the experiment according to the parameters setup
        """
        pass

    def Helstrom(self, pi_0, rho_0, pi_1, rho_1, M=1):
        """ Calculate Helstrom error probability
            which is defined as

                P_e = 0.5 (1 - ||p1 * rho1 - p0 * rho0||)

            pi_0, rho_0: state 1 and its a priori probability pi_0
            pi_1, rho_1: state 2 and its a priori probability pi_1
            M: number of copies
        """
        if M == 1:
            gamma = pi_1 * rho_1 - pi_0 * rho_0
        else:
            pass # TODO: for those M != 1
        return 0.5 * (1 - gamma.norm())

    def Chernoff(self, rho_0, rho_1, approx=False):
        """ Approximated Q for QCB
            Actually the trace of sqrt(rho_1) * sqrt(rho_2)
        """
        if approx:
            # s = 0.5
            return (rho_0.sqrtm() * rho_1.sqrtm()).tr().real
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