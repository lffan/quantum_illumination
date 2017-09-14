# -*- coding: utf-8 -*-
# Created: 2017-07-18
# Modified: 2017-09-14

""" two-mode laser, TMSS and non-Gaussian states """

from qutip import *
import numpy as np

__author__ = 'Longfei Fan'


class LaserTwoMode(object):
    """A class for two-mode lasers"""

    def __init__(self, n_max):
        """
        To initialize a two-mode state.

        Parameters
        ----------
        n_max: integer
            maximum photon number for calculation is n_max - 1

        Return
        -----
        ValueError:
            when n_max is not valid
        """
        if n_max > 0:
            self.n_max = n_max
            self.name = None
            self.state = None
        else:
            raise ValueError("N must be a positive integer.")

    def __tm_sqz(self, s):
        """
        Two mode mixing operator in the form of matrix with Fock basis

        Parameters
        ----------
        s: float (TODO: a complex number??)
            The squeezed parameter

        Return
        ------
            qubit.Qobj()
        """
        a = destroy(self.n_max)
        tms = - np.conj(s) * tensor(a, a) + s * tensor(a.dag(), a.dag())
        return tms.expm()

    def __tm_mix(self, s):
        """
        Two-mode mixing operator (Beam splitter)

        Parameters
        ----------
        s: float (TODO: a complex number??)
            The mixing parameter

        Return
        ------
        qubit.Qobj()
        """
        a = destroy(self.n_max)
        tmm = s * tensor(a.dag(), a) - np.conj(s) * tensor(a, a.dag())
        return tmm.expm()


class TMSS(LaserTwoMode):
    """
    Two-mode squeezed state |TMSS \rangle = sum_0^n \lambda^n |n, n\rangle

    Parameters
    ----------
    l: double float
        state parameter, for TMSS l = tanh(s), where s is the squeezed para
    n_max: positive int
        photon truncation number as we are doing numerical calculation
        i.e. photon numbers can be in [0, n_max - 1]

    Return
    ------
    qutip.Qobj()
        a qutip object, a TMSS in bra form (column vector)
    """

    def __init__(self, l, n_max):
        super().__init__(n_max)
        self.state_name = "TMSS"
        self.state = np.sum([l ** n * tensor(basis(n_max, n), basis(n_max, n))
                             for n in np.arange(n_max)])


class PS(LaserTwoMode):

    def __init__(self, l, n_max):
        """
        Photon subtracted state |PS> = a b |TMSS>

        Parameters
        ----------
        l: double float
            state parameter, for TMSS l = tanh(s), where s is the squeezed para
        n_max: positive int
            photon truncation number as we are doing numerical calculation
            i.e. photon numbers can be in [0, n_max - 1]

        Return
        ------
        qutip.Qobj()
            a qutip object, a photon subtracted state in bra form
        """
        super().__init__(n_max)
        self.state_name = "PS"
        self.state = np.sum([(n + 1) * l ** n * tensor(basis(n_max, n), basis(n_max, n))
                             for n in np.arange(n_max)])


class PA(LaserTwoMode):

    def __init__(self, l, n_max):
        """
        Photon added state |PA> = a^\dagger b^\dagger |TMSS>

        Parameters
        ----------
        l: double float
            state parameter, for TMSS l = tanh(s), where s is the squeezed para
        n_max: positive int
            photon truncation number as we are doing numerical calculation
            i.e. photon numbers can be in [0, n_max - 1]

        Return
        ------
        qutip.Qobj()
            a qutip object, a photon added state in bra form
        """
        super().__init__(n_max)
        self.state_name = "PA"
        self.state = np.sum([(n + 1) * l ** n * tensor(basis(n_max, n + 1), basis(n_max, n + 1))
                             for n in np.arange(n_max - 1)])


class PSA(LaserTwoMode):
    """Photon added then subtracted state"""

    def __init__(self, l, n_max):
        super().__init__(n_max)
        self.state_name = "PSA"
        self.state = np.sum([(n + 1) ** 2 * l ** n * tensor(basis(n_max, n), basis(n_max, n))
                             for n in np.arange(n_max)])


class PAS(LaserTwoMode):
    """Photon subtracted then added state"""

    def __init__(self, l, n_max):
        super().__init__(n_max)
        self.state_name = "PAS"
        self.state = np.sum([(n + 1) ** 2 * l ** n * tensor(basis(n_max, n + 1), basis(n_max, n + 1))
                             for n in np.arange(n_max - 1)])


class PCS(LaserTwoMode):

    def __init__(self, l, n_max, rs):
        """
        States obtained by a coherent superposition operation of photon subtraction
        and addition on two-mode squeezed state

        l: double float
            state parameter, for TMSS l = tanh(s), where s is the squeezed para
        n_max: positive int
            photon truncation number as we are doing numerical calculation
            i.e. photon numbers can be in [0, n_max - 1]
        rt: int list, optional
            superposition factors used when state == "PCS"

        Parameters
        ----------
        rt: a list of int
            define the superposition factor
        """
        super().__init__(n_max)
        self.state_name = "PCS"
        self.__create_state(l, n_max, rs)

    def __create_state(self, l, n_max, rs):
        """
        create a PCS state
        """
        ra, rb = rs
        ta = np.sqrt(1 - ra ** 2)
        tb = np.sqrt(1 - rb ** 2)
        nums = np.arange(n_max)

        state1 = np.sum([l ** (n + 1) * (n + 1) *
                         tensor(basis(n_max, n), basis(n_max, n)) for n in nums])

        state2 = np.sum([l ** (n + 1) * np.sqrt((n + 1) * (n + 2)) *
                         tensor(basis(n_max, n), basis(n_max, n + 2)) for n in nums[:-2]])

        state3 = np.sum([l ** (n + 1) * np.sqrt((n + 1) * (n + 2)) *
                         tensor(basis(n_max, n + 2), basis(n_max, n)) for n in nums[:-2]])

        state4 = np.sum([l ** n * (n + 1) *
                         tensor(basis(n_max, n + 1), basis(n_max, n + 1)) for n in nums[:-1]])

        self.state = ta * tb * state1 + ta * rb * state2 + ra * tb * state3 + ra * rb * state4


def test_run():
    n_max = 64
    lmd = 0.1
    state_all = [TMSS(lmd, n_max), PS(lmd, n_max), PA(lmd, n_max),
                 PSA(lmd, n_max), PAS(lmd, n_max), PCS(lmd, n_max, (0.7, 0.7))]
    for item in state_all:
        name = item.state_name
        state = item.state
        print("State: {}".format(name))
        print("<n>: {}".format(expect(num(item.n_max), state.ptrace(1))))
        print("Entropy: {}\n".format(entropy_vn(state.ptrace(1))))


if __name__ == "__main__":
    test_run()
