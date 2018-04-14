# -*- coding: utf-8 -*-
# Created: 2017-07-18
# Modified: 2017-09-14

""" two-mode laser, TMSS and non-Gaussian states """

import qutip as qu
import numpy as np

__author__ = 'Longfei Fan'


class LaserTwoMode(object):
    """A class for two-mode lasers"""

    def __init__(self, l, n_max):
        """
        To initialize a two-mode state.

        Parameters
        ----------
        l: float
            lambda factor defined in equations of states
        n_max: integer
            maximum photon number for calculation is n_max - 1

        Return
        -----
        ValueError:
            when n_max is not valid
        """
        if n_max > 0:
            self.n_max = n_max
            self.lmd = l
            self.state_name = None
            self.state = None
            self.num = None
            self.exact_num = None
            self.entanglement = None
        else:
            raise ValueError("N must be a positive integer.")

    def get_attrs(self):
        return {'nmax': self.n_max, 'State': self.state_name,
                'sqz': np.arctanh(self.lmd), 'lambda': self.lmd,
                'Exact_N': self.exact_num, 'Aver_N': self.num,
                'A_aver_N': self.num / 2, 'B_aver_N': self.num / 2,
                'VN_Entropy': self.entanglement}


class TMSS(LaserTwoMode):
    """
    Two-mode squeezed state |TMSS \rangle = sum_0^n \lambda^n |n, n\rangle

    Parameters
    ----------
    l: double float
        state parameter, for TMSS l = tanh(s), where s is the squeezed para
    n_max: positive int
        photon truncation number as we are doing numerical calculation
    expr.set_input_laser(s_name, l)   i.e. photon numbers can be in [0, n_max - 1]

    Return
    ------
    qutip.Qobj()
        a qutip object, a TMSS in bra form (column vector)
    """

    def __init__(self, l, n_max):
        super().__init__(l, n_max)
        self.state_name = "TMSS"
        self.state = qu.Qobj(np.sum([l ** n * qu.tensor(qu.basis(n_max, n), qu.basis(n_max, n))
                                     for n in np.arange(n_max)[::-1]])).unit()
        self.num = qu.expect(qu.num(self.n_max), self.state.ptrace(0)) * 2
        self.exact_num = l ** 2 / (- l ** 2 + 1) * 2
        self.entanglement = qu.entropy_vn(self.state.ptrace(0))
        # print("TMSS aver n: {}, {}".format(self.num, self.exact_num))


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
        super().__init__(l, n_max)
        self.state_name = "PS"
        self.state = qu.Qobj(np.sum([(n + 1) * l ** n * qu.tensor(qu.basis(n_max, n), qu.basis(n_max, n))
                                     for n in np.arange(n_max)[::-1]])).unit()
        self.num = qu.expect(qu.num(self.n_max), self.state.ptrace(0)) * 2
        self.exact_num = 4 * l ** 2 * (l ** 2 + 2) / (1 - l ** 4)
        self.entanglement = qu.entropy_vn(self.state.ptrace(0))
        # print("PS aver n: {}, {}".format(self.num, self.exact_num))


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
        super().__init__(l, n_max)
        self.state_name = "PA"
        self.state = qu.Qobj(np.sum([(n + 1) * l ** n * qu.tensor(qu.basis(n_max, n + 1), qu.basis(n_max, n + 1))
                                     for n in np.arange(n_max - 1)[::-1]])).unit()
        self.num = qu.expect(qu.num(self.n_max), self.state.ptrace(0)) * 2
        self.exact_num = 2 * (l ** 4 + 4 * l ** 2 + 1) / (- l ** 4 + 1)
        self.entanglement = qu.entropy_vn(self.state.ptrace(0))
        # print("PA aver n:{}, {}".format(self.num, self.exact_num))


class PSA(LaserTwoMode):
    """Photon added then subtracted state"""

    def __init__(self, l, n_max):
        super().__init__(l, n_max)
        self.state_name = "PSA"
        self.state = qu.Qobj(np.sum([(n + 1) ** 2 * l ** n * qu.tensor(qu.basis(n_max, n), qu.basis(n_max, n))
                                     for n in np.arange(n_max)[::-1]])).unit()
        self.num = qu.expect(qu.num(self.n_max), self.state.ptrace(0)) * 2
        self.exact_num = 4 * l**2 * (l**6 + 18 * l**4 + 33 * l**2 + 8) / (- l**8 - 10 * l**6 + 10 * l**2 + 1)
        self.entanglement = qu.entropy_vn(self.state.ptrace(0))
        # print("PSA aver n: {}, {}".format(self.num, self.exact_num))


class PAS(LaserTwoMode):
    """Photon subtracted then added state"""

    def __init__(self, l, n_max):
        super().__init__(l, n_max)
        self.state_name = "PAS"
        self.state = qu.Qobj(np.sum([(n + 1) ** 2 * l ** n * qu.tensor(qu.basis(n_max, n + 1), qu.basis(n_max, n + 1))
                                     for n in np.arange(n_max - 1)[::-1]])).unit()
        self.num = qu.expect(qu.num(self.n_max), self.state.ptrace(0)) * 2
        self.exact_num = 2 * (l**8 + 26 * l**6 + 66 * l**4 + 26 * l**2 + 1) / (- l**8 - 10 * l**6 + 10 * l**2 + 1)
        self.entanglement = qu.entropy_vn(self.state.ptrace(0))
        # print("PAS aver n: {}, {}".format(self.num, self.exact_num))


# class PCS_sym_etgl(LaserTwoMode):
#     def __int__(self, l, n_max):
#         super.__init__(l, n_max)
#


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
        rs: a list of int
            define the superposition factor
        """
        super().__init__(l, n_max)
        self.state_name = "PCS"
        self.rs = rs
        self.state = self.__create_state(l, n_max, rs)
        self.a_num = qu.expect(qu.num(self.n_max), self.state.ptrace(0))
        self.b_num = qu.expect(qu.num(self.n_max), self.state.ptrace(1))
        self.num = self.a_num + self.b_num
        self.entanglement = qu.entropy_vn(self.state.ptrace(1))
        # print("{}".format(self.entanglement))
        # print("PCS aver n: {}".format(self.num))

    def get_attrs(self):
        return {'nmax': self.n_max, 'State': self.state_name,
                'sqz': np.arctanh(self.lmd), 'lambda': self.lmd,
                'Aver_N': self.num, 'A_aver_N': self.a_num, 'B_aver_N': self.b_num,
                'VN_Entropy': self.entanglement,
                'ra': self.rs[0], 'rb': self.rs[1]}

    @staticmethod
    def __create_state(l, n_max, rs):
        """
        create a PCS state
        """
        ra, rb = rs
        if ra > 1:
            if ra - 1 < 1e-6:
                ra = 1
            else:
                raise ValueError("ra cannot be larger than 1.0")
        if rb > 1:
            if rb - 1 < 1e-6:
                rb = 1
            else:
                raise ValueError("rb cannot be larger than 1.0")

        ta = np.sqrt(1 - ra ** 2)
        tb = np.sqrt(1 - rb ** 2)
        nums = np.arange(n_max)[::-1]

        state1 = np.sum([l ** n * (n + 1) *
                         qu.tensor(qu.basis(n_max, n), qu.basis(n_max, n)) for n in nums])

        state2 = np.sum([l ** n * np.sqrt((n + 1) * (n + 2)) *
                         qu.tensor(qu.basis(n_max, n), qu.basis(n_max, n + 2)) for n in nums[2:]])

        state3 = np.sum([l ** n * np.sqrt((n + 1) * (n + 2)) *
                         qu.tensor(qu.basis(n_max, n + 2), qu.basis(n_max, n)) for n in nums[2:]])

        state4 = np.sum([l ** n * (n + 1) *
                         qu.tensor(qu.basis(n_max, n + 1), qu.basis(n_max, n + 1)) for n in nums[1:]])

        return qu.Qobj(l * ta * tb * state1 + l * ta * rb * state2 +
                       l * ra * tb * state3 + ra * rb * state4).unit()


def tm_sqz(s, n_max):
    """
    Two mode mixing operator in the form of matrix with Fock basis

    Parameters
    ----------
    s: float (TODO: a complex number??)
        The squeezed parameter
    n_max: positive integer
        Photon numbers can be in [0, n_max - 1]

    Return
    ------
        a new qubit.Qobj()
    """
    a = qu.destroy(n_max)
    tms = - np.conj(s) * qu.tensor(a, a) + s * qu.tensor(a.dag(), a.dag())
    return tms.expm()


def tm_mix(s, n_max):
    """
    Two-mode mixing operator (Beam splitter)

    Parameters
    ----------
    s: float (TODO: a complex number??)
        The mixing parameter
    n_max: positive interger
        Photon numbers can be in [0, n_max - 1]

    Return
    ------
        a new qubit.Qobj()
    """
    a = qu.destroy(n_max)
    tmm = s * qu.tensor(a.dag(), a) - np.conj(s) * qu.tensor(a, a.dag())
    return tmm.expm()


def create_states():
    print('\n')
    n_max = 10
    lmd = np.sqrt(0.01/1.01)

    # Test creating states
    state_all = [TMSS(lmd, n_max), PS(lmd, n_max), PA(lmd, n_max),
                 PSA(lmd, n_max), PAS(lmd, n_max), PCS(lmd, n_max, (0.7, 0.7))]

    for item in state_all:
        name = item.state_name
        state = item.state
        print("State: {}".format(item.state_name))
        print("Aver N: {:f} (numerical), {:f} (analytical)".format(item.num, item.exact_num if item.state_name != 'PCS' else 0))
        print("Entropy of entanglement: {:f}\n".format(item.entanglement))

    # Test two-mode mixing
    input1 = qu.ket2dm(qu.tensor(qu.basis(2, 0), qu.basis(2, 1)))
    tm_mix_op = tm_mix(0.1, 2)
    output1 = tm_mix_op * input1 * tm_mix_op.dag()
    print(input1)
    print(output1)

    # Test two-mode squeezing
    s = np.arctanh(lmd)
    input2 = qu.ket2dm(qu.tensor(qu.basis(n_max, 0), qu.basis(n_max, 0)))
    tm_sqz_op = tm_sqz(s, n_max)
    output2 = tm_sqz_op * input2 * tm_sqz_op.dag()
    print("\nEntropy of entanglement: {}".format(qu.entropy_vn(output2.ptrace(1))))


def create_pcs():
    n_max = 10
    lmd = np.sqrt(0.01 / 1.01)

    points = np.linspace(0, 1, 5)
    rss = [(ra, rb) for ra in points for rb in points]
    for rs in rss:
        pcs = PCS(lmd, n_max, rs)
        print(pcs.get_attrs())


if __name__ == "__main__":
    create_states()
    create_pcs()
