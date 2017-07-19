# -*- coding: utf-8 -*-
# Created on 2017-07-18
# Modified on 2017-07-18

"""quantum illumination class and utils"""

__author__ = 'Longfei Fan'


import numpy as np
from qutip import *
import laser2


class QuantumIllumin(object):
    """A class for a quantum illumination class"""
    
    def __init__(self, n_max, src_state, src_l, thml_aver_n, reflectance, rt=False):
        self.src = laser2.LaserTwoMode(src_state, src_l, n_max, rt)
        self.r = reflectance
        self.rho_absent = None
        self.rho_present = None
        
        
    def calc_rho_absent(self):
        self.rho_absent = tensor(self.src.get_state().ptrace(0), \
                                 thermal_dm(self.n_max, them_aver_n))
    
    
    def calc_rho_present(self):
        pass
    


def tm_sqz(n_max, s):
    """ Two-mode squeezing operator
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. 's' is the squeezing parameter.
        return: a qutip.Qobj(), operator
    """
    a = destroy(n_max)
    sqz = - np.conj(s) * tensor(a, a) + s * tensor(a.dag(), a.dag())
    return tms.expm()


def tm_mix(n_max, s):
    """ Two-mode mixing operator. Photon number is truncated at n_max. """
    a = destroy(n_max)
    tmm = s * tensor(a.dag(), a) - np.conj(s) * tensor(a, a.dag())
    return tmm.expm()
        