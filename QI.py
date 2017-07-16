#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created on 2016-07-04
# Modified on 2016-11-30

""" Quantum Illumination with Non-Gaussian States """

__author__ = 'Longfei Fan'


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial


# two-mode operator ==========================================================


def tm_squeezing(N, s):
    """ Two-mode squeezing operator
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. 's' is the squeezing parameter.
        return: a qutip.Qobj(), operator
    """
    a = destroy(N)
    tms = - np.conj(s) * tensor(a, a) + s * tensor(a.dag(), a.dag())
    return tms.expm()


def tm_mixing(N, s):
    """ Two-mode mixing operator (Beam splitter)
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. The squeezed parameter
        return: a qubit.Qobj(), operator
    """
    a = destroy(N)
    tmm = s * tensor(a.dag(), a) - np.conj(s) * tensor(a, a.dag())
    return tmm.expm()


# one-mode states ==========================================================


def cohe(N, alpha):
    """ Coherent states
        N: truncted photon number
        alpha: complex number (eigenvalue) for requested coherent state
    """
    return coherent_dm(N, alpha)


# two-mode states ==========================================================


def TMSS(N, l):
    """ Two-mode squeezed state
        N: a positive interger. Photon number is truncated at N, i.e [0, N-1]
        s: a complex number. 's' is the squeezing parameter.
        l: l = tanh(s)
        return: a qutip.Qobj(), vector state
    """
    # l = np.tanh(s.real)
    state = np.sum([l**n * tensor(basis(N, n), basis(N, n)) \
                    for n in xrange(N)])
    return state.unit()


def p_sub(N, l):
    """ Photon subtracted two-mode squeezed state
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. 's' is the squeezing parameter.
        return: a qutip.Qobj(), vector state
    """
    # l = np.tanh(s.real)
    state = np.sum([(n+1) * l**n * tensor(basis(N, n), basis(N, n)) \
                    for n in xrange(N)])
    return state.unit()


def p_add(N, l):
    """ Photon added two-mode squeezed state
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. 's' is the squeezing parameter.
        return: a qutip.Qobj(), vector state
    """
    # l = np.tanh(s.real)
    state = np.sum([(n+1) * l**n * tensor(basis(N, n + 1), basis(N, n + 1)) \
                    for n in xrange(N - 1)])
    return state.unit()    


def p_sub_add(N, l):
    """ Photon added then subtracted two-mode squeezed state
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. 's' is the squeezing parameter.
        return: a qutip.Qobj(), vector state
    """
    # l = np.tanh(s.real)
    state = np.sum([(n+1)**2 * l**n * tensor(basis(N, n), basis(N, n)) \
                    for n in xrange(N)])
    return state.unit()


def p_add_sub(N, l):
    """ Photon subtracted then added two-mode squeezed state
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. 's' is the squeezing parameter.
        return: a qutip.Qobj(), vector state
    """
    # l = np.tanh(s.real)
    state = np.sum([(n+1)**2 * l**n * tensor(basis(N, n + 1), basis(N, n + 1)) \
                    for n in xrange(N - 1)])
    return state.unit()


def p_cohe_sub_add(N, l, rt_list):
    """ Coherent superposition of photon subtraction and addition 
        on two-mode squeezed state
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. 's' is the squeezing parameter.
        r: complex numbers.
        return: a qutip.Qobj(), vector state
    """
    # l = np.tanh(s.real)
    ta, ra, tb, rb = rt_list
    
    nlist = range(N)
    state1 = np.sum([l**(n+1) * (n+1) * tensor(basis(N, n), basis(N, n)) \
                    for n in nlist])
    state2 = np.sum([l**(n+1) * np.sqrt((n+1)*(n+2)) \
                    * tensor(basis(N, n), basis(N, n + 2)) for n in nlist[:-2]])
    state3 = np.sum([l**(n+1) * np.sqrt((n+1)*(n+2)) \
                    * tensor(basis(N, n + 2), basis(N, n)) for n in nlist[:-2]])
    state4 = np.sum([l**n * (n+1) * tensor(basis(N, n + 1), basis(N, n + 1)) \
                    for n in nlist[:-1]])
    state = ta * tb * state1 + ta * rb * state2 + \
            ra * tb * state3 + ra * rb * state4
    
    return state.unit()


# calculate lambda from average photon numbers =============================


# rho_0 and rho_1 ==========================================================


def RHO_0(state, N, l, Nth, rt_list=False):
    """ State obtained if the object is absent.
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        l: a complex number. 'l' is the squeezing parameter.
        r: complex numbers.
        return: a qutip.Qobj(), density matrix
    """
    if not rt_list:
        rho_AB = ket2dm(state(N, l))
    else:
        rho_AB = ket2dm(state(N, l, rt_list))
    # ptrace(sel), sel is a list of intergers that mark the component system that should be kept.
    return tensor(rho_AB.ptrace(0), thermal_dm(N, Nth)) # rho_A is kept here.


# wrong
# def RHO_1(state, N, s, Nth, eta, rt_list=False):
#     """ State obtained if the object is present.
#         N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
#         s: a complex number. 's' is the squeezing parameter.
#         r: complex numbers.
#         return: a qutip.Qobj(), density matrix
#     """
#     theta = np.arctan(np.sqrt((1 - eta)/eta))
#     if not rt_list:
#         rho_AB = ket2dm(state(N, s))
#     else:
#         rho_AB = ket2dm(state(N, s, rt_list))

#     # tensor product of state AB and thermal state
#     rho = tensor(rho_AB, thermal_dm(N, Nth/(1-eta)))
    
#     # state A unchanged, tm_mixing acted on state B and thermal
#     op = tensor(qeye(N), tm_mixing(N, 1j * theta))

#     rho_1 = op * rho * op.dag()
#     print "C kept"
#     return rho_1.ptrace([0, 2])


def RHO_1(state, N, l, Nth, kappa, rt_list=False):
    """ State obtained if the object is present.
        N: a positive interger. Photon number is truncated at N, i.e (0, N-1)
        s: a complex number. 's' is the squeezing parameter.
        r: complex numbers.
        return: a qutip.Qobj(), density matrix
    """
    if not rt_list:
        rho_AB = ket2dm(state(N, l))
    else:
        rho_AB = ket2dm(state(N, l, rt_list))

    # tensor product of state AB and thermal state
    rho = tensor(rho_AB, thermal_dm(N, Nth/(1-kappa)))
    
    # state A unchanged, tm_mixing acted on state B and thermal
    theta = np.arccos(np.sqrt(kappa))
    op = tensor(qeye(N), tm_mixing(N, theta))

    rho_1 = op * rho * op.dag()
    return rho_1.ptrace([0, 1])


# Helstrom, upper bound (QCB) and lower bound ========================


def Helstrom(pi_0, rho_0, pi_1, rho_1, M=1):
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
        pass
    return 0.5 * (1 - gamma.norm())
        
    

def QCB(rho_0, rho_1, approx=False):
    """ Approximated Q for QCB
        Actually the trace of sqrt(rho_1) * sqrt(rho_2)
    """
    if approx:
        # s = 0.5
        return (rho_0.sqrtm() * rho_1.sqrtm()).tr().real
    else:
        # give the optimal QCB by varying value of s
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