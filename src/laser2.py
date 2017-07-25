# -*- coding: utf-8 -*-
# Created on 2017-07-18
# Last Modified on 2017-07-18
# Latest Modified on 2017-07-24

""" two-mode laser, TMSS and non-Gaussian states """

__author__ = 'Longfei Fan'


from qutip import *
import numpy as np


class LaserTwoMode(object):
    """A class for two-mode lasers"""
    
    def __init__(state_name, n_max):
        """
        To initialize a two-mode state.
        
        Parameters
        ----------
        state: string
            can only be one of 'TMSS', 'PS', 'PA', 'PAS', 'PSA', and 'PCS'
            used to specify what kind of two-mode state will be created
            
        Raise 
        -----
        ValueError:
            when paras are not valid
        """
        if n > 0 and state_name in ('TMSS', 'PS', 'PA', 'PSA', 'PAS', 'PCS'):
            self.name = state_name
            self.state = None
        else:
            raise ValueError('{} is not a valid name of two mode state!'.format())
    
    def __tm_sqz(self, s):
        """
        Two mode mixing operator in the form of matrix with Fock basis
        
        Parameters
        ----------
        s: a complex number
            The squeezed parameter
        
        Return
        ------
        qubit.Qobj()
        """
        a = destroy(N)
        tms = - np.conj(s) * tensor(a, a) + s * tensor(a.dag(), a.dag())
        return tms.expm()
    
    def __tm_mix(self, s)
        """
        Two-mode mixing operator (Beam splitter)
        
        Parameters
        ----------
        s: a complex number
            The mixing parameter
        
        Return
        ------
        qubit.Qobj()
        """
        a = destroy(N)
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
    def __init__(name, l, n_max):
        super.__init__(name, n_max)
        self.state = np.sum([l**n * tensor(basis(n_max, n), basis(n_max, n)) 
                             for n in range(n_max)])
    
    
class PS(LaserTwoMode):
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
    def __init__(name, l, n_max):
        super().__init__(name, n_max)
        self.state = np.sum([(n+1) * l**n * tensor(basis(n_max, n), basis(n_max, n)) 
                        for n in range(n_max)])
    
class PA(LaserTwoMode):
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
    def __init__(name, l, n_max):
        super().__init__(name, n_max)
        self.state = np.sum([(n+1) * l**n * tensor(basis(n_max, n+1), basis(n_max, n+1)) 
                             for n in range(n_max - 1)])
    
class PSA(LaserTwoMode):
    """Photon added then subtracted state"""
    def __init__(name, l, n_max):
        super().__init__(name, n_max)
        self.state = np.sum([(n+1)**2 * l**n * tensor(basis(n_max, n), basis(n_max, n)) 
                             for n in range(n_max)])

class PAS(LaserTwoMode):
    """Photon subtracted then added state"""
    def __init__(name, l, n_max):
        super().__init(name, n_max)
        self.state = np.sum([(n+1)**2 * l**n * tensor(basis(n_max, n + 1), basis(n_max, n + 1)) 
                             for n in rrange(n_max - 1)])
    

class PCS(LaserTwoMode):
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
    rt_list: a list of int
        define the superposition factor
    """
    def __init__(name, l, n_max, rt):
        super().__init__(name, n_max)
        self.state = self.__create_state(l, n_max, rt)

            
    def __create_state(l, n_max, rt):
                ta, ra, tb, rb = rt
        nlist = range(n_max)
        state1 = np.sum([l**(n+1) * (n+1) * tensor(basis(n_max, n), basis(n_max, n)) \
                        for n in nlist])
        state2 = np.sum([l**(n+1) * np.sqrt((n+1)*(n+2)) \
                        * tensor(basis(n_max, n), basis(n_max, n + 2)) for n in nlist[:-2]])
        state3 = np.sum([l**(n+1) * np.sqrt((n+1)*(n+2)) \
                        * tensor(basis(n_max, n + 2), basis(n_max, n)) for n in nlist[:-2]])
        state4 = np.sum([l**n * (n+1) * tensor(basis(n_max, n + 1), basis(n_max, n + 1)) \
                        for n in nlist[:-1]])
        self.state = ta * tb * state1 + ta * rb * state2 + \
                ra * tb * state3 + ra * rb * state4