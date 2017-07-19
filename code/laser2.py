# -*- coding: utf-8 -*-
# Created on 2017-07-18
# Modified on 2017-07-18

""" two-mode laser """

__author__ = 'Longfei Fan'


from qutip import *
import numpy as np


class LaserTwoMode(object):
    """A class for two-mode lasers"""
    
    def __init__(self, state, l, n_max, rt=False):
        """
        To initialize a two-mode state.
        
        Parameters
        ----------
        state: string
            can only be one of 'TMSS', 'PS', 'PA', 'PAS', 'PSA', and 'PCS'
            used to specify what kind of two-mode state will be created
        l: double float
            state parameter, for TMSS l = tanh(s), where s is the squeezed para
        n_max: positive int
            photon truncation number as we are doing numerical calculation
            i.e. photon numbers can be in [0, n_max - 1]
        rt: int list, optional
            superposition factors used when state == "PCS"
            
        Raise 
        -----
        ValueError:
            when paras are not valid
        """
        if n_max > 0:
            if state == 'TMSS':
                self.state, self.aver_n = self.__TMSS(l, n_max)
            elif state == 'PS':
                self.state, self.aver_n = self.__PS(l, n_max)
            elif state == 'PA':
                self.state, self.aver_n = self.__PA(l, n_max)
            elif state == 'PSA':
                self.state, self.aver_n = self.__PSA(l, n_max)
            elif state == 'PAS':
                self.state, self.aver_n = self.__PAS(l, n_max)
            elif state == 'PCS':
                self.state, self.aver_n = self.__PCS(l, n_max, rt)
            else:
                raise ValueError('\'{}\' is not a valid name of two mode state!'.format(state))
        else:
            raise ValueError('n_max must be larger than zero!')
        
    
    def __TMSS(self, l, n_max):
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
        state = np.sum([l**n * tensor(basis(n_max, n), basis(n_max, n)) \
                        for n in range(n_max)])
        aver_n = l**2 / (1 - l**2)
        return state.unit(), aver_n
    
    
    def __PS(self, l, n_max):
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
        state = np.sum([(n+1) * l**n * tensor(basis(n_max, n), basis(n_max, n)) \
                        for n in range(n_max)])
        aver_n = 2*l**2 * (2 + l**2) / (1 - l**4)
        return state.unit(), aver_n
    
    
    def __PA(self, l, n_max):
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
        state = np.sum([(n+1) * l**n * tensor(basis(n_max, n+1), basis(n_max, n+1)) \
                        for n in range(n_max - 1)])
        aver_n = (1 + 4*l**2 + l**4) / (1 - l**4)
        return state.unit(), aver_n
    
    def __PSA(self, l, n_max):
        """Photon added then subtracted state"""
        state = np.sum([(n+1)**2 * l**n * tensor(basis(n_max, n), basis(n_max, n)) \
                        for n in range(n_max)])
        aver_n = 2*l**2 * (8 + 33*l**2 + 18*l**4 + l**6) / (1 + 10*l**2 - 10*l**6 + l**8)
        return state.unit(), aver_n


    def __PAS(self, l, n_max):
        """Photon subtracted then added state"""
        state = np.sum([(n+1)**2 * l**n * tensor(basis(n_max, n + 1), basis(n_max, n + 1)) \
                        for n in range(n_max - 1)])
        aver_n = (1 + 26*l**2 + 66*l**4 + 26*l**6 + l**8) / (1 + 10*l**2 - 10*l**6 - l**8)
        return state.unit(), aver_n
    

    def __PCS(self, l, n_max, rt):
        """
        States obtained by a coherent superposition operation of photon subtraction 
        and addition on two-mode squeezed state
        
        Parameters
        ----------
        rt_list: a list of int
            define the superposition factor
        """
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
        state = (ta * tb * state1 + ta * rb * state2 + \
                ra * tb * state3 + ra * rb * state4).unit()
        aver_n = expect(num(n_max), state.ptrace(0))

        return state, aver_n
    
    
    def get_state(self):
        return self.state
    
    
    def get_aver_n(self):
        return self.aver_n
    
    
    def vn_entropy(self):
        return entropy_vn(self.state.ptrace(0))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            