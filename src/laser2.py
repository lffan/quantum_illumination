# -*- coding: utf-8 -*-
# Created on 2017-07-18
# Modified on 2017-07-18

""" two-mode laser """

__author__ = 'Longfei Fan'


from qutip import *
import numpy as np


class LaserTwoMode(object):
    """A class for two-mode lasers"""
    
    def __init__(state, l, n_max, rt=False):
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
        if n > 0 and state:
            if state == 'TMSS':
                self.state = self.__TMSS(l, n_max)
            elif state == 'PS':
                self.state = self.__PS(l, n_max)
            elif state == 'PA':
                self.state = self.__PA(l, n_max)
            elif state == 'PSA':
                self.state = self.__PSA(l, n_max)
            elif state == 'PAS':
                self.state = self.__PAS(l, n_max)
            elif state == 'PCS':
                self.state = self.__PCS(l, n_max, rt)
            else:
                raise ValueError('{} is not a valid name of two mode state!'.format())
        
    
    def __TMSS(l, n_max):
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
        return state.unit()
    
    
    def __PS(l, n_max):
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
        return state.unit()
    
    
    def __PA(l, n_max):
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
        return state.unit()
    
    def __PSA(N, l):
        """Photon added then subtracted state"""
        state = np.sum([(n+1)**2 * l**n * tensor(basis(n_max, n), basis(n_max, n)) \
                        for n in range(n_max)])
        return state.unit()


    def __PAS(N, l):
        """Photon subtracted then added state"""
        state = np.sum([(n+1)**2 * l**n * tensor(basis(n_max, n + 1), basis(n_max, n + 1)) \
                        for n in rrange(n_max - 1)])
        return state.unit()
    

    def __PCS(N, l, rt_list):
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
        state = ta * tb * state1 + ta * rb * state2 + \
                ra * tb * state3 + ra * rb * state4

        return state.unit()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            