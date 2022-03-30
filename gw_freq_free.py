#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Sylvia Bintrim <sjb2225@columbia.edu>
#

'''
G0W0 approximation with dTDA screening based on iterative diagonalization
(spatial orbitals)
'''

from functools import reduce
import numpy
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import scf
from pyscf import dft 
from pyscf import df
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__
import time

einsum = lib.einsum

start_time = time.time()

def kernel(gw, nroots=1, koopmans=True, guess=None, eris=None, **kwargs):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(gw.stdout, gw.verbose)
    if gw.verbose >= logger.WARN:
        gw.check_sanity()
    gw.dump_flags()

    if eris is None:
        eris = gw.ao2mo()

    # GHF or customized RHF/UHF may be of complex type
    real_system = (gw._scf.mo_coeff[0].dtype == np.double)

    if gw.diagonal is False: 
        matvec, diag = gw.gen_matvec(eris=eris)

        size = gw.vector_size()
        nroots = min(nroots, size)
        if guess is not None:
            user_guess = True
            for g in guess:
                assert g.size == size
        else:
            user_guess = False
            guess = gw.get_init_guess(nroots=nroots, koopmans=koopmans, diag=diag)

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = lib.davidson_nosym1
        if user_guess or koopmans:
            assert len(guess) == nroots
            def eig_close_to_init_guess(w, v, nroots, envs):
                x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
                s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
                snorm = np.einsum('pi,pi->i', s.conj(), s)
                idx = np.argsort(-snorm)[:nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)
            conv, es, vs = eig(matvec, guess, precond, pick=eig_close_to_init_guess,
                               tol=gw.conv_tol, max_cycle=gw.max_cycle,
                               max_space=gw.max_space, nroots=nroots, verbose=log)
        else:
            def pickeig(w, v, nroots, envs):
                real_idx = np.where(abs(w.imag) < 1e-3)[0]
                return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
            conv, es, vs = eig(matvec, guess, precond, pick=pickeig,
                               tol=gw.conv_tol, max_cycle=gw.max_cycle,
                               max_space=gw.max_space, nroots=nroots, verbose=log)

    else:
        conv = list()
        es = list()
        vs = list()
        size = gw.vector_size()
        nroots_ = 1
        for diag_idx in gw.diag_idxs:
            matvec, diag = gw.gen_matvec(eris=eris, diag_idx=diag_idx)
            guess = gw.get_init_guess(nroots=1, diag=diag, diag_idx=diag_idx)

            def precond(r, e0, x0):
                return r/(e0-diag+1e-12)

            eig = lib.davidson_nosym1
            def eig_close_to_init_guess(w, v, nroots_, envs):
                x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
                s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
                snorm = np.einsum('pi,pi->i', s.conj(), s)
                idx = np.argsort(-snorm)[:nroots_]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)
            convn, en, vn = eig(matvec, guess, precond, pick=eig_close_to_init_guess,
                                tol=gw.conv_tol, max_cycle=gw.max_cycle,
                                max_space=gw.max_space, nroots=nroots_, verbose=log)
            #def pickeig(w, v, nroots_, envs):
            #    real_idx = np.where(abs(w.imag) < 1e-3)[0]
            #    return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
            #convn, en, vn = eig(matvec, guess, precond, pick=pickeig,
            #                    tol=gw.conv_tol, max_cycle=gw.max_cycle,
            #                    max_space=gw.max_space, nroots=nroots_, verbose=log)

            conv.append(convn[0])
            es.append(en[0])
            vs.append(vn[0])
        conv = np.asarray(conv)
        es = np.asarray(es)
        vs = np.asarray(vs)

    if gw.verbose >= logger.INFO:
        for n, en, vn, convn in zip(range(nroots), es, vs, conv):
            r1ip, r1ea, r2ip, r2ea = gw.vector_to_amplitudes(vn)
            if isinstance(r1ip, np.ndarray):
                qp_weight = np.linalg.norm(r1ip)**2 + np.linalg.norm(r1ea)**2
            else: # for UGW 
                r1ip = np.hstack([x.ravel() for x in r1ip])
                r1ea = np.hstack([x.ravel() for x in r1ea])
                qp_weight = np.linalg.norm(r1ip)**2 + np.linalg.norm(r1ea)**2
            logger.info(gw, 'GW root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, qp_weight, convn)
        log.timer('GW', *cput0)
    if nroots == 1:
        return conv[0], es[0].real, vs[0]
    else:
        return conv, es.real, vs

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    r1ip = vector[:nocc].copy()
    r1ea = vector[nocc:nmo].copy()
    r2ip = vector[nmo:nmo+nocc**2*nvir].copy().reshape(nocc,nocc,nvir)
    r2ea = vector[nmo+nocc**2*nvir:].copy().reshape(nocc,nvir,nvir)
    return r1ip, r1ea, r2ip, r2ea


def amplitudes_to_vector(r1ip, r1ea, r2ip, r2ea):
    vector = np.hstack((r1ip, r1ea, r2ip.ravel(),r2ea.ravel()))
    return vector

def matvec_diag(gw, vector, diag_idx, eris=None):
    '''matrix-vector multiplication in the diagonal approximation'''
    if eris is None: eris = gw.ao2mo()
    nocc = gw.nocc
    nmo = gw.nmo

    r1ip, r1ea, r2ip, r2ea = vector_to_amplitudes(vector, nmo, nocc)

    mo_e_occ = eris.mo_energy[:nocc]
    mo_e_vir = eris.mo_energy[nocc:]
    vk_occ = eris.vk[:nocc,:nocc]
    vxc_occ = eris.vxc[:nocc,:nocc]
    vk_vir = eris.vk[nocc:,nocc:]
    vxc_vir = eris.vxc[nocc:,nocc:]

    if diag_idx < nocc:
        occ_idx = diag_idx

        Hr1ip  = einsum('i,i->i', mo_e_occ, r1ip) 
        Hr1ip -= einsum('ii,i->i', vk_occ, r1ip) 
        Hr1ip -= einsum('ii,i->i', vxc_occ, r1ip) 
        Hr1ip[occ_idx] += 2*einsum('klc,klc', eris.ooov[:,:,occ_idx,:], r2ip)
        Hr1ip[occ_idx] += 2*einsum('kcd,kcd', eris.oovv[:,occ_idx,:,:].conj(), r2ea)

        Hr1ea = np.zeros_like(r1ea) 
        
        Hr2ip = eris.ooov[:,:,occ_idx,:].conj()*r1ip[occ_idx]
        
        Hr2ea = eris.oovv[:,occ_idx,:,:]*r1ip[occ_idx]

    else:
        vir_idx = diag_idx - nocc

        Hr1ip = np.zeros_like(r1ip)

        Hr1ea = einsum('a,a->a', mo_e_vir, r1ea)
        Hr1ea -= einsum('aa,a->a', vk_vir, r1ea)
        Hr1ea -= einsum('aa,a->a', vxc_vir, r1ea)
        Hr1ea[vir_idx] += 2*einsum('klc,klc', eris.oovv[:,:,vir_idx,:], r2ip)
        Hr1ea[vir_idx] += 2*einsum('kcd,kcd', eris.ovvv[:,vir_idx,:,:].conj(), r2ea)
        
        Hr2ip = eris.oovv[:,:,vir_idx,:].conj()*r1ea[vir_idx]
        
        Hr2ea = eris.ovvv[:,vir_idx,:,:]*r1ea[vir_idx]

    Hr2ip += einsum('i,ija->ija', mo_e_occ, r2ip)
    Hr2ip += einsum('j,ija->ija', mo_e_occ, r2ip)
    Hr2ip -= einsum('a,ija->ija', mo_e_vir, r2ip)
    #Hr2ip -= 2*einsum('jcal,ilc->ija', eris.ovvo, r2ip)
    Hr2ip -= 2*einsum('Qja, Qlc,ilc->ija', eris.Lov, eris.Lov, r2ip)

    Hr2ea += einsum('a,iab->iab', mo_e_vir, r2ea)
    Hr2ea += einsum('b,iab->iab', mo_e_vir, r2ea)
    Hr2ea -= einsum('i,iab->iab', mo_e_occ, r2ea)
    #Hr2ea += 2*einsum('icak,kcb->iab', eris.ovvo, r2ea)
    Hr2ea += 2*einsum('Qia,Qjc,jcb->iab', eris.Lov, eris.Lov, r2ea)

    vector = amplitudes_to_vector(Hr1ip, Hr1ea, Hr2ip, Hr2ea)
    return vector

def matvec(gw, vector, eris=None):
    if eris is None: eris = gw.ao2mo()
    nocc = gw.nocc
    nmo = gw.nmo

    r1ip, r1ea, r2ip, r2ea = vector_to_amplitudes(vector, nmo, nocc)    
    
    mo_e_occ = eris.mo_energy[:nocc]
    mo_e_vir = eris.mo_energy[nocc:]
    vk_oo = eris.vk[:nocc,:nocc]
    vk_ov = eris.vk[:nocc,nocc:]
    vk_vv = eris.vk[nocc:,nocc:]
    vxc_oo = eris.vxc[:nocc,:nocc]
    vxc_ov = eris.vxc[:nocc,nocc:]
    vxc_vv = eris.vxc[nocc:,nocc:]

    # r1-r1 update is the full fock matrix in the chosen mean-field basis
    Hr1ip  = einsum('i,i->i', mo_e_occ, r1ip) 
    Hr1ip -= einsum('ij,j->i', vk_oo, r1ip) 
    Hr1ip -= einsum('ij,j->i', vxc_oo, r1ip) 
    Hr1ip -= einsum('ib,b->i', vk_ov, r1ea) 
    Hr1ip -= einsum('ib,b->i', vxc_ov, r1ea) 
    Hr1ip += 2*einsum('klic,klc->i', eris.ooov, r2ip)
    Hr1ip += 2*einsum('kicd,kcd->i', eris.oovv.conj(), r2ea)

    Hr1ea = einsum('a,a->a', mo_e_vir, r1ea)
    Hr1ea -= einsum('ab,b->a', vk_vv, r1ea)
    Hr1ea -= einsum('ab,b->a', vxc_vv, r1ea)
    Hr1ea -= einsum('ib,i->b', vk_ov, r1ip)
    Hr1ea -= einsum('ib,i->b', vxc_ov, r1ip)
    Hr1ea += 2*einsum('klac,klc->a', eris.oovv, r2ip)
    Hr1ea += 2*einsum('kacd,kcd->a', eris.ovvv.conj(), r2ea)
    
    Hr2ip = einsum('ijka,k->ija', eris.ooov.conj(), r1ip)
    Hr2ip += einsum('ijba,b->ija', eris.oovv.conj(), r1ea)
    Hr2ip += einsum('i,ija->ija', mo_e_occ, r2ip)
    Hr2ip += einsum('j,ija->ija', mo_e_occ, r2ip)
    Hr2ip -= einsum('a,ija->ija', mo_e_vir, r2ip)
    #Hr2ip -= 2*einsum('jcal,ilc->ija', eris.ovvo, r2ip)
    Hr2ip -= 2*einsum('Qja, Qlc,ilc->ija', eris.Lov, eris.Lov, r2ip)
    
    Hr2ea = einsum('icab,c->iab', eris.ovvv, r1ea)
    Hr2ea += einsum('ijab,j->iab', eris.oovv, r1ip)
    Hr2ea += einsum('a,iab->iab', mo_e_vir, r2ea)
    Hr2ea += einsum('b,iab->iab', mo_e_vir, r2ea)
    Hr2ea -= einsum('i,iab->iab', mo_e_occ, r2ea)
    #Hr2ea += 2*einsum('icak,kcb->iab', eris.ovvo, r2ea)
    Hr2ea += 2*einsum('Qia,Qjc,jcb->iab', eris.Lov, eris.Lov, r2ea)

    vector = amplitudes_to_vector(Hr1ip, Hr1ea, Hr2ip, Hr2ea)
    return vector


class GW(lib.StreamObject):
    '''generalized (spin-orbital) G0W0 with dTDA screening
    Attributes:
        diagonal : bool
            Whether to use the diagonal approximation to the Green's function and 
            self-energy.  Default is False.
    Saved results:
        converged : bool
            Whether GW roots are converged or not
        e : float
            GW eigenvalues (IP or EA)
        v : float
            GW eigenvectors in 1p+1h+2p1h+2h1p space
    '''
    def __init__(self, mf, frozen=None, diagonal=False, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, scf.ghf.GHF) or isinstance(mf, dft.gks.GKS) or isinstance(mf, dft.rks.RKS) or isinstance(mf,scf.hf.SCF))
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', 1e-7)

        self.frozen = frozen
        self.diagonal = diagonal
##################################################
# don't modify the following attributes, they are not input options
        self.converged = False
        self.e = None
        self.v = None
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            logger.info(self, 'frozen orbitals %s', self.frozen)
        logger.info(self, 'diagonal approx = %d', self.diagonal)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def kernel(self, nroots=1, koopmans=True, guess=None, eris=None):
        if eris is None: 
            eris = self.ao2mo(self.mo_coeff)
        self.converged, self.e, self.v = \
                kernel(self, nroots, koopmans, guess, eris)
        return self.e, self.v

    matvec = matvec
    matvec_diag = matvec_diag

    def gen_matvec(self, eris=None, diag_idx=None):
        # diag_idx triggers the diagonal approximation
        if eris is None:
            eris = self.ao2mo()
        diag = self.get_diag()
        if diag_idx is None:
            matvec = lambda xs: [self.matvec(x, eris=eris) for x in xs]
        else:
            matvec = lambda xs: [self.matvec_diag(x, diag_idx, eris=eris) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1ip, r1ea, r2ip, r2ea):
        return amplitudes_to_vector(r1ip, r1ea, r2ip, r2ea)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nvir + nocc**2*nvir + nocc*nvir**2

    def get_diag(self, eris=None):
        if eris is None:
            eris = self.ao2mo()
        nocc = self.nocc
        nvir = self.nmo - nocc
        mo_e_occ = eris.mo_energy[:nocc]
        mo_e_vir = eris.mo_energy[nocc:]
        Hr1ip = mo_e_occ + np.diag(-eris.vk - eris.vxc)[:nocc]
        Hr1ea = mo_e_vir + np.diag(-eris.vk - eris.vxc)[nocc:]
        Hr2ip = np.zeros((nocc,nocc,nvir))
        Hr2ea = np.zeros((nocc,nvir,nvir))
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    Hr2ip[i,j,a] = mo_e_occ[i] + mo_e_occ[j] - mo_e_vir[a]
        for i in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    Hr2ea[i,a,b] = mo_e_vir[a] + mo_e_vir[b] - mo_e_occ[i]

        return self.amplitudes_to_vector(Hr1ip, Hr1ea, Hr2ip, Hr2ea)

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        mem_incore = 2*(nocc**3*nvir+2*nocc**2*nvir**2+nocc*nvir**3) * 8/1e6
        mem_now = lib.current_memory()[0]
        self.mol.incore_anyway = True
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            raise NotImplementedError 


class GWIP(GW):

    def kernel(self, nroots=1, koopmans=True, guess=None, eris=None):
        if self.diagonal:
            self.diag_idxs = np.arange(self.nocc-nroots,self.nocc)
        return GW.kernel(self, nroots, koopmans, guess, eris)

    def get_init_guess(self, nroots=1, koopmans=True, diag=None, diag_idx=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if diag_idx is not None:
            g = np.zeros(int(size), dtype)
            g[diag_idx] = 1.0
            guess.append(g)
        elif koopmans:
            for n in range(nroots):
                g = np.zeros(int(size), dtype)
                g[self.nocc-n-1] = 1.0
                guess.append(g)
        else:
            d1ip, d1ea, d2ip, d2ea = self.vector_to_amplitudes(diag)
            idx = self.amplitudes_to_vector(-d1ip, -d1ea+1e9, -d2ip, -d2ea+1e9).argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype)
                g[i] = 1.0
                guess.append(g)
        return guess


class GWEA(GW):

    def kernel(self, nroots=1, koopmans=True, guess=None, eris=None):
        if self.diagonal:
            self.diag_idxs = np.arange(self.nocc,self.nocc+nroots)
        return GW.kernel(self, nroots, koopmans, guess, eris)

    def get_init_guess(self, nroots=1, koopmans=True, diag=None, diag_idx=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if diag_idx is not None:
            g = np.zeros(int(size), dtype)
            g[diag_idx] = 1.0
            guess.append(g)
        elif koopmans:
            for n in range(nroots):
                g = np.zeros(int(size), dtype)
                g[self.nocc+n] = 1.0
                guess.append(g)
        else:
            d1ip, d1ea, d2ip, d2ea = self.vector_to_amplitudes(diag)
            idx = self.amplitudes_to_vector(d1ip+1e9, d1ea, d2ip+1e9, d2ea).argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype)
                g[i] = 1.0
                guess.append(g)
        return guess


class _PhysicistsERIs:
    '''<pq|rs> not antisymmetrized
    
    This is gccsd _PhysicistsERIs without vvvv and without antisym'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.e_hf = None
        self.orbspin = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None

    def _common_init_(self, mygw, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mygw.mo_coeff
        mo_idx = mygw.get_frozen_mask()
        # if getattr(mo_coeff, 'orbspin', None) is not None:
        #     self.orbspin = mo_coeff.orbspin[mo_idx]
        #     mo_coeff = lib.tag_array(mo_coeff[:,mo_idx], orbspin=self.orbspin)
        # else:
        #     orbspin = scf.ghf.guess_orbspin(mo_coeff)
        #     mo_coeff = mo_coeff[:,mo_idx]
        #     if not np.any(orbspin == -1):
        #         self.orbspin = orbspin[mo_idx]
        #         mo_coeff = lib.tag_array(mo_coeff, orbspin=self.orbspin)
        self.mo_coeff = mo_coeff

        self.mo_energy = mygw._scf.mo_energy
        dm = mygw._scf.make_rdm1(mygw.mo_coeff, mygw.mo_occ)
        vj, vk = mygw._scf.get_jk(mygw.mol, dm) 
        vj = reduce(np.dot, (mo_coeff.conj().T, vj, mo_coeff))
        self.vk = 0.5*reduce(np.dot, (mo_coeff.conj().T, vk, mo_coeff))
        vxc = mygw._scf.get_veff(mygw.mol, dm)
        self.vxc = reduce(np.dot, (mo_coeff.conj().T, vxc, mo_coeff)) - vj
        # Note: Recomputed fock matrix since SCF may not be fully converged.
        #dm = mygw._scf.make_rdm1(mygw.mo_coeff, mygw.mo_occ)
        #vhf = mygw._scf.get_veff(mygw.mol, dm)
        #fockao = mygw._scf.get_fock(vhf=vhf, dm=dm)
        #self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        #self.e_hf = mygw._scf.energy_tot(dm=dm, vhf=vhf)
        self.nocc = mygw.nocc
        self.mol = mygw.mol

        mo_e = self.mo_energy
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mygw, 'HOMO-LUMO gap %s too small for GCCSD', gap)
        return self
        
#from RCCSD.
def _make_eris_incore(mygw, mo_coeff=None, ao2mofn=None):
    cput0 = (time.clock(), time.time())
    eris = _PhysicistsERIs()
    eris._common_init_(mygw, mo_coeff)
    nocc = mygw.nocc
    nmo = mygw.nmo
    #nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    co = np.asarray(mo_coeff[:,:nocc], order = 'F')
    cv = np.asarray(mo_coeff[:,nocc:], order = 'F')

    if callable(ao2mofn):
        # eri1 = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
        eris.ooov = ao2mofn((co, co, co, cv)).reshape(nocc, nocc, nocc, nvir)
        eris.oovv = ao2mofn((co, co, cv, cv)).reshape(nocc, nocc, nvir, nvir)
        eris.ovvo = ao2mofn((co, cv, cv, co)).reshape(nocc, nvir, nvir, nocc)
        eris.ovvv = ao2mofn((co, cv, cv, cv)).reshape(nocc, nvir, nvir, nvir)
    else:
        eris.ooov = ao2mo.general(mygw._scf._eri, (co, co, co, cv), compact=False).reshape(nocc,nocc,nocc,nvir).transpose(0,2,1,3)
        eris.oovv = ao2mo.general(mygw._scf._eri, (co, cv, co, cv), compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        eris.ovvo = ao2mo.general(mygw._scf._eri, (co, cv, cv, co), compact=False).reshape(nocc,nvir,nvir,nocc).transpose(0,2,1,3)
        eris.ovvv = ao2mo.general(mygw._scf._eri, (co, cv, cv, cv), compact=False).reshape(nocc,nvir,nvir,nvir).transpose(0,2,1,3)
    #     eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    #     eri1 = ao2mo.restore(1, eri1, nmo)
    
    # eri1 = eri1.transpose(0,2,1,3)
    # eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    # eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    # eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    # eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    # eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    # eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    # eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    # eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
    logger.timer(mygw, 'CCSD integral transformation', *cput0)
    
    #from dfccsd
    def _init_df_eris():
        with_df = df.DF(mf.mol, auxbasis='def2-SVP-JKFIT' )
        naux = with_df.get_naoaux()        
        Loo = numpy.empty((naux,nocc,nocc))
        Lov = numpy.empty((naux,nocc,nvir))
        Lvv = numpy.empty((naux, nvir, nvir))
        mo = numpy.asarray(eris.mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        p1 = 0
        Lpq = None
        for k, eri1 in enumerate(with_df.loop()):
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1-p0,nmo,nmo)
            Loo[p0:p1] = Lpq[:,:nocc,:nocc]
            Lov[p0:p1] = Lpq[:,:nocc,nocc:]
            Lvv[p0:p1] = Lpq[:, nocc:, nocc:]
        Lpq = None
        #Lvo = Lov.transpose(0,2,1).reshape(naux,nvir*nocc)
        Lov = Lov.reshape(naux,nocc*nvir)
        Loo = Loo.reshape(naux, nocc*nocc)
        Lvv = Lvv.reshape(naux, nvir*nvir)
        
        Lov = Lov.reshape(naux,nocc,nvir)
        #Lvo = Lvo.reshape(naux,nvir,nocc)
        Loo = Loo.reshape(naux, nocc, nocc)
        Lvv = Lvv.reshape(naux, nvir, nvir)
        return Lov, Loo, Lvv
    eris.Lov, eris.Loo, eris.Lvv = _init_df_eris()
    
    return eris


if __name__ == '__main__':
    from pyscf import gto
    
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [['C',(0.000000, 0.000000, 0.000000)],
            ['C', (0.000000, 0.000000, 1.515000)],
            ['O', (1.001953, 0.000000, 2.193373)],
            ['H', (-1.019805, 0.000000, 1.997060)],
            ['H', (-0.905700, -0.522900, -0.363000)],
            ['H', (0.000000, 1.045800, -0.363000)],
            ['H', (0.905700, -0.522900, -0.363000)]]
    mol.basis = 'def2-SVP'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    gw = GWIP(mf, diagonal=True)
    eip,v = gw.kernel(nroots=2)
    gw = GWEA(mf, diagonal=True)
    eea,v = gw.kernel(nroots=2)
    print('freq-free GW IP (eV) =', eip*27.21139)
    print('freq-free GW EA (eV) =', eea*27.21139)
    
    #Original PySCF GW (with frequency)
    from pyscf import gw
    from pyscf import tddft
    # IP and EA
    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo - nocc
    
    orbs = [nocc-2, nocc-1, nocc, nocc+1]

    td = tddft.dTDA(mf)
    td.nstates = nocc*nvir
    e, xy = td.kernel()
    # Make a fake Y vector of zeros
    td_xy = list()
    for e,xy in zip(td.e,td.xy):
        x,y = xy
        td_xy.append((x,0*x))
    td.xy = td_xy

    mygw = gw.GW(mf, freq_int='exact', tdmf=td)
    mygw.kernel(orbs=orbs)
    eip1 = mygw.mo_energy[nocc-2:nocc]
    eea1 = mygw.mo_energy[nocc:nocc+2]
    print('original GW IP (eV) =', eip1*27.21139)
    print('original GW EA (eV) =', eea1*27.21139)
    
    print('freq-free - original IP (eV) = ', (eip-eip1)*27.21139)
    print('freq-free - original EA (eV) = ', (eea-eea1)*27.21139)