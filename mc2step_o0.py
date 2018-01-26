#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

from functools import reduce
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf import ao2mo
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf import mcscf
#import mc_o0

# ref. JCP, 82, 5053;  JCP, 73, 2342
# Eq. (C6) of JCP, 73, 2342 is wrong.  For spin-free 2rdm, [ij,kl] -> [kj,il]
# involves the change of alpha and beta spins for bra functions, of which the
# ERIs are zero: (ij|kl)[i_alpha k_beta, l_beta j_alpha] != 0
# but (kj|il)[k_beta i_alpha, l_beta, j_alpha] == 0 so that
# [k_beta i_alpha, l_beta, j_alpha] == 0 => [ij,kl] != -[kj,il]

def inter_grad_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2):
    ncas = mo_cas.shape[1]
    ncore = mo_core.shape[1]
    nmocc = ncas + ncore
    nao,nmo = mo.shape
    dm1 = numpy.zeros((nmo,nmo))
    for i in range(ncore):
        dm1[i,i] = 2
    dm1[ncore:nmocc,ncore:nmocc] = rdm1
    dm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    dm2[ncore:nmocc,ncore:nmocc,ncore:nmocc,ncore:nmocc] = rdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] += -2
        dm2[i,i,ncore:nmocc,ncore:nmocc] = dm2[ncore:nmocc,ncore:nmocc,i,i] = 2*rdm1
        dm2[i,ncore:nmocc,ncore:nmocc,i] = dm2[ncore:nmocc,i,i,ncore:nmocc] = -rdm1

#    dm2 = (dm2 + dm2.transpose(1,0,2,3)) * .5
#    for i in range(nmocc):
#        for j in range(j+1):
#            for k in range(nmocc):
#                for l in range(k+1):
#                    assert(abs(dm2[i,j,k,l]-dm2[j,i,k,l]) < 1e-14)
#                    assert(abs(dm2[i,j,k,l]-dm2[i,j,l,k]) < 1e-14)
#                    assert(abs(dm2[i,j,k,l]-dm2[k,l,i,j]) < 1e-14)

    h1e_mo = reduce(numpy.dot, (mo.T, mf.get_hcore(mol), mo))
    jeri = ao2mo.incore.general(mf._eri, (mo, mo, mo, mo), compact=False)
    jeri = jeri.reshape(nmo,nmo,nmo,nmo)

    g = numpy.dot(h1e_mo, dm1) \
            + numpy.dot(jeri.reshape(nmo,-1), dm2.reshape(nmo,-1).T)
    h = numpy.zeros((nmo,nmo,nmo,nmo))
    for i in range(nmo):
        for j in range(nmo):
            v = numpy.dot(jeri[i,j,:,:].flatten(), dm2.reshape(nmo*nmo,-1)) \
              + numpy.dot(jeri[i,:,:,j].flatten(), \
                          dm2.transpose(1,2,0,3).reshape(nmo*nmo,-1))
            v += numpy.dot(jeri[i,:,:,j].flatten(), \
                           dm2.transpose(1,3,0,2).reshape(nmo*nmo,-1))
            h[i,:,j,:] += dm1 * h1e_mo[i,j] + v.reshape(nmo,nmo)
        h[:,i,i,:] += g

    numpy.set_printoptions(5)
    return g, h

def _inter_grad_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2):
    ncas = mo_cas.shape[1]
    ncore = mo_core.shape[1]
    nmocc = ncas + ncore
    nao,nmo = mo.shape
    dm1 = numpy.zeros((nmo,nmo))
    for i in range(ncore):
        dm1[i,i] = 2
    dm1[ncore:nmocc,ncore:nmocc] = rdm1
    dm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    dm2[ncore:nmocc,ncore:nmocc,ncore:nmocc,ncore:nmocc] = rdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] += -2
        dm2[i,i,ncore:nmocc,ncore:nmocc] = dm2[ncore:nmocc,ncore:nmocc,i,i] = 2*rdm1
        dm2[i,ncore:nmocc,ncore:nmocc,i] = dm2[ncore:nmocc,i,i,ncore:nmocc] = -rdm1

#    dm2 = (dm2 + dm2.transpose(1,0,2,3)) * .5
#    for i in range(nmocc):
#        for j in range(j+1):
#            for k in range(nmocc):
#                for l in range(k+1):
#                    assert(abs(dm2[i,j,k,l]-dm2[j,i,k,l]) < 1e-14)
#                    assert(abs(dm2[i,j,k,l]-dm2[i,j,l,k]) < 1e-14)
#                    assert(abs(dm2[i,j,k,l]-dm2[k,l,i,j]) < 1e-14)

    h1e_mo = reduce(numpy.dot, (mo.T, mf.get_hcore(mol), mo))
    jeri = ao2mo.incore.general(mf._eri, (mo, mo, mo, mo), compact=False)
    jeri = jeri.reshape(nmo,nmo,nmo,nmo)

    g = numpy.dot(h1e_mo, dm1) \
            + numpy.dot(jeri.reshape(nmo,-1), dm2.reshape(nmo,-1).T)
    h = numpy.zeros((nmo,nmo,nmo,nmo))
    for i in range(nmo):
        for j in range(nmo):
            v = numpy.dot(jeri[i,j,:,:].flatten(), dm2.reshape(nmo*nmo,-1)) \
              + numpy.dot(jeri[:,i,j,:].flatten(), \
                          dm2.transpose(0,3,1,2).reshape(nmo*nmo,-1))
            v -= numpy.dot(jeri[:,i,:,:].reshape(nmo*nmo,-1).T, \
                           dm2.transpose(0,2,1,3)[:,:,:,j].reshape(nmo*nmo,-1)).T.flatten()
            h[i,:,:,j] += -dm1 * h1e_mo[i,j] - v.reshape(nmo,nmo)
        h[:,i,i,:] += g         # (a)

    numpy.set_printoptions(5)
    return g, h

def grad_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2):
    #g, h = inter_grad_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
    g, h = _inter_grad_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
    g = g - g.transpose()
    #h = h + h.transpose(1,0,3,2) - h.transpose(0,1,3,2) - h.transpose(1,0,2,3)
    h = h - h.transpose(0,1,3,2)
    h = h - h.transpose(1,0,2,3)
    # symmetrize h, since (a) is not symm.  see Eq. (21) IJQC, 109,2178
    h = (h + h.transpose(2,3,0,1)) * .5
    return g, h

def index_univar(ncore, ncas, nvir,inner=False):
    ntot = ncore + ncas + nvir
    ilst = []
    jlst = []
    for i in range(ncore, ncore+ncas):
        for j in range(ncore):
            ilst.append(i)
            jlst.append(j)
    if inner:
        for i in range(ncore, ncore+ncas):
            for j in range(ncore, ncore+ncas):
                ilst.append(i)
                jlst.append(j)
    for i in range(ncore+ncas, ntot):
        for j in range(ncore+ncas):
            ilst.append(i)
            jlst.append(j)
    return ilst, jlst
# solve Eq. (29) of JCP, 82, 5053?

# IJQC, 109, 2178
def aug_hess(g, h):
    n = g.size
    ah = numpy.empty((n+1,n+1))
    ah[0,0] = 0
    ah[1:,0] = ah[0,1:] = g.flatten()
    ah[1:,1:] = h.reshape(n,n)
    w, u = scipy.linalg.eigh(ah)
    ith = numpy.argmax(abs(u[0])>.1)  # should it be < 1.1 like orz did?
    dx = u[1:,ith]/u[0,ith]
    #print 'w',w
    #print 'u0', u[0]
    #print 'dx',dx
    print 'o0 aug', ith, w[ith], u[0,ith], numpy.linalg.norm(dx), numpy.linalg.norm(g)
    if numpy.linalg.norm(dx) > .25:
        dx = dx * (.1/numpy.linalg.norm(dx))
    if abs(u[0,ith]) < .1 or abs(u[0,ith]) > 1.1:
        raise ValueError('incorrect displacement in augmented hessian %g'
                         % u[0,ith])
    return w[ith], dx

def expmat(a):
    #w,v = numpy.linalg.eig(a)
    #return numpy.dot(v*numpy.exp(w), v.T).real
    x1 = numpy.dot(a,a)
    u = numpy.eye(a.shape[0]) + a + .5 * x1
    #x2 = numpy.dot(x1,a)
    #u = u + 1./6 * x2
    u,w,vh = numpy.linalg.svd(u)
    return numpy.dot(u,vh)

def cycle(mol, mf, cas_eo):
    nelec_cas, nmocas = cas_eo
    nocc = mol.nelectron // 2
    nmocc = nocc - nelec_cas//2 + nmocas
    mo = mf.mo_coeff
    nao, nmo = mo.shape

    mc = mcscf.CASCI(mf, nmocas, nelec_cas)

    mo_cas = numpy.array(mo[:,nmocc-nmocas:nmocc], order='F')
    mo_core = mo[:,:nmocc-nmocas]
    elast = 0
    for i in range(10):

        mo_new = numpy.hstack((mo_core, mo_cas))
        mc.kernel(mo_new)
        rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, nmocas, nelec_cas)
        e = mc.e_cas

        print 'cycle=', i, 'e =', e, 'e_tot =', mc.e_tot, 'de', e-elast
        if abs(elast - e) < 1e-10:
            break
        elast = e

        for im in range(8):
            g, h = grad_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
            ilst, jlst = index_univar(nmocc-nmocas, nmocas, nmo-nmocc)
            g = g[ilst,jlst]
            h = h[ilst,jlst]
            h = h[:,ilst,jlst]
            #print 'b',scipy.linalg.eigh(h)[0]
            w, dx = aug_hess(g, h)
            dr = numpy.zeros((nmo, nmo))
            dr[ilst,jlst] = dx
            dr[jlst,ilst] = -dx

            u = expmat(dr)
            mo = numpy.dot(mo, u)
            mo_cas = numpy.array(mo[:,nmocc-nmocas:nmocc], order='F')
            mo_core = mo[:,:nmocc-nmocas]

    return mc.e_tot


if __name__ == '__main__':
    import scf
    import gto

    #numpy.set_printoptions(3)
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    emc = cycle(mol, m, (8,8))
    print ehf, emc, emc-ehf
    #-75.9577817425 -75.9741650131 -0.0163832706488


#    mc = mcscf.CASSCF(m, 8, 8)
#    mc.verbose = 4
#    #mc.max_cycle_macro = 5
#    mc.mc2step()
