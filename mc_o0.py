#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import numpy
from pyscf import gto
from pyscf import lib
from pyscf import ao2mo
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import ci

#    corelist = []
#    actlist = []
#    for i,occ in enumerate(occs):
#        if abs(occ-2) < 1e-8:
#            corelist.append(i)
#        elif 1e-8 < occ < 2:
#            actlist.append(i)
#    moc = mo_coeff[:,corelist]
#    dm_core = numpy.dot(moi, moi.T) * 2
# ref. JCP, 74, 2384
def fock_active(mol, mf, mo_cas, dm_core, ci_rdm1):
    dm_ao = reduce(numpy.dot, (mo_cas, ci_rdm1, mo_cas.T))
    vhf = mf.get_veff(mol, numpy.array((dm_core,dm_ao)))
    fockc = mf.get_hcore(mol) + vhf[0]
    focka = vhf[1]
    return fockc, focka

def orb_grad(mol, mf, mo_cas, mo_core, g3mo, rdm1, rdm2):
    nao, nmo = mo_cas.shape
    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    fc, fa = fock_active(mol, mf, mo_cas, dm_core, rdm1)
    fi = numpy.dot((fc + fa), mo_core) * 2
    fa = reduce(numpy.dot, (fc, mo_cas, rdm1)) \
            + numpy.dot(g3mo.reshape(-1,nao).T, rdm2.reshape(-1,nmo))
    return numpy.hstack((fi, fa))

def dsy_sqrtm(a):
    w, v = numpy.linalg.eigh(a)
    assert(numpy.all(w>0))
    return numpy.dot(v/numpy.sqrt(w), v.T)

def update_orb(mol, mf, mo_core, mo_cas, rdm1, rdm2):
    nmocas = mo_cas.shape[1]
    g2mo = ao2mo._ao2mo.nr_e1_incore(mf._eri, mo_cas, \
                                     (0,nmocas,0,nmocas))
    g3mo = partial_eri_to_g3mo(g2mo, mo_cas)
    sinv = numpy.linalg.inv(mol.intor_symmetric('cint1e_ovlp_sph'))
    f1 = orb_grad(mol, mf, mo_cas, mo_core, g3mo, rdm1, rdm2)
    mo = numpy.hstack((mo_core, mo_cas))
    eij1 = numpy.dot(mo.T, f1)
#    print mo
#    print eij1
    print numpy.linalg.norm(eij1-eij1.T)
    xsx = reduce(numpy.dot, (f1.T, sinv, f1))
    mo = reduce(numpy.dot, (sinv, f1, dsy_sqrtm(xsx)))
    return mo

def partial_eri_to_g3mo(eri, mo):
    nao, nmo = mo.shape
    def unpack(a):
        mat = numpy.empty((nao,nao))
        ij = 0
        for i in range(nao):
            for j in range(i+1):
                mat[i,j] = mat[j,i] = a[ij]
                ij += 1
        return mat
    eri1 = numpy.empty((nmo,nmo,nmo,nao))
    ij = 0
    for i in range(nmo):
        for j in range(i+1):
            v = numpy.dot(mo.T, unpack(eri[ij]))
            eri1[i,j] = eri1[j,i] = v
            ij += 1
    return eri1

def run_casci(mol, mf, mo_core, mo_cas, eri_mo, nelec):
    core_dm = numpy.dot(mo_core, mo_core.T) * 2
    corevhf = mf.get_veff(mol, core_dm)
    hcore = mf.get_hcore(mol)
    h1e_ao = hcore + corevhf
    h1e_mo = reduce(numpy.dot, (mo_cas.T, h1e_ao, mo_cas))
    energy_core = numpy.einsum('ij,ji', core_dm, hcore) \
                + numpy.einsum('ij,ji', core_dm, corevhf) * .5

    nmo = h1e_mo.shape[0]
    rdm1 = numpy.empty((nmo,nmo))
    rdm2 = numpy.empty((nmo,nmo,nmo,nmo))
    rec = ci.fci._run(mol, nelec, h1e_mo, eri_mo, rdm1=rdm1, rdm2=rdm2)
    e = ci.fci.find_fci_key(rec, 'FCI STATE 1 ENERGY') + energy_core
    rdm1, rdm2 = ci.fci.reorder_rdm(rdm1, rdm2)
    log.debug(mol, rec)
    return e, rdm1, rdm2

def cycle(mol, mf, cas_eo):
    nelec_cas, nmocas = cas_eo
    nocc = mol.nelectron / 2
    mo = mf.mo_coeff[:, :nocc-nelec_cas/2+nmocas]
    nao = mo.shape[0]
    for i in range(10):
        mo_cas = numpy.array(mo[:,-nmocas:], order='F')
        mo_core = mo[:,:-nmocas]
        eri_mo = ao2mo.incore.full(mf._eri, mo_cas)
        e, rdm1, rdm2 = run_casci(mol, mf, mo_core, mo_cas, eri_mo, nelec_cas)
        print 'e =', e + mol.nuclear_repulsion()

        mo = update_orb(mol, mf, mo_core, mo_cas, rdm1, rdm2)


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

    mol.basis = {'H': '6-31g',
                 'O': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    print m.scf()
    cycle(mol, m, (2,2))

#    mol = gto.Mole()
#    mol.verbose = 0
#    mol.output = None#"out_h2o"
#    mol.atom = [
#        ['H', (0, 0, 0.)],
#        ['H', (0, 0, 1.)],]
#
#    mol.basis = {'H': '6-31g',}
#    mol.build()
#
#    m = scf.RHF(mol)
#    print m.scf()
#    cycle(mol, m, (2,4))
#
