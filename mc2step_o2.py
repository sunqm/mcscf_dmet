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
from pyscf import fci

# ref. JCP, 82, 5053;  JCP, 73, 2342

def inter_grad(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2):
    ncas = mo_cas.shape[1]
    ncore = mo_core.shape[1]
    nmocc = ncas + ncore
    nao, nmo = mo.shape
    dm1 = numpy.zeros((nmo,nmo))
    for i in range(ncore):
        dm1[i,i] = 2
    dm1[ncore:nmocc,ncore:nmocc] = rdm1

    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    dm_active = reduce(numpy.dot, (mo_cas, rdm1, mo_cas.T))
    vhf_core, vhf_active = \
            mf.get_veff(mol, numpy.array((dm_core,dm_active)))

    # todo, optimize me: partial trans -> dot to rdm2 -> partial trans
    jeri = ao2mo.incore.general(mf._eri, (mo_cas, mo_cas, mo_cas, mo), \
                                compact=False)
    #g2dm = numpy.dot(dm2.reshape(ncas,-1), jeri.reshape(nmo,-1).T)
    g2dm = numpy.dot(jeri.reshape(-1, nmo).T, rdm2.reshape(-1,ncas))

    h1e_mo = reduce(numpy.dot, (mo.T, mf.get_hcore(mol), mo))
    g = numpy.dot(h1e_mo, dm1)
    g[:,:ncore] += reduce(numpy.dot, (mo.T, vhf_active, mo_core)) * 2
    g[:,:ncore] += reduce(numpy.dot, (mo.T, vhf_core, mo_core)) * 2
    g[:,ncore:nmocc] += g2dm \
            + reduce(numpy.dot, (mo.T, vhf_core, mo_cas, rdm1))
    return g

def orb_grad(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2):
    g = inter_grad(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
    return g - g.transpose()

def inter_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2):
    ncas = mo_cas.shape[1]
    ncore = mo_core.shape[1]
    nmocc = ncas + ncore
    nao,nmo = mo.shape
    dm1 = numpy.zeros((nmo,nmo))
    for i in range(ncore):
        dm1[i,i] = 2
    dm1[ncore:nmocc,ncore:nmocc] = rdm1

    g = inter_grad(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
    h = numpy.zeros((nmo,nmo,nmo,nmo))

    h1e_mo = reduce(numpy.dot, (mo.T, mf.get_hcore(mol), mo))
    jeri = ao2mo.incore.general(mf._eri, (mo, mo, mo, mo), compact=False)
    jeri = jeri.reshape(nmo,nmo,nmo,nmo)
    eri1 = jeri[:,:,ncore:nmocc,ncore:nmocc]
    h1 = numpy.dot(eri1.reshape(-1,ncas*ncas), rdm2.reshape(ncas*ncas,-1).T)
    rdm2a = rdm2 + rdm2.transpose(0,1,3,2)
    eri1 = jeri[:,ncore:nmocc,ncore:nmocc,:]
    h2 = numpy.dot(eri1.transpose(0,3,1,2).reshape(-1,ncas*ncas),
                   rdm2a.transpose(1,2,0,3).reshape(ncas*ncas,-1))

    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    dm_active = reduce(numpy.dot, (mo_cas, rdm1, mo_cas.T))
    vhf_core, vhf_active = \
            mf.get_veff(mol, numpy.array((dm_core,dm_active)))
    vhf1 = reduce(numpy.dot, (mo.T, vhf_core+vhf_active, mo))
    for i in range(ncore):
        h[:,i,:,i] = vhf1 * 2
    vhf2 = reduce(numpy.dot, (mo.T, vhf_core, mo))
    h3 = numpy.dot(vhf2.reshape(-1,1), rdm1.reshape(1,-1))

    h[:,ncore:nmocc,:,ncore:nmocc] = \
            (h1+h2+h3).reshape(nmo,nmo,ncas,ncas).transpose(0,2,1,3)

    eri1 = jeri[:,:ncore,:ncore,:] * 4
    h[:,:ncore,:,:ncore] += eri1.transpose(0,1,3,2)
    eri1 = jeri[:,:ncore,:,:ncore] * 4
    h[:,:ncore,:,:ncore] += eri1
    eri1 = jeri[:,:,:ncore,:ncore] * 2
    h[:,:ncore,:,:ncore] -= eri1.transpose(0,3,1,2)
    eri1 = jeri[:,:ncore,:,:ncore] * 2
    h[:,:ncore,:,:ncore] -= eri1.transpose(0,3,2,1)

    eri1 = jeri[:,ncore:nmocc,:ncore,:].transpose(1,0,3,2) * 2
    eri1+= jeri[:,ncore:nmocc,:,:ncore].transpose(1,0,2,3) * 2
    eri1-= jeri[:,:,:ncore,ncore:nmocc].transpose(3,0,1,2)
    eri1-= jeri[:,:ncore,:,ncore:nmocc].transpose(3,0,2,1)
    h1 = numpy.dot(rdm1.T,eri1.reshape(ncas,-1)).reshape(ncas,nmo,nmo,ncore)
    h[:,ncore:nmocc,:,:ncore] = h1.transpose(1,0,2,3)

    eri1 = jeri[:,:ncore,ncore:nmocc,:].transpose(0,1,3,2)
    eri2 = jeri[:,:,ncore:nmocc,:ncore].transpose(0,3,1,2)
    eri1 = eri1 * 2 - eri2
    h[:,:ncore,:,ncore:nmocc] = numpy.dot(eri1, rdm1)
    eri1 = jeri[:,:ncore,:,ncore:nmocc]
    eri2 = jeri[:,ncore:nmocc,:,:ncore].transpose(0,3,2,1)
    eri1 = eri1 * 2 - eri2
    h[:,:ncore,:,ncore:nmocc] += numpy.dot(eri1, rdm1)

    h1 = numpy.dot(h1e_mo.reshape(-1,1), \
                   dm1.reshape(1,-1)).reshape(nmo,nmo,nmo,nmo)
    h += h1.transpose(0,2,1,3)
    for i in range(nmo):
        h[:,i,i,:] += g
    return h

def orb_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2):
    h = inter_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
    h = h - h.transpose(0,1,3,2)
    h = h - h.transpose(1,0,2,3)
    h = (h + h.transpose(2,3,0,1)) * .5
    return h

def gen_hess_op(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2):
    ncas = mo_cas.shape[1]
    ncore = mo_core.shape[1]
    nmocc = ncas + ncore
    nao,nmo = mo.shape
    dm1 = numpy.zeros((nmo,nmo))
    for i in range(ncore):
        dm1[i,i] = 2
    dm1[ncore:nmocc,ncore:nmocc] = rdm1

    g = inter_grad(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
    h = numpy.zeros((nmo,nmo,nmo,nmo))

    # part1
    h1e_mo = reduce(numpy.dot, (mo.T, mf.get_hcore(mol), mo))
    jeri = ao2mo.incore.general(mf._eri, (mo, mo, mo, mo), compact=False)
    jeri = jeri.reshape(nmo,nmo,nmo,nmo)
    eri1 = jeri[:,:,ncore:nmocc,ncore:nmocc]
    h1 = numpy.dot(eri1.reshape(-1,ncas*ncas), rdm2.reshape(ncas*ncas,-1).T)
    rdm2a = rdm2 + rdm2.transpose(0,1,3,2)
    eri1 = jeri[:,ncore:nmocc,ncore:nmocc,:]
    h2 = numpy.dot(eri1.transpose(0,3,1,2).reshape(-1,ncas*ncas),
                   rdm2a.transpose(1,2,0,3).reshape(ncas*ncas,-1))

    # part2
    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    dm_active = reduce(numpy.dot, (mo_cas, rdm1, mo_cas.T))
    vhf_core, vhf_active = \
            mf.get_veff(mol, numpy.array((dm_core,dm_active)))
    vhf1 = reduce(numpy.dot, (mo.T, vhf_core+vhf_active, mo))
    for i in range(ncore):
        h[:,i,:,i] = vhf1 * 2
    # part3
    vhf2 = reduce(numpy.dot, (mo.T, vhf_core, mo))
    h3 = numpy.dot(vhf2.reshape(-1,1), rdm1.reshape(1,-1))

    h[:,ncore:nmocc,:,ncore:nmocc] = \
            (h1+h2+h3).reshape(nmo,nmo,ncas,ncas).transpose(0,2,1,3)

    # part4
    eri1 = jeri[:,:ncore,:ncore,:] * 4
    h[:,:ncore,:,:ncore] += eri1.transpose(0,1,3,2)
    eri1 = jeri[:,:ncore,:,:ncore] * 4
    h[:,:ncore,:,:ncore] += eri1
    eri1 = jeri[:,:,:ncore,:ncore] * 2
    h[:,:ncore,:,:ncore] -= eri1.transpose(0,3,1,2)
    eri1 = jeri[:,:ncore,:,:ncore] * 2
    h[:,:ncore,:,:ncore] -= eri1.transpose(0,3,2,1)

    # part5
    eri1 = jeri[:,ncore:nmocc,:ncore,:].transpose(1,0,3,2) * 2
    eri1+= jeri[:,ncore:nmocc,:,:ncore].transpose(1,0,2,3) * 2
    eri1-= jeri[:,:,:ncore,ncore:nmocc].transpose(3,0,1,2)
    eri1-= jeri[:,:ncore,:,ncore:nmocc].transpose(3,0,2,1)
    h1 = numpy.dot(rdm1.T,eri1.reshape(ncas,-1)).reshape(ncas,nmo,nmo,ncore)
    h[:,ncore:nmocc,:,:ncore] = h1.transpose(1,0,2,3)

    # part6
    eri1 = jeri[:,:ncore,ncore:nmocc,:].transpose(0,1,3,2)
    eri2 = jeri[:,:,ncore:nmocc,:ncore].transpose(0,3,1,2)
    eri1 = eri1 * 2 - eri2
    h[:,:ncore,:,ncore:nmocc] = numpy.dot(eri1, rdm1)
    eri1 = jeri[:,:ncore,:,ncore:nmocc]
    eri2 = jeri[:,ncore:nmocc,:,:ncore].transpose(0,3,2,1)
    eri1 = eri1 * 2 - eri2
    h[:,:ncore,:,ncore:nmocc] += numpy.dot(eri1, rdm1)

    # part7
    h1 = numpy.dot(h1e_mo.reshape(-1,1), \
                   dm1.reshape(1,-1)).reshape(nmo,nmo,nmo,nmo)
    h += h1.transpose(0,2,1,3)
    # part8
    for i in range(nmo):
        h[:,i,i,:] += g

    ilst, jlst = index_univar(ncore, ncas, nmo-nmocc)

    h = h - h.transpose(0,1,3,2)
    h = h - h.transpose(1,0,2,3)
    h = (h + h.transpose(2,3,0,1)) * .5
    h = h[ilst,jlst][:,ilst,jlst]

    def h_op(x):
        return numpy.dot(h,x)
    return h_op, h.diagonal()

def index_univar(ncore, ncas, nvir, inner=False):
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

# IJQC, 109, 2178
# use davidson algorithm to solve Ax = xe
def aug_hess(g, h, tol=1e-8, maxiter=20):
    n = g.size
    ah = numpy.empty((n+1,n+1))
    ah[0,0] = 0
    g = g.flatten()
    ah[1:,0] = ah[0,1:] = g
    h = h.reshape(n,n)
    ah[1:,1:] = h

    x0 = ah[0,1:]/numpy.linalg.norm(ah[0,1:])
    h_op = lambda x: numpy.dot(h, x)
    precond = lambda x, e: x/(h.diagonal()-e)

    return davidson_aug_hess(h_op, g, precond, x0, tol, maxiter)


def davidson_aug_hess(h_op, g, precond, x0, tol=1e-8, maxiter=20):
    # the first trial vector is (1,0,0,...), which is not included in xs
    x0 = x0/numpy.linalg.norm(x0)
    xs = [x0]
    ax = [h_op(x0)]

    for istep in range(min(maxiter,x0.size)):
        xstmp = numpy.array(xs)
        axtmp = numpy.array(ax)
        nvec = len(xs)
        asub = numpy.zeros((nvec+1,nvec+1))
        asub[1:,0] = asub[0,1:] = numpy.dot(xstmp, g)
        asub[1:,1:] = numpy.dot(xstmp, axtmp.T)
        w, v = scipy.linalg.eigh(asub)
        index = numpy.argmax(abs(v[0])>.01)
        v_t = v[:,index]
        w_t = w[index]
        xtrial = numpy.dot(v_t[1:], xstmp)

        gx = numpy.dot(asub[0,1:], v_t[1:])
        hx = numpy.dot(v_t[1:], axtmp)
        # be careful with g*v_t[0], as the first trial vector is (1,0,0,...)
        dx = hx + g*v_t[0] - xtrial * w_t
        rr = numpy.linalg.norm(dx)
        #rr = numpy.sqrt(rr**2 + (gx-v_t[0]*w_t)**2)
# w_t is likely ~0. So precond(dx, w_t) ~ hx/h_ii, which is easily stuck.
        if rr < tol:
            break
        xs.append(precond(dx, w_t))
        q, r = numpy.linalg.qr(numpy.array(xs).T)
        xs[-1] = q[:,-1]
        ax.append(h_op(xs[-1]))
        #print('step, rr, w_t, v_t[0], index', istep, rr, w_t, v_t[0], index, r[-1,-1])
    print('step =', istep)

    dx = xtrial/v_t[0]
    print(w_t, v_t[0], numpy.linalg.norm(dx), numpy.linalg.norm(g))
    if numpy.linalg.norm(dx) > .25:
        dx = dx * (.5/numpy.linalg.norm(dx))
    if abs(v_t[0]) < .01 or abs(v_t[0]) > 1.1:
        raise ValueError('incorrect displacement in augmented hessian %g'
                         % v_t[0])
    return w_t, dx

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
    #mo = numpy.hstack((mo[:,[2,3,4,0,1]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[1,3,4,0,2]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[2,1,4,0,3]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[2,3,1,0,4]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[0,3,4,1,2]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[0,2,4,1,3]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[0,2,3,1,4]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[0,1,4,2,3]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[0,1,3,2,4]],mo[:,5:]))
    #mo = numpy.hstack((mo[:,[0,1,2,3,4]],mo[:,5:]))
    mo_cas = numpy.array(mo[:,nmocc-nmocas:nmocc], order='F')
    mo_core = mo[:,:nmocc-nmocas]
    elast = 0
    for i in range(4):

        eri_mo = ao2mo.incore.full(mf._eri, mo_cas)
        #e, rdm1, rdm2 = mc_o0.run_casci(mol, mf, mo_core, mo_cas, eri_mo, nelec_cas)
        dm_core = mo_core.dot(mo_core.T) * 2
        vhf = mf.get_veff(mol, dm_core)
        h1 = reduce(numpy.dot, (mo_cas.T, mf.get_hcore()+vhf, mo_cas))
        e, civec = fci.direct_spin1.kernel(h1, eri_mo, nmocas, nelec_cas)
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(civec, nmocas, nelec_cas)
        print('e =', e)
        if abs(elast - e) < 1e-10:
            break
        elast = e

        for im in range(2):
            g = orb_grad(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
            ilst, jlst = index_univar(nmocc-nmocas, nmocas, nmo-nmocc)
            g = g[ilst,jlst]

            h = orb_hess(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
            h = h[ilst,jlst]
            h = h[:,ilst,jlst]
            w, dx = aug_hess(g, h)

#            h_op, h_diag = gen_hess_op(mol, mf, mo_cas, mo_core, mo, rdm1, rdm2)
#            precond = lambda x, e: x/(h_diag-e)
#            x0 = g/numpy.linalg.norm(g)
#            w, dx = davidson_aug_hess(h_op, g, precond, x0)

            dr = numpy.zeros((nmo, nmo))
            dr[ilst,jlst] = dx
            dr[jlst,ilst] = -dx

            u = expmat(dr)
            mo = numpy.dot(mo, u)
            mo_cas = numpy.array(mo[:,nmocc-nmocas:nmocc], order='F')
            mo_core = mo[:,:nmocc-nmocas]

    return e


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto

    ##numpy.set_printoptions(3)
    #mol = gto.Mole()
    #mol.verbose = 0
    #mol.output = None#"out_h2o"
    #mol.atom = [
    #    ['O', ( 0., 0.    , 0.   )],
    #    ['H', ( 0., -0.757, 0.587)],
    #    ['H', ( 0., 0.757 , 0.587)],]

    #mol.basis = {'H': 'sto-3g',
    #             'O': '6-31g',}
    #mol.build()

    #m = scf.RHF(mol)
    #ehf = m.scf()
    #emc = cycle(mol, m, (4,4))
    #print(ehf, emc, emc-ehf)
    ##-75.9577817425 -75.9741650131 -0.0163832706488

    mol = gto.Mole()
    mol.verbose = 1
    mol.output = "out_o2"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        #['H', ( 0.,-0.5   ,-1.   )],
        #['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = {'H': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    emc = cycle(mol, m, (6,6))
    print(ehf, emc, emc-ehf)
