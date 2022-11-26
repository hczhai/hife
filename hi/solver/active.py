
from .mol import TIME_ST, TIME_ED

MF_LOAD = """
from pyscf import scf
mfchk = "%s"
mol, mfx = scf.chkfile.load_scf(mfchk)
mf = scf.UHF(mol)
mf.chkfile = "mf.chk"
mf.mo_coeff = mfx["mo_coeff"]
mf.mo_energy = mfx["mo_energy"]
mf.mo_occ = mfx["mo_occ"]
"""

PM_LOC = """
import numpy as np
import scipy.linalg

# Active space procedure and Boys/PM-Localization
# Original author : Zhendong Li

def sqrtm(s):
    e, v = np.linalg.eigh(s)
    return np.dot(v * np.sqrt(e), v.T.conj())

def lowdin(s):
    e, v = np.linalg.eigh(s)
    return np.dot(v / np.sqrt(e), v.T.conj())

def loc(mol, mocoeff, tol=1E-6, maxcycle=1000, iop=0):
    part = {}
    for iatom in range(mol.natm):
        part[iatom] = []
    ncgto = 0
    for binfo in mol._bas:
        atom_id = binfo[0]
        lang = binfo[1]
        ncntr = binfo[3]
        nbas = ncntr * (2 * lang + 1)
        part[atom_id] += range(ncgto, ncgto + nbas)
        ncgto += nbas
    partition = []
    for iatom in range(mol.natm):
        partition.append(part[iatom])
    ova = mol.intor_symmetric("cint1e_ovlp_sph")
    print()
    print('[pm_loc_kernel]')
    print(' mocoeff.shape=',mocoeff.shape)
    print(' tol=',tol)
    print(' maxcycle=',maxcycle)
    print(' partition=',len(partition),'\\n',partition)
    k = mocoeff.shape[0]
    n = mocoeff.shape[1]
    natom = len(partition)
 
    def genPaij(mol,mocoeff,ova,partition,iop):
        c = mocoeff.copy()
        # Mulliken matrix
        if iop == 0:
            cts = c.T.dot(ova)
            natom = len(partition)
            pija = np.zeros((natom,n,n))
            for iatom in range(natom):
                idx = partition[iatom]
                tmp = np.dot(cts[:,idx],c[idx,:])
                pija[iatom] = 0.5*(tmp+tmp.T)
        # Lowdin
        elif iop == 1:
            s12 = sqrtm(ova)
            s12c = s12.T.dot(c)
            natom = len(partition)
            pija = np.zeros((natom,n,n))
            for iatom in range(natom):
                idx = partition[iatom]
                pija[iatom] = np.dot(s12c[idx,:].T,s12c[idx,:])
        # Boys
        elif iop == 2:
            rmat = mol.intor_symmetric('cint1e_r_sph',3)
            pija = np.zeros((3,n,n))
            for icart in range(3):
                pija[icart] = c.T @ rmat[icart] @ c
        # P[i,j,a]
        pija = pija.transpose(1,2,0).copy()
        return pija
 
    u = np.identity(n)
    pija = genPaij(mol,mocoeff,ova,partition,iop)
 
    # Start
    def funval(pija):
        return np.einsum('iia,iia',pija,pija)
 
    fun = funval(pija)
    print(' initial funval = ',fun)
    for icycle in range(maxcycle):
        delta = 0.0
        # i>j
        ijdx = []
        for i in range(n-1):
            for j in range(i+1,n):
                bij = abs(np.sum(pija[i,j]*(pija[i,i]-pija[j,j])))
                ijdx.append((i,j,bij))
        ijdx = sorted(ijdx,key=lambda x:x[2], reverse=True)
        for i,j,bij in ijdx:
            # determine angle
            vij = pija[i,i]-pija[j,j]
            aij = np.dot(pija[i,j],pija[i,j]) - 0.25*np.dot(vij,vij)
            bij = np.dot(pija[i,j],vij)
            if abs(aij)<1.e-10 and abs(bij)<1.e-10: continue
            p1 = np.sqrt(aij**2+bij**2)
            cos4a = -aij/p1
            sin4a = bij/p1
            cos2a = np.sqrt((1+cos4a)*0.5)
            sin2a = np.sqrt((1-cos4a)*0.5)
            cosa  = np.sqrt((1+cos2a)*0.5)
            sina  = np.sqrt((1-cos2a)*0.5)
            # Why? Because we require alpha in [0,pi/2]
            if sin4a < 0.0:
                cos2a = -cos2a
                sina, cosa = cosa, sina
            # stationary condition
            if abs(cosa-1.0)<1.e-10: continue
            if abs(sina-1.0)<1.e-10: continue
            # incremental value
            delta += p1*(1-cos4a)
            # Transformation
            # Urot
            ui = u[:,i]*cosa+u[:,j]*sina
            uj = -u[:,i]*sina+u[:,j]*cosa
            u[:,i] = ui.copy()
            u[:,j] = uj.copy()
            # Bra-transformation of Integrals
            tmp_ip = pija[i,:,:]*cosa+pija[j,:,:]*sina
            tmp_jp = -pija[i,:,:]*sina+pija[j,:,:]*cosa
            pija[i,:,:] = tmp_ip.copy()
            pija[j,:,:] = tmp_jp.copy()
            # Ket-transformation of Integrals
            tmp_ip = pija[:,i,:]*cosa+pija[:,j,:]*sina
            tmp_jp = -pija[:,i,:]*sina+pija[:,j,:]*cosa
            pija[:,i,:] = tmp_ip.copy()
            pija[:,j,:] = tmp_jp.copy()
        fun = fun+delta
        print('icycle=', icycle, 'delta=', delta, 'fun=', fun)
        if delta < tol:
            break

    # Check
    ierr = 0
    if delta < tol: 
        print('CONG: PMloc converged!')
    else:
        ierr = 1
        print('WARNING: PMloc not converged')
    return ierr, u

def loc_pg(mol, mocoeff, orb_sym):
    assert mocoeff.shape[1] == len(orb_sym)
    ierr = 0
    ru = np.zeros((len(orb_sym), len(orb_sym)), dtype=mocoeff.dtype)
    for isym in set(orb_sym):
        mask = np.array(orb_sym) == isym
        jerr, u = loc(mol, mocoeff[:, mask])
        ru[np.outer(mask, mask)] = u.flatten()
        ierr = ierr | jerr
    return ierr, ru

"""

ACT = """
# 1. Read UHF-alpha/beta orbitals from chkfile

ma, mb = mf.mo_coeff
norb = ma.shape[1]
nalpha = (mol.nelectron + mol.spin) // 2
nbeta  = (mol.nelectron - mol.spin) // 2
print('Nalpha = %d, Nbeta %d, Sz = %d, Norb = %d' % (nalpha, nbeta, mol.spin, norb))

# 2. Sanity check, using orthogonality

ova = mol.intor_symmetric("cint1e_ovlp_sph")
diff = ma.T @ ova @ ma - np.identity(norb)
assert np.linalg.norm(diff) < 1E-7
diff = mb.T @ ova @ mb - np.identity(norb)
assert np.linalg.norm(diff) < 1E-7

pTa = np.dot(ma[:, :nalpha], ma[:, :nalpha].T)
pTb = np.dot(mb[:, :nbeta], mb[:, :nbeta].T)
pT = 0.5 * (pTa + pTb)

# Lowdin basis
s12 = sqrtm(ova)
s12inv = lowdin(ova)
pT = s12 @ pT @ s12
print('Idemponency of DM: %s' % np.linalg.norm(pT.dot(pT) - pT))
enorb = mf.mo_energy

fa = ma @ np.diag(enorb[0]) @ ma.T
fb = mb @ np.diag(enorb[1]) @ mb.T
fav = (fa + fb) / 2
fock_sf = fOAO = s12 @ fav @ s12

# 'natural' occupations and orbitals
eig, coeff = np.linalg.eigh(pT)
eig = 2 * eig
eig[abs(eig) < 1E-14] = 0.0

# Rotate back to AO representation and check orthogonality
coeff = np.dot(s12inv, coeff)
diff = coeff.T @ ova @ coeff - np.identity(norb)
assert np.linalg.norm(diff) < 1E-7

# 3. Search for active space

# 3.1 Transform the entire MO space into core, active, and external space
# based on natural occupancy

fexpt = coeff.T @ ova @ fav @ ova @ coeff
enorb = np.diag(fexpt)
index = np.argsort(-eig)
enorb = enorb[index]
nocc  = eig[index]
coeff = coeff[:, index]

# Reordering and define active space according to thresh

thresh = 0.05
active = (thresh <= nocc) & (nocc <= 2 - thresh)
act_idx = np.where(active)[0]
print('active orbital indices %s' % act_idx)
print('Num active orbitals %d' % len(act_idx))
cOrbs = coeff[:, :act_idx[0]]
aOrbs = coeff[:, act_idx]
vOrbs = coeff[:, act_idx[-1]+1:]
norb = cOrbs.shape[0]
nc = cOrbs.shape[1]
na = aOrbs.shape[1]
nv = vOrbs.shape[1]
print('core orbs:', cOrbs.shape)
print('act  orbs:', aOrbs.shape)
print('vir  orbs:', vOrbs.shape)
assert nc + na + nv == norb

# 3.2 Localizing core, active, external space separately, based on certain
# local orbitals.

cOrbsOAO = np.dot(s12, cOrbs)
aOrbsOAO = np.dot(s12, aOrbs)
vOrbsOAO = np.dot(s12, vOrbs)
assert 'Ortho-cOAO', np.linalg.norm(np.dot(cOrbsOAO.T, cOrbsOAO) - np.identity(nc)) < 1E-7
assert 'Ortho-aOAO', np.linalg.norm(np.dot(aOrbsOAO.T, aOrbsOAO) - np.identity(na)) < 1E-7
assert 'Ortho-vOAO', np.linalg.norm(np.dot(vOrbsOAO.T, vOrbsOAO) - np.identity(nv)) < 1E-7

def scdm(coeff, overlap, aux):
    no = coeff.shape[1]
    ova = coeff.T @ overlap @ aux
    q, r, piv = scipy.linalg.qr(ova, pivoting=True)
    bc = ova[:, piv[:no]]
    ova = np.dot(bc.T, bc)
    s12inv = lowdin(ova)
    cnew = coeff @ bc @ s12inv
    return cnew

clmo = cOrbs
almo = aOrbs
ierr, uc = loc(mol, clmo)
ierr, ua = loc(mol, almo)
clmo = clmo.dot(uc)
almo = almo.dot(ua)

# clmo = scdm(cOrbs, ova, s12inv)  # local "AOs" in core space
# almo = scdm(aOrbs, ova, s12inv)  # local "AOs" in active space
vlmo = scdm(vOrbs, ova, s12inv)  # local "AOs" in external space

# 3.3 Sorting each space (core, active, external) based on "orbital energy" to
# prevent high-lying orbitals standing in valence space.

# Get <i|F|i>

def psort(ova, fav, pT, s12, coeff):
   pTnew = 2.0 * (coeff.T @ s12 @ pT @ s12 @ coeff)
   nocc  = np.diag(pTnew)
   index = np.argsort(-nocc)
   ncoeff = coeff[:, index]
   nocc = nocc[index]
   enorb = np.diag(coeff.T @ ova @ fav @ ova @ coeff)
   enorb = enorb[index]
   return ncoeff, nocc, enorb

# E-SORT

mo_c, n_c, e_c = psort(ova, fav, pT, s12, clmo)
mo_o, n_o, e_o = psort(ova, fav, pT, s12, almo)
mo_v, n_v, e_v = psort(ova, fav, pT, s12, vlmo)

# coeff is the local molecular orbitals

coeff = np.hstack((mo_c, mo_o, mo_v))
mo_occ = np.hstack((n_c, n_o, n_v))

# Test orthogonality for the localize MOs as before

diff = coeff.T @ ova @ coeff - np.identity(norb)
assert np.linalg.norm(diff) < 1E-7

# Population analysis to confirm that our LMO (coeff) make sense

lcoeff = s12.dot(coeff)

diff = lcoeff.T @ lcoeff - np.identity(norb)
assert np.linalg.norm(diff) < 1E-7

print('\\nLowdin population for LMOs:')

labels = mol.ao_labels(None)
texts = [None] * norb
for iorb in range(norb):
    vec = lcoeff[:, iorb] ** 2
    ivs = np.argsort(vec)
    if iorb < nc:
        text = "[C %3d] occ = %.5f" % (iorb, mo_occ[iorb])
        ftext = " fii = %10.3f" % e_c[iorb]
    elif iorb >= nc and iorb < nc+na:
        text = "[A %3d] occ = %.5f" % (iorb, mo_occ[iorb])
        ftext = " fii = %10.3f" % e_o[iorb - nc]
    else:
        text = "[V %3d] occ = %.5f" % (iorb, mo_occ[iorb])
        ftext = " fii = %10.3f" % e_v[iorb - nc - na]
    gtext = ''
    for iao in ivs[::-1][:3]:
        gtext += "(%3d-%2s-%7s = %5.3f) " % (labels[iao][0], labels[iao][1],
            labels[iao][2] + labels[iao][3], vec[iao])
    print(text + ftext + " " + gtext)
    texts[iorb] = text + "\\n" + gtext

dmao = np.einsum('i,pi,qi->pq', mo_occ, coeff, coeff, optimize=True)
np.save("lo_occ.npy", mo_occ)
np.save("lo_coeff.npy", coeff)
np.save("lo_texts.npy", texts)
"""

def write(fn, pma):
    with open(fn, "w") as f:

        f.write(TIME_ST)

        lde = pma["load_mf"]
        if "/" not in lde:
            lde = "../" + lde

        f.write(MF_LOAD % (lde + "/mf.chk"))

        f.write(PM_LOC)
        f.write(ACT)
        f.write(TIME_ED)
