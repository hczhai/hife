
from .mol import TIME_ST, TIME_ED
from .active import PM_LOC

MF_LOAD = """
from pyscf import scf, symm
import numpy as np
mfchk = "%s"
mol, mfx = scf.chkfile.load_scf(mfchk)
mo_coeff = mfx["mo_coeff"]
mo_energy = mfx["mo_energy"]
mo_occ = mfx["mo_occ"]

if mo_coeff[0].ndim == 2:
    ma, mb = mo_coeff

    nalpha = (mol.nelectron + mol.spin) // 2
    nbeta = (mol.nelectron - mol.spin) // 2
    pTa = np.dot(ma[:, :nalpha], ma[:, :nalpha].T)
    pTb = np.dot(mb[:, :nbeta], mb[:, :nbeta].T)
    pav = 0.5 * (pTa + pTb)

    fa = ma @ np.diag(mo_energy[0]) @ ma.T
    fb = mb @ np.diag(mo_energy[1]) @ mb.T
    fav = 0.5 * (fa + fb)
else:
    pav = 0.5 * (mo_coeff @ np.diag(mo_occ) @ mo_coeff.T)
    fav = mo_coeff @ np.diag(mo_energy) @ mo_coeff.T

ova = mol.intor_symmetric('cint1e_ovlp_sph')
nactorb = None
nactelec = None
split_low = 0.0
split_high = 0.0
alpha = False
beta = False
uno = False
average_occ = False
loc_with_pg = False
"""

SELECT = """
from pyscf import tools
import numpy as np
import os

for fname in ["mo_coeff.npy", "lo_coeff.npy", "nat_coeff.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        coeff = np.load(lde + "/" + fname)
        break

for fname in ["mf_occ.npy", "lo_occ.npy", "nat_occ.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        mo_occ = np.load(lde + "/" + fname)
        break

if alpha:
    coeff = coeff[0]
    mo_occ = mo_occ[0]
elif beta:
    coeff = coeff[1]
    mo_occ = mo_occ[1]
elif uno:
    # 1. Read UHF-alpha/beta orbitals from chkfile
    ma, mb = coeff
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

    print('alpha occ = ', mo_occ[0])
    print('beta  occ = ', mo_occ[1])

    pTa = ma @ np.diag(mo_occ[0]) @ ma.T
    pTb = mb @ np.diag(mo_occ[1]) @ mb.T
    pT = 0.5 * (pTa + pTb)

    # Lowdin basis
    s12 = sqrtm(ova)
    s12inv = lowdin(ova)
    pT = s12 @ pT @ s12
    print('Idemponency of DM: %s' % np.linalg.norm(pT.dot(pT) - pT))

    # 'natural' occupations and orbitals
    mo_occ, coeff = np.linalg.eigh(pT)
    mo_occ = 2 * mo_occ
    mo_occ[abs(mo_occ) < 1E-14] = 0.0

    # Rotate back to AO representation and check orthogonality
    coeff = np.dot(s12inv, coeff)
    diff = coeff.T @ ova @ coeff - np.identity(norb)
    assert np.linalg.norm(diff) < 1E-7

    index = np.argsort(-mo_occ)
    mo_occ  = mo_occ[index]
    coeff = coeff[:, index]

    np.save("lo_coeff.npy", coeff)
    np.save("lo_occ.npy", mo_occ)

    if average_occ:
        for fname in ["cc_mo_coeff.npy"]:
            if os.path.isfile(lde + "/" + fname):
                print("use: " + lde + "/" + fname)
                mo_coeff = np.load(lde + "/" + fname)
                break

        for fname in ["cc_dmmo.npy"]:
            if os.path.isfile(lde + "/" + fname):
                print("use: " + lde + "/" + fname)
                dmmo = np.load(lde + "/" + fname)
                break
        
        dmao = np.einsum('...pi,...ij,...qj->...pq', mo_coeff, dmmo, mo_coeff, optimize=True)
        if dmao.ndim == 3:
            dmao = np.einsum('spq->pq', dmao, optimize=True)

        coeff_inv = np.linalg.pinv(coeff)
        dmmo = np.einsum('ip,pq,jq->ij', coeff_inv, dmao, coeff_inv, optimize=True)

        print('AVERAGE TRACE = %8.5f' % np.trace(dmmo))

        nat_occ, u = np.linalg.eigh(dmmo)
        print('AVERAGE NAT OCC = ', ''.join(['%8.5f,' % x for x in nat_occ[::-1]]))

"""

SELECT2 = """

def psort(ova, fav, pT, coeff, orb_sym=None):
    pTnew = 2.0 * (coeff.T @ ova @ pT @ ova @ coeff)
    nocc  = np.diag(pTnew)
    index = np.argsort(-nocc)
    ncoeff = coeff[:, index]
    nocc = nocc[index]
    enorb = np.diag(coeff.T @ ova @ fav @ ova @ coeff)
    enorb = enorb[index]
    if orb_sym is not None:
        orb_sym = orb_sym[index]
    return ncoeff, nocc, enorb, orb_sym

if cas_list is None:
    assert nactorb is not None
    assert nactelec is not None
    ncore = (mol.nelectron - nactelec) // 2
    cas_list = list(range(ncore, ncore + nactorb))

print('cas list = ', cas_list)

orb_sym = None
if loc_with_pg:
    orb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, coeff, tol=1e-2)
    orb_sym = np.array([symm.irrep_name2id(mol.groupname, ir) for ir in orb_sym])

if split_low == 0.0 and split_high == 0.0:

    print('simple localization')

    actmo = coeff[:, np.array(cas_list, dtype=int)]
    if do_loc:
        if not loc_with_pg:
            ierr, ua = loc(mol, actmo)
            actmo = actmo.dot(ua)
        else:
            act_orb_sym = orb_sym[np.array(cas_list, dtype=int)]
            ierr, ua = loc_pg(mol, actmo, act_orb_sym)
            actmo = actmo.dot(ua)
    if not loc_with_pg:
        actmo, actocc, e_o, _ = psort(ova, fav, pav, actmo, None)
    else:
        act_orb_sym = orb_sym[np.array(cas_list, dtype=int)]
        actmo, actocc, e_o, act_orb_sym = psort(ova, fav, pav, actmo, act_orb_sym)

else:

    print('split localization at', split_low, '~', split_high)
    assert do_loc
    assert split_high >= split_low
    actmo = coeff[:, np.array(cas_list, dtype=int)]
    if loc_with_pg:
        act_orb_sym = orb_sym[np.array(cas_list, dtype=int)]
    actocc = mo_occ[np.array(cas_list, dtype=int)]
    print('active occ = ', np.sum(actocc, axis=-1), actocc)
    lidx = actocc <= split_low
    midx = (actocc > split_low) & (actocc <= split_high)
    hidx = actocc > split_high

    if len(actmo[:, lidx]) != 0:
        print('low orbs = ', np.array(list(range(len(lidx))))[lidx])
        if not loc_with_pg:
            ierr, ua = loc(mol, actmo[:, lidx])
            actmo[:, lidx] = actmo[:, lidx].dot(ua)
            actmo[:, lidx], actocc[lidx], _, _ = psort(ova, fav, pav, actmo[:, lidx], None)
        else:
            ierr, ua = loc_pg(mol, actmo[:, lidx], act_orb_sym[lidx])
            actmo[:, lidx] = actmo[:, lidx].dot(ua)
            actmo[:, lidx], actocc[lidx], _, act_orb_sym[lidx] = psort(ova, fav, pav, actmo[:, lidx], act_orb_sym[lidx])

    if len(actmo[:, midx]) != 0:
        print('mid orbs = ', np.array(list(range(len(midx))))[midx])
        if not loc_with_pg:
            ierr, ua = loc(mol, actmo[:, midx])
            actmo[:, midx] = actmo[:, midx].dot(ua)
            actmo[:, midx], actocc[midx], _, _ = psort(ova, fav, pav, actmo[:, midx])
        else:
            ierr, ua = loc_pg(mol, actmo[:, midx], act_orb_sym[midx])
            actmo[:, midx] = actmo[:, midx].dot(ua)
            actmo[:, midx], actocc[midx], _, act_orb_sym[midx] = psort(ova, fav, pav, actmo[:, midx], act_orb_sym[midx])


    if len(actmo[:, hidx]) != 0:
        print('high orbs = ', np.array(list(range(len(hidx))))[hidx])
        if not loc_with_pg:
            ierr, ua = loc(mol, actmo[:, hidx])
            actmo[:, hidx] = actmo[:, hidx].dot(ua)
            actmo[:, hidx], actocc[hidx], _, _ = psort(ova, fav, pav, actmo[:, hidx])
        else:
            ierr, ua = loc_pg(mol, actmo[:, hidx], act_orb_sym[hidx])
            actmo[:, hidx] = actmo[:, hidx].dot(ua)
            actmo[:, hidx], actocc[hidx], _, act_orb_sym[hidx] = psort(ova, fav, pav, actmo[:, hidx], act_orb_sym[hidx])

coeff[:, np.array(sorted(cas_list), dtype=int)] = actmo
mo_occ[np.array(sorted(cas_list), dtype=int)] = actocc
if loc_with_pg:
    orb_sym[np.array(sorted(cas_list), dtype=int)] = act_orb_sym

# sort_mo from pyscf.mcscf.addons

cas_list = np.array(sorted(cas_list), dtype=int)
mask = np.ones(coeff.shape[1], dtype=bool)
mask[cas_list] = False
idx = np.where(mask)[0]
nactorb = len(cas_list)
nactelec = int(np.round(sum(mo_occ[cas_list])) + 0.1)
assert (mol.nelectron - nactelec) % 2 == 0
ncore = (mol.nelectron - nactelec) // 2
print("NACTORB = %d NACTELEC = %d NCORE = %d" % (nactorb, nactelec, ncore))
coeff = np.hstack((coeff[:, idx[:ncore]], coeff[:, cas_list], coeff[:, idx[ncore:]]))
print('lo occ =', mo_occ[cas_list])
mo_occ = np.hstack((mo_occ[idx[:ncore]], mo_occ[cas_list], mo_occ[idx[ncore:]]))
if loc_with_pg:
    print('loc orb_sym =', orb_sym[cas_list])
    orb_sym = np.hstack((orb_sym[idx[:ncore]], orb_sym[cas_list], orb_sym[idx[ncore:]]))

np.save("lo_coeff.npy", coeff)
np.save("lo_occ.npy", mo_occ)
np.save("active_space.npy", (nactorb, nactelec))
if loc_with_pg:
    np.save("lo_orb_sym.npy", orb_sym)
"""

def write(fn, pma):
    with open(fn, "w") as f:

        f.write(TIME_ST)

        lde = pma["load_mf"]
        if "/" not in lde:
            lde = "../" + lde

        f.write(MF_LOAD % (lde + "/mf.chk"))

        lde = pma["load_coeff"]
        if "/" not in lde:
            lde = "../" + lde

        f.write("lde = '%s'\n" % lde)
        if "cas_list" in pma:
            f.write("cas_list = %s\n" % pma["cas_list"])
        else:
            f.write("cas_list = None\n")

        if "nactorb" in pma:
            f.write("nactorb = %s\n" % pma["nactorb"])

        if "nactelec" in pma:
            f.write("nactelec = %s\n" % pma["nactelec"])

        if "split_low" in pma:
            f.write("split_low = %s\n" % pma["split_low"])

        if "split_high" in pma:
            f.write("split_high = %s\n" % pma["split_high"])

        if "alpha" in pma:
            f.write("alpha = True\n")

        if "beta" in pma:
            f.write("beta = True\n")

        if "uno" in pma:
            f.write("uno = True\n")

        if "average_occ" in pma:
            f.write("average_occ = True\n")

        if "loc_with_pg" in pma:
            f.write("loc_with_pg = True\n")

        f.write("do_loc = %s\n" % (False if "no_loc" in pma else True))

        f.write(PM_LOC)
        f.write(SELECT)

        if "uno" not in pma or "nactorb" in pma or "nactelec" in pma or "cas_list" in pma:
            f.write(SELECT2)

        f.write(TIME_ED)
