
from .mol import TIME_ST, TIME_ED
from .active import PM_LOC

MF_LOAD = """
from pyscf import scf
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

for fname in ["mf_occ.npy", "lo_occ.npy", "nat_coeff.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        mo_occ = np.load(lde + "/" + fname)
        break

def psort(ova, fav, pT, coeff):
   pTnew = 2.0 * (coeff.T @ ova @ pT @ ova @ coeff)
   nocc  = np.diag(pTnew)
   index = np.argsort(-nocc)
   ncoeff = coeff[:, index]
   nocc = nocc[index]
   enorb = np.diag(coeff.T @ ova @ fav @ ova @ coeff)
   enorb = enorb[index]
   return ncoeff, nocc, enorb

actmo = coeff[:, np.array(cas_list, dtype=int)]
if do_loc:
    ierr, ua = loc(mol, actmo)
    actmo = actmo.dot(ua)
actmo, n_o, e_o = psort(ova, fav, pav, actmo)

coeff[:, np.array(sorted(cas_list), dtype=int)] = actmo
mo_occ[np.array(sorted(cas_list), dtype=int)] = n_o

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
mo_occ = np.hstack((mo_occ[idx[:ncore]], mo_occ[cas_list], mo_occ[idx[ncore:]]))

np.save("lo_coeff.npy", coeff)
np.save("lo_occ.npy", mo_occ)
np.save("active_space.npy", (nactorb, nactelec))
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
        f.write("cas_list = %s\n" % pma["cas_list"])
        f.write("do_loc = %s\n" % (False if "no_loc" in pma else True))

        f.write(PM_LOC)
        f.write(SELECT)
        f.write(TIME_ED)
