
from .mol import TIME_ST, TIME_ED

MF_LOAD = """
from pyscf import scf
import numpy as np
mfchk = "%s"
mol, mfx = scf.chkfile.load_scf(mfchk)
x2c = %s
nactorb = None
nactelec = None
"""

CASCI = """
from pyscf import mcscf
import os

for fname in ["mo_coeff.npy", "lo_coeff.npy", "nat_coeff.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        coeff = np.load(lde + "/" + fname)
        break

if nactelec is None or nactelec is None:
    print("use: " + lde + "/active_space.npy")
    nactorb, nactelec = np.load(lde + "/active_space.npy")

print("act: orb = %%d elec = %%d spin = %%d" %% (nactorb, nactelec, spin))

nacta = (nactelec + spin) // 2
nactb = (nactelec - spin) // 2

if coeff.ndim == 3:
    print('use UHF')
    mf = scf.UHF(mol)
else:
    print('use RHF')
    mf = scf.RHF(mol)
if x2c:
    mf = scf.sfx2c(mf)

mf.chkfile = "mf.chk"
mf.mo_coeff = coeff

mc = mcscf.%s(mf, nactorb, (nacta, nactb))
mc.fcisolver.conv_tol = %s
mc.natorb = True
mc.conv_tol = %s
mc.max_cycle_macro = %s
mc.kernel()
dmao = mc.make_rdm1()
"""

ALL_FINAL = """
import numpy as np
coeff_inv = np.linalg.pinv(mc.mo_coeff)
dmmo = np.einsum('...ip,...pq,...jq->...ij', coeff_inv, dmao, coeff_inv)
if dmmo[0].ndim == 2:
    mc_occ = np.diag(dmmo[0]) + np.diag(dmmo[1])
else:
    mc_occ = np.diag(dmmo)

np.save("mc_occ.npy", mc_occ)
np.save("mc_mo_coeff.npy", mc.mo_coeff)
np.save("mc_e_tot.npy", mc.e_tot)
np.save("mc_dmao.npy", dmao)
np.save("mc_dmmo.npy", dmmo)

nat_occ, u = np.linalg.eigh(dmmo)
nat_coeff = np.einsum('...pi,...ij->...pj', mc.mo_coeff, u, optimize=True)
np.save("nat_coeff.npy", nat_coeff[..., ::-1])
np.save("nat_occ.npy", nat_occ[..., ::-1])
np.save("active_space.npy", (nactorb, nactelec))
np.save("spin.npy", (spin, ))
"""

def write(fn, pmc, pmf, is_casci=True):
    with open(fn, "w") as f:

        f.write(TIME_ST)

        lde = pmc["load_mf"]
        if "/" not in lde:
            lde = "../" + lde

        f.write(MF_LOAD % (lde + "/mf.chk", "x2c" in pmf))

        lde = pmc["load_coeff"]
        if "/" not in lde:
            lde = "../" + lde

        f.write("lde = '%s'\n" % lde)

        if "nactorb" in pmc:
            f.write("nactorb = %s\n" % pmc["nactorb"])

        if "nactelec" in pmc:
            f.write("nactelec = %s\n" % pmc["nactelec"])

        if "spin" in pmc:
            f.write("spin = %s\n" % pmc["spin"])
        else:
            f.write("spin = %s\n" % pmf["spin"])

        f.write(CASCI % (
            "CASCI" if is_casci else "CASSCF",
            pmc["fci_conv_tol"],
            pmc["conv_tol"],
            pmc["max_cycle"])
        )

        f.write(ALL_FINAL)
        f.write(TIME_ED)
