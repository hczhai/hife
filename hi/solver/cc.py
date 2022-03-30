
from .mol import TIME_ST, TIME_ED

MF_LOAD = """
from pyscf import scf
mfchk = "%s"
mol, mfx = scf.chkfile.load_scf(mfchk)
mf = %s
mf.chkfile = "mf.chk"
mf.mo_coeff = mfx["mo_coeff"]
mf.mo_energy = mfx["mo_energy"]
mf.mo_occ = mfx["mo_occ"]
"""

CC = """
from pyscf import cc
mc = cc.CCSD(mf)
mc.max_cycle = %s
"""

CC_FINAL = """
mc.kernel()
e_ccsd = mc.e_tot
print('ECCSD    = ', e_ccsd)
print("PART TIME (CCSD) = %20.3f" % (time.perf_counter() - txst))

e_ccsd_t = e_ccsd + mc.ccsd_t()
print('ECCSD(T) = ', e_ccsd_t)
print("PART TIME (CCSD(T))  = %20.3f" % (time.perf_counter() - txst))

dm = mc.make_rdm1()
if dm[0].ndim == 2:
    mc_occ = np.diag(dm[0]) + np.diag(dm[1])
else:
    mc_occ = np.diag(dm)
print("PART TIME (1PDM)  = %20.3f" % (time.perf_counter() - txst))

"""

ALL_FINAL = """
import numpy as np
np.save("cc_occ.npy", mc_occ)
np.save("cc_mo_coeff.npy", mc.mo_coeff)
np.save("cc_e_tot.npy", mc.e_tot)
np.save("cc_dmmo.npy", dm)

# dmao = np.einsum('xpi,xij,xqj->xpq', mc.mo_coeff, dm, mc.mo_coeff)
# coeff_inv = np.linalg.pinv(mc.mo_coeff)
# dmmo = np.einsum('xip,xpq,xjq->xij', coeff_inv, dmao, coeff_inv)

nat_occ, u = np.linalg.eigh(dm)
nat_coeff = np.einsum('...pi,...ij->...pj', mc.mo_coeff, u, optimize=True)
np.save("nat_coeff.npy", nat_coeff[..., ::-1])
np.save("nat_occ.npy", nat_occ[..., ::-1])
"""

def write(fn, pmc, pmf):
    with open(fn, "w") as f:

        f.write(TIME_ST)

        def xmethod(method, x2c):
            if method == "uhf":
                r = "scf.UHF(mol)"
            elif method == "uks":
                r = "scf.UKS(mol)"
            elif method == "rhf":
                r = "scf.RHF(mol)"
            elif method == "rks":
                r = "scf.RKS(mol)"
            else:
                raise RuntimeError("Unknown mf method %s!" % method)
            return "scf.sfx2c(%s)" % r if x2c else r

        mme = xmethod(pmf["method"], "x2c" in pmf)
        lde = pmc["load_mf"]
        if "/" not in lde:
            lde = "../" + lde

        f.write(MF_LOAD % (lde + "/mf.chk", mme))

        if "KS" in mme:
            if "U" in mme:
                f.write("mfhf = scf.UHF(mol)\n")
            else:
                f.write("mfhf = scf.RHF(mol)\n")
            if "x2c" in pmf:
                f.write("mfhf = scf.sfx2c(mfhf)\n")
            f.write("mfhf.__dict__.update(mf.__dict__)\n")
            f.write("mf = mfhf\n")

        f.write(CC % (pmc["max_cycle"]))

        if "level_shift" in pmc:
            f.write("mc.level_shift = %s\n" % pmc["level_shift"])

        f.write(CC_FINAL)
        f.write(ALL_FINAL)
        f.write(TIME_ED)
