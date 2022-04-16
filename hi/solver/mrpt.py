
from .mol import TIME_ST, TIME_ED

MF_LOAD = """
from pyscf import scf
import numpy as np
mfchk = "%s"
mol, mfx = scf.chkfile.load_scf(mfchk)
x2c = %s
nactorb = None
nactelec = None
spin = None
fix_spin = False
"""

CASCI = """
from pyscf import mcscf
import os

mcscf.casci.FRAC_OCC_THRESHOLD = %s

for fname in ["mo_coeff.npy", "lo_coeff.npy", "mc_mo_coeff.npy", "nat_coeff.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        coeff = np.load(lde + "/" + fname)
        break

if nactelec is None or nactelec is None:
    print("use: " + lde + "/active_space.npy")
    nactorb, nactelec = np.load(lde + "/active_space.npy")

if spin is None:
    print("use: " + lde + "/spin.npy")
    spin, = np.load(lde + "/spin.npy")

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

mc = mcscf.CASCI(mf, nactorb, (nacta, nactb))
if fix_spin:
    from pyscf import fci
    mc.fcisolver = fci.addons.fix_spin(mc.fcisolver, 0.2, spin * (spin + 2) / 4.0)
mc.fcisolver.conv_tol = %s
mc.canonicalization = True
mc.natorb = True
mc.kernel()
"""

SCNEVPT2 = """
from pyscf import mrpt
sc = mrpt.NEVPT(mc).set(canonicalized=True).run()
"""

ICNEVPT2 = """
from pyblock2.icmr.icnevpt2_full import ICNEVPT2
ic = ICNEVPT2(mc).run()
"""

ICMRREPT2 = """
from pyblock2.icmr.icmrrept2_full import ICMRREPT2
ic = ICMRREPT2(mc).run()
"""

ICMRLCC = """
from pyscf import lib, fci
mc.fcisolver.scratchDirectory = lib.param.TMPDIR
mc.fcisolver.memory = int(mol.max_memory / 1000)
def make_rdm4(**kwargs):
    dms = fci.rdm.make_dm1234('FCI4pdm_kern_sf', mc.ci, mc.ci, mc.ncas, mc.nelecas)
    E1, E2, E3, E4 = [np.zeros_like(dm) for dm in dms]
    deltaAA = np.eye(mc.ncas)
    E1 += np.einsum('pa->pa', dms[0], optimize=True)
    E2 += np.einsum('paqb->pqab', dms[1], optimize=True)
    E2 += -1 * np.einsum('aq,pb->pqab', deltaAA, E1, optimize=True)
    E3 += np.einsum('paqbgc->pqgabc', dms[2], optimize=True)
    E3 += -1 * np.einsum('ag,pqcb->pqgabc', deltaAA, E2, optimize=True)
    E3 += -1 * np.einsum('aq,pgbc->pqgabc', deltaAA, E2, optimize=True)
    E3 += -1 * np.einsum('bg,pqac->pqgabc', deltaAA, E2, optimize=True)
    E3 += -1 * np.einsum('aq,bg,pc->pqgabc', deltaAA, deltaAA, E1, optimize=True)
    E4 += np.einsum('aebfcgdh->abcdefgh', dms[3], optimize=True)
    E4 += -1 * np.einsum('eb,acdfgh->abcdefgh', deltaAA, E3, optimize=True)
    E4 += -1 * np.einsum('ec,abdgfh->abcdefgh', deltaAA, E3, optimize=True)
    E4 += -1 * np.einsum('ed,abchfg->abcdefgh', deltaAA, E3, optimize=True)
    E4 += -1 * np.einsum('fc,abdegh->abcdefgh', deltaAA, E3, optimize=True)
    E4 += -1 * np.einsum('fd,abcehg->abcdefgh', deltaAA, E3, optimize=True)
    E4 += -1 * np.einsum('gd,abcefh->abcdefgh', deltaAA, E3, optimize=True)
    E4 += -1 * np.einsum('eb,fc,adgh->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
    E4 += -1 * np.einsum('eb,fd,achg->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
    E4 += -1 * np.einsum('eb,gd,acfh->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
    E4 += -1 * np.einsum('ec,fd,abgh->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
    E4 += -1 * np.einsum('ec,gd,abhf->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
    E4 += -1 * np.einsum('ed,fc,abhg->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
    E4 += -1 * np.einsum('fc,gd,abeh->abcdefgh', deltaAA, deltaAA, E2, optimize=True)
    E4 += -1 * np.einsum('eb,fc,gd,ah->abcdefgh', deltaAA, deltaAA, deltaAA, E1, optimize=True)
    return E4
mc.fcisolver.make_rdm4 = make_rdm4
from pyscf.icmpspt import icmpspt
e_corr = icmpspt.mrlcc(mc, nfro=0)
lib.logger.note(mc, 'E(ICMRLCC) = %.16g  E_corr_pt = %.16g', mc.e_tot + e_corr, e_corr)
"""

SCNEVPT2_B2 = """
from pyblock2.icmr.scnevpt2 import SCNEVPT2
ic = SCNEVPT2(mc).run()
"""

def write(fn, pmc, pmf):
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
            f.write("spin = None\n")
        
        if "fix_spin" in pmc:
            f.write("fix_spin = True\n")

        f.write(CASCI % (
            pmc["frac_occ_tol"],
            pmc["fci_conv_tol"])
        )

        if pmc["method"] == "sc-nevpt2":
            if "solver" in pmc and pmc["solver"] == "block2":
                f.write(SCNEVPT2_B2)
            else:
                f.write(SCNEVPT2)
        elif pmc["method"] == "ic-nevpt2":
            f.write(ICNEVPT2)
        elif pmc["method"] == "ic-mrrept2":
            f.write(ICMRREPT2)
        elif pmc["method"] == "ic-mrlcc":
            f.write(ICMRLCC)
        else:
            raise RuntimeError("Unknown mrpt method: %s" % pmc["method"])

        f.write(TIME_ED)
