
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
mc.fcisolver.conv_tol = %s
mc.kernel()
"""

SCNEVPT2 = """
from pyscf import mrpt
sc = mrpt.NEVPT(mc).run()
"""

ICNEVPT2 = """
from pyblock2.icmr.icnevpt2_full import ICNEVPT2
ic = ICNEVPT2(mc).run()
"""

ICMRREPT2 = """
from pyblock2.icmr.icmrrept2_full import ICMRREPT2
ic = ICMRREPT2(mc).run()
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

        f.write(CASCI % pmc["fci_conv_tol"])

        if pmc["method"] == "sc-nevpt2":
            if "solver" in pmc and pmc["solver"] == "block2":
                f.write(SCNEVPT2_B2)
            else:
                f.write(SCNEVPT2)
        elif pmc["method"] == "ic-nevpt2":
            f.write(ICNEVPT2)
        elif pmc["method"] == "ic-mrrept2":
            f.write(ICMRREPT2)
        else:
            raise RuntimeError("Unknown mrpt method: %s" % pmc["method"])

        f.write(TIME_ED)
