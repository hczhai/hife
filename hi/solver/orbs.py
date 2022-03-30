
from .mol import TIME_ST, TIME_ED

MF_LOAD = """
from pyscf import scf
mfchk = "%s"
mol, mfx = scf.chkfile.load_scf(mfchk)

idx_a = None
idx_b = None
"""

WRITE_ORBS_LOAD = """
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
"""

WRITE_ORBS = """
if not os.path.exists("./orbs"):
    os.makedirs("./orbs")

assert coeff.ndim == 2 and mo_occ.ndim == 1

norb = coeff.shape[1]
tools.molden.from_mo(mol, 'orbs.molden', coeff)

if idx_a is None:
    idx_a = 0

if idx_b is None:
    idx_b = norb

def sqrtm(s):
    e, v = np.linalg.eigh(s)
    return np.dot(v * np.sqrt(e), v.T.conj())

ova = mol.intor_symmetric("cint1e_ovlp_sph")
s12 = sqrtm(ova)
lcoeff = s12.dot(coeff)
diff = lcoeff.T @ lcoeff - np.identity(norb)
assert np.linalg.norm(diff) < 1E-7

labels = mol.ao_labels(None)
texts = [None] * norb
for iorb in range(norb):
    vec = lcoeff[:, iorb] ** 2
    ivs = np.argsort(vec)
    x = "V" if mo_occ[iorb] < 0.5 else ("A" if mo_occ[iorb] <= 1.5 else "C")
    text = "[%s %3d] occ=%.5f\\n" % (x, iorb, mo_occ[iorb])
    for iao in ivs[::-1][:3]:
        text += "(%d-%s-%s = %.3f) " % (labels[iao][0], labels[iao][1],
            labels[iao][2] + labels[iao][3], vec[iao])
    print(text.replace("\\n", " "))
    texts[iorb] = text

jmol_script = 'orbs.spt'
fspt = open('orbs.spt', 'w')
fspt.write('''
initialize;
set background [xffffff];
set frank off
set autoBond true;
set bondRadiusMilliAngstroms 66;
set bondTolerance 0.5;
set forceAutoBond false;
''')
fspt.write('load orbs.molden;\\n')
fspt.write('rotate -30 y;\\n')
fspt.write('rotate 60 z;\\n')
fspt.write('zoom 130;\\n')
fspt.write('translate x 2;\\n')
for i in range(idx_a, idx_b):
    fspt.write('isoSurface MO %d fill noMesh noDots;\\n' % (i+1))
    fspt.write('color isoSurface translucent 0.6 [x3300cc];\\n')
    fspt.write('color isoSurface phase [x3E92CC] [xFFB238];\\n')
    fspt.write('set echo top left; echo "%s"; color echo gray;\\n' % texts[i])
    fspt.write('font echo 14 monospaced;\\n')
    fspt.write('write JPG 90 "orbs/%s-%03d.jpg";\\n' % ('orbs.spt', i))
fspt.write('exitjmol\\n')
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

        f.write(WRITE_ORBS_LOAD)

        if "alpha" in pma:
            f.write("coeff = coeff[0]\n")
            f.write("mo_occ = mo_occ[0]\n")

        if "beta" in pma:
            f.write("coeff = coeff[1]\n")
            f.write("mo_occ = mo_occ[1]\n")

        if "from" in pma:
            f.write("idx_a = %s\n" % pma["from"])

        if "to" in pma:
            f.write("idx_b = %s\n" % pma["to"])

        f.write(WRITE_ORBS)
        f.write(TIME_ED)
