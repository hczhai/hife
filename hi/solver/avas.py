
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

AVAS = """
import numpy as np
from pyscf.mcscf import avas
ao_labels = %s
threshold = %s
nactorb, nactelec, coeff = avas.avas(mf, ao_labels,
    canonicalize=False, threshold=threshold)
ncore = (mol.nelectron - nactelec) // 2
print("NACTORB = %%d NACTELEC = %%d NCORE = %%d" %% (nactorb, nactelec, ncore))

dmao = mf.make_rdm1()
if dmao[0].ndim == 2:
    dmao = dmao[0] + dmao[1]
coeff_inv = np.linalg.pinv(coeff)
dmmo = np.einsum('ip,pq,jq->ij', coeff_inv, dmao, coeff_inv, optimize=True)
mo_occ = np.diag(dmmo)

norb = coeff.shape[1]

def sqrtm(s):
    e, v = np.linalg.eigh(s)
    return np.dot(v * np.sqrt(e), v.T.conj())

ova = mol.intor_symmetric("cint1e_ovlp_sph")
s12 = sqrtm(ova)
lcoeff = s12.dot(coeff)
diff = lcoeff.T @ lcoeff - np.identity(norb)
assert np.linalg.norm(diff) < 1E-7

labels = mol.ao_labels(None)
for iorb in range(norb):
    vec = lcoeff[:, iorb] ** 2
    ivs = np.argsort(vec)
    x = "V" if mo_occ[iorb] < 0.5 else ("A" if mo_occ[iorb] <= 1.5 else "C")
    text = "[%%s %%3d] occ=%%.5f\\n" %% (x, iorb, mo_occ[iorb])
    for iao in ivs[::-1][:3]:
        text += "(%%d-%%s-%%s = %%.3f) " %% (labels[iao][0], labels[iao][1],
            labels[iao][2] + labels[iao][3], vec[iao])
    print(text.replace("\\n", " "))

np.save("lo_coeff.npy", coeff)
np.save("lo_occ.npy", mo_occ)
np.save("active_space.npy", (nactorb, nactelec))
"""

def write(fn, pmc, pmf):
    with open(fn, "w") as f:

        f.write(TIME_ST)

        if "dftd3" in pmf:
            f.write("from pyscf import dftd3\n")

        def xmethod(method, x2c):
            if method == "uhf" or method == "uks":
                r = "scf.UHF(mol)"
            elif method == "rhf" or method == "rks":
                r = "scf.RHF(mol)"
            else:
                raise RuntimeError("Unknown mf method %s!" % method)
            r = "scf.sfx2c(%s)" % r if x2c else r
            return r

        mme = xmethod(pmf["method"], "x2c" in pmf)
        lde = pmc["load_mf"]
        if "/" not in lde:
            lde = "../" + lde

        f.write(MF_LOAD % (lde + "/mf.chk", mme))

        if "dftd3" in pmf:
            f.write("mf = dftd3.dftd3(mf)\n")

        f.write(AVAS % (
            pmc["ao_labels"].split(";"),
            pmc["threshold"]))

        f.write(TIME_ED)
