
from .mol import TIME_ST, TIME_ED

MF_LOAD = """
from pyscf import scf, lib
import numpy as np
mfchk = "%s"
mol, mfx = scf.chkfile.load_scf(mfchk)
if spin is not None:
    mol.spin = spin
    mol.build()
mf = %s
mf.chkfile = "mf.chk"
mf.mo_coeff = mfx["mo_coeff"]
mf.mo_energy = mfx["mo_energy"]
mf.mo_occ = mfx["mo_occ"]

do_ccsd_t = True
bcc = False
"""

CC_LOAD_COEFF = """
import numpy as np
import os

for fname in ["mo_coeff.npy", "lo_coeff.npy", "mc_mo_coeff.npy", "nat_coeff.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        coeff = np.load(lde + "/" + fname)
        break

for fname in ["mf_occ.npy", "lo_occ.npy", "mc_occ.npy", "nat_occ.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        mo_occ = np.load(lde + "/" + fname)
        break

print('pre  occ adjust', np.sum(mo_occ, axis=-1))
mo_occ = np.round(mo_occ / (3 - mo_occ.ndim)) * (3 - mo_occ.ndim)
print('post occ adjust', np.sum(mo_occ, axis=-1), mo_occ)

mf.mo_coeff = coeff
mf.mo_occ = mo_occ
mf.mo_energy = None
mf.e_tot = mf.energy_tot()
print('ref energy = ', mf.e_tot)
"""

CC_PRE = """
from sys import argv
import os
is_restart = len(argv) >= 2 and argv[1] == "1"

if not is_restart:
    for fname in ['/ccdiis.h5', '/ccdiis-lambda.h5']:
        if os.path.isfile(lib.param.TMPDIR + fname):
            fid = 1
            while os.path.isfile(lib.param.TMPDIR + fname + '.%d' % fid):
                fid += 1
            os.rename(lib.param.TMPDIR + fname,
                lib.param.TMPDIR + fname + '.%d' % fid)
"""

CC = """
print('mf occ', np.sum(mf.mo_occ, axis=-1), mf.mo_occ)

from pyscf import cc
mc = cc.CCSD(mf)
mc.diis_file = lib.param.TMPDIR + '/ccdiis.h5'
mc.max_cycle = %s
"""

CC_FROZEN = """
print('mf occ', np.sum(mf.mo_occ, axis=-1), mf.mo_occ)

from pyscf import cc
mc = cc.CCSD(mf, frozen=%s)
mc.diis_file = lib.param.TMPDIR + '/ccdiis.h5'
mc.max_cycle = %s
"""

CC_FINAL = """
if is_restart and os.path.isfile(lib.param.TMPDIR + '/ccdiis.h5'):
    print("restart ccsd from ", lib.param.TMPDIR + '/ccdiis.h5')
    mc.restore_from_diis_(lib.param.TMPDIR + '/ccdiis.h5')
    t1, t2 = mc.t1, mc.t2
    mc.kernel(t1, t2)
else:
    mc.kernel()
e_ccsd = mc.e_tot
print('ECCSD    = ', e_ccsd)
print("PART TIME (CCSD) = %20.3f" % (time.perf_counter() - txst))

if bcc:
    from libdmet.solver.cc import bcc_loop
    mc = bcc_loop(mc, utol=bcc_conv_tol, max_cycle=bcc_max_cycle, verbose=mol.verbose)
    e_bccsd = mc.e_tot
    print('EBCCSD   = ', e_bccsd)
    print("PART TIME (BCCSD) = %20.3f" % (time.perf_counter() - txst))

if do_ccsd_t:
    e_ccsd_t = mc.e_tot + mc.ccsd_t()
    print('ECCSD(T) = ', e_ccsd_t)
    print("PART TIME (CCSD(T))  = %20.3f" % (time.perf_counter() - txst))

mc.diis_file = lib.param.TMPDIR + '/ccdiis-lambda.h5'
if is_restart and os.path.isfile(lib.param.TMPDIR + '/ccdiis-lambda.h5'):
    print("restart ccsd-lambda from ", lib.param.TMPDIR + '/ccdiis-lambda.h5')
    from pyscf import lib
    ccvec = lib.diis.restore(lib.param.TMPDIR + '/ccdiis-lambda.h5').extrapolate()
    l1, l2 = mc.vector_to_amplitudes(ccvec)
    mc.restore_from_diis_(lib.param.TMPDIR + '/ccdiis-lambda.h5')
    mc.solve_lambda(mc.t1, mc.t2, l1, l2)
else:
    mc.solve_lambda(mc.t1, mc.t2)
print("PART TIME (CCSD-lambda)  = %20.3f" % (time.perf_counter() - txst))

import numpy as np
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

print('nat occ', np.sum(nat_occ, axis=-1), nat_occ)
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

        if "spin" in pmc:
            f.write("spin = %s\n" % pmc["spin"])
        else:
            f.write("spin = None\n")

        f.write(MF_LOAD % (lde + "/mf.chk", mme))

        if "no_ccsd_t" in pmc:
            f.write("do_ccsd_t = False\n")

        if "KS" in mme:
            if "U" in mme:
                f.write("mfhf = scf.UHF(mol)\n")
            else:
                f.write("mfhf = scf.RHF(mol)\n")
            if "x2c" in pmf:
                f.write("mfhf = scf.sfx2c(mfhf)\n")
            f.write("mfhf.__dict__.update(mf.__dict__)\n")
            f.write("mf = mfhf\n")

        if "load_coeff" in pmc:
            lde = pmc["load_coeff"]
            if "/" not in lde:
                lde = "../" + lde

            f.write("lde = '%s'\n" % lde)

            f.write(CC_LOAD_COEFF)

        if "bcc" in pmc:
            f.write("bcc = True\n")
            f.write("bcc_conv_tol = %s\n" % pmc["bcc_conv_tol"])
            f.write("bcc_max_cycle = %s\n" % pmc["bcc_max_cycle"])

        f.write(CC_PRE)

        if "frozen" in pmc:
            f.write(CC_FROZEN % (pmc["frozen"], pmc["max_cycle"]))
        else:
            f.write(CC % (pmc["max_cycle"]))

        if "level_shift" in pmc:
            f.write("mc.level_shift = %s\n" % pmc["level_shift"])

        f.write(CC_FINAL)
        f.write(ALL_FINAL)
        f.write(TIME_ED)
