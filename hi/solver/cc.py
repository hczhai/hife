
from .mol import TIME_ST, TIME_ED, handle_io_error

XCC = """
from pyscf.cc import rccsd, ccsd_lambda
from pyscf.cc import uccsd, uccsd_lambda
import numpy as np

def remove_act_amps_r(cc, t1, t2):
    if t1 is not None:
        t1[cc.norb_start:, :cc.norb_end - len(t1)] = 0
    t2[cc.norb_start:, cc.norb_start:, :cc.norb_end - len(t2), :cc.norb_end - len(t2)] = 0
    return t1, t2

def x_energy_rccsd(cc, t1, t2, eris):
    t1, t2 = remove_act_amps_r(cc, t1, t2)
    return rccsd.RCCSD.energy(cc, t1, t2, eris)

def x_update_amps_rccsd(cc, t1, t2, eris):
    t1, t2 = remove_act_amps_r(cc, t1, t2)
    t1new, t2new = rccsd.RCCSD.update_amps(cc, t1, t2, eris)
    t1new, t2new = remove_act_amps_r(cc, t1new, t2new)
    return t1new, t2new

def x_update_lambda_rccsd(cc, t1, t2, l1, l2, eris=None, imds=None):
    l1, l2 = remove_act_amps_r(cc, l1, l2)
    l1new, l2new = ccsd_lambda.update_lambda(cc, t1, t2, l1, l2, eris, imds)
    l1new, l2new = remove_act_amps_r(cc, l1new, l2new)
    return l1new, l2new

def x_init_amps_rccsd(cc, eris):
    emp2, t1, t2 = rccsd.RCCSD.init_amps(cc, eris)
    t1, t2 = remove_act_amps_r(cc, t1, t2)
    eris_ovov = np.array(eris.ovov)
    emp2 = 2 * np.einsum('ijab,iajb', t2, eris_ovov.conj(), optimize=True).real
    emp2 -=    np.einsum('jiab,iajb', t2, eris_ovov.conj(), optimize=True).real
    return emp2, t1, t2

def x_solve_lambda_rccsd(cc, t1=None, t2=None, l1=None, l2=None, eris=None):
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    cc.converged_lambda, cc.l1, cc.l2 = \\
        ccsd_lambda.kernel(cc, eris, t1, t2, l1, l2,
                            max_cycle=cc.max_cycle,
                            tol=cc.conv_tol_normt,
                            verbose=cc.verbose,
                            fupdate=x_update_lambda_rccsd)
    return cc.l1, cc.l2

class XRCCSD(rccsd.RCCSD):
    def __init__(self, mf, nactorb, nactelec, **kwargs):
        nce = (mf.mol.nelectron - nactelec) // 2
        self.norb_start = nce
        self.norb_end = self.norb_start + nactorb
        self.nactelec = nactelec
        self.nactorb = nactorb
        rccsd.RCCSD.__init__(self, mf, **kwargs)
    energy = x_energy_rccsd
    update_amps = x_update_amps_rccsd
    init_amps = x_init_amps_rccsd
    solve_lambda = x_solve_lambda_rccsd

def remove_act_amps_u(cc, t1, t2):
    if t1 is not None:
        t1a, t1b = t1
        lac, lbc = len(t1a), len(t1b)
        t1a[cc.norb_start:, :cc.norb_end - lac] = 0
        t1b[cc.norb_start:, :cc.norb_end - lbc] = 0
        t1 = t1a, t1b
    t2aa, t2ab, t2bb = t2
    lac, lbc = len(t2aa), len(t2bb)
    t2aa[cc.norb_start:, cc.norb_start:, :cc.norb_end - lac, :cc.norb_end - lac] = 0
    t2ab[cc.norb_start:, cc.norb_start:, :cc.norb_end - lac, :cc.norb_end - lbc] = 0
    t2bb[cc.norb_start:, cc.norb_start:, :cc.norb_end - lbc, :cc.norb_end - lbc] = 0
    t2 = t2aa, t2ab, t2bb
    return t1, t2

def x_energy_uccsd(cc, t1, t2, eris):
    t1, t2 = remove_act_amps_u(cc, t1, t2)
    return uccsd.UCCSD.energy(cc, t1, t2, eris)

def x_update_amps_uccsd(cc, t1, t2, eris):
    t1, t2 = remove_act_amps_u(cc, t1, t2)
    t1new, t2new = uccsd.UCCSD.update_amps(cc, t1, t2, eris)
    t1new, t2new = remove_act_amps_u(cc, t1new, t2new)
    return t1new, t2new

def x_update_lambda_uccsd(cc, t1, t2, l1, l2, eris=None, imds=None):
    l1, l2 = remove_act_amps_u(cc, l1, l2)
    l1new, l2new = uccsd_lambda.update_lambda(cc, t1, t2, l1, l2, eris, imds)
    l1new, l2new = remove_act_amps_u(cc, l1new, l2new)
    return l1new, l2new

def x_init_amps_uccsd(cc, eris):
    emp2, t1, t2 = uccsd.UCCSD.init_amps(cc, eris)
    t1, t2 = remove_act_amps_u(cc, t1, t2)
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    t2aa, t2ab, t2bb = t2
    e  =      np.einsum('iJaB,iaJB', t2ab, eris_ovOV, optimize=True)
    e += 0.25*np.einsum('ijab,iajb', t2aa, eris_ovov, optimize=True)
    e -= 0.25*np.einsum('ijab,ibja', t2aa, eris_ovov, optimize=True)
    e += 0.25*np.einsum('ijab,iajb', t2bb, eris_OVOV, optimize=True)
    e -= 0.25*np.einsum('ijab,ibja', t2bb, eris_OVOV, optimize=True)
    emp2 = e.real
    return emp2, t1, t2

def x_solve_lambda_uccsd(cc, t1=None, t2=None, l1=None, l2=None, eris=None):
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo(cc.mo_coeff)
    cc.converged_lambda, cc.l1, cc.l2 = \\
        ccsd_lambda.kernel(cc, eris, t1, t2, l1, l2,
                            max_cycle=cc.max_cycle,
                            tol=cc.conv_tol_normt,
                            verbose=cc.verbose,
                            fintermediates=uccsd_lambda.make_intermediates,
                            fupdate=x_update_lambda_uccsd)
    return cc.l1, cc.l2
    
class XUCCSD(uccsd.UCCSD):
    def __init__(self, mf, nactorb, nactelec, **kwargs):
        if not isinstance(nactelec, tuple):
            nactelec = nactelec - nactelec // 2, nactelec // 2
        nae = nactelec[0] + nactelec[1]
        nce = mf.mol.nelectron - nae
        self.norb_start = nce // 2
        self.norb_end = self.norb_start + nactorb
        self.nactelec = nactelec
        uccsd.UCCSD.__init__(self, mf, **kwargs)
    energy = x_energy_uccsd
    update_amps = x_update_amps_uccsd
    init_amps = x_init_amps_uccsd
    solve_lambda = x_solve_lambda_uccsd
"""

MF_LOAD = """
from pyscf import scf, lib, symm
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
do_spin_square = False
nat_with_pg = False
save_amps = False
xcc_nelec = None
xcc_ncas = None
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

if xcc_ncas is not None:
    if isinstance(mf, scf.uhf.UHF):
        xna = (xcc_nelec + spin) // 2
        xnb = (xcc_nelec - spin) // 2
        mc = XUCCSD(mf, xcc_ncas, (xna, xnb))
    else:
        mc = XRCCSD(mf, xcc_ncas, xcc_nelec)
else:
    mc = cc.CCSD(mf)

mc.diis_file = lib.param.TMPDIR + '/ccdiis.h5'
mc.max_cycle = %s
"""

CC_FROZEN = """
print('mf occ', np.sum(mf.mo_occ, axis=-1), mf.mo_occ)

from pyscf import cc, scf

nfrozen = %s
if xcc_ncas is not None:
    if isinstance(mf, scf.uhf.UHF):
        xna = (xcc_nelec + spin) // 2
        xnb = (xcc_nelec - spin) // 2
        mc = XUCCSD(mf, xcc_ncas, (xna, xnb), frozen=nfrozen)
    else:
        mc = XRCCSD(mf, xcc_ncas, xcc_nelec, frozen=nfrozen)
else:
    mc = cc.CCSD(mf, frozen=nfrozen)
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

if save_amps:
    np.save("ccsd_t1.npy", mc.t1)
    np.save("ccsd_t2.npy", mc.t2)

if do_spin_square:
    S2 = mc.spin_square()[0]
    print('CCSD <S^2> = ', S2)
    print("PART TIME (CCSD S2) = %20.3f" % (time.perf_counter() - txst))

if bcc:
    from libdmet.solver.cc import bcc_loop
    mc = bcc_loop(mc, utol=bcc_conv_tol, max_cycle=bcc_max_cycle, verbose=mol.verbose)
    e_bccsd = mc.e_tot
    print('EBCCSD   = ', e_bccsd)
    print("PART TIME (BCCSD) = %20.3f" % (time.perf_counter() - txst))

    if do_spin_square:
        S2 = mc.spin_square()[0]
        print('BCCSD <S^2> = ', S2)
        print("PART TIME (BCCSD S2) = %20.3f" % (time.perf_counter() - txst))

if do_ccsd_t:
    eris = mc.ao2mo()
    e_ccsd_t = mc.e_tot + mc.ccsd_t(eris=eris)
    print('ECCSD(T) = ', e_ccsd_t)
    print("PART TIME (CCSD(T))  = %20.3f" % (time.perf_counter() - txst))

    if do_spin_square:
        from pyscf.cc import uccsd_t_lambda, uccsd_t_rdm
        from pyscf.fci import spin_op
        conv, l1, l2 = uccsd_t_lambda.kernel(mc, tol=1E-7)
        print("PART TIME (CCSD(T) Lambda) = %20.3f" % (time.perf_counter() - txst))
        assert conv
        dm1 = uccsd_t_rdm.make_rdm1(mc, t1, t2, l1, l2, eris)
        print("PART TIME (CCSD(T) RDM1) = %20.3f" % (time.perf_counter() - txst))

        import numpy as np
        if dm1[0].ndim == 2:
            mc_occ_t = np.diag(dm1[0]) + np.diag(dm1[1])
        else:
            mc_occ_t = np.diag(dm1)

        np.save("cc_t_occ.npy", mc_occ_t)
        np.save("cc_t_mo_coeff.npy", mc.mo_coeff)
        np.save("cc_t_e_tot.npy", e_ccsd_t)
        np.save("cc_t_dmmo.npy", dm1)

        nat_occ_t, u_t = np.linalg.eigh(dm1)
        nat_coeff_t = np.einsum('...pi,...ij->...pj', mc.mo_coeff, u_t, optimize=True)
        np.save("cc_t_nat_coeff.npy", nat_coeff_t[..., ::-1])
        np.save("cc_t_nat_occ.npy", nat_coeff_t[..., ::-1])

        print('ccsd(t) nat occ', np.sum(nat_occ_t, axis=-1), nat_occ_t)

        dm2 = uccsd_t_rdm.make_rdm2(mc, t1, t2, l1, l2, eris)
        print("PART TIME (CCSD(T) RDM2) = %20.3f" % (time.perf_counter() - txst))
        S2 = spin_op.spin_square_general(*dm1, *dm2, mc.mo_coeff, mc._scf.get_ovlp())[0]
        print('CCSD(T) <S^2> = ', S2)
        print("PART TIME (CCSD(T) S2) = %20.3f" % (time.perf_counter() - txst))

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

# dmao = np.einsum('xpi,xij,xqj->xpq', mc.mo_coeff, dm, mc.mo_coeff, optimize=True)
# coeff_inv = np.linalg.pinv(mc.mo_coeff)
# dmmo = np.einsum('xip,xpq,xjq->xij', coeff_inv, dmao, coeff_inv, optimize=True)

nat_occ, u = np.linalg.eigh(dm)
nat_coeff = np.einsum('...pi,...ij->...pj', mc.mo_coeff, u, optimize=True)
np.save("nat_coeff.npy", nat_coeff[..., ::-1])
np.save("nat_occ.npy", nat_occ[..., ::-1])

print('nat occ', np.sum(nat_occ, axis=-1), nat_occ)

if nat_with_pg:
    np.save("nat_coeff_no_pg.npy", nat_coeff[..., ::-1])
    np.save("nat_occ_no_pg.npy", nat_occ[..., ::-1])

    orb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mc.mo_coeff, tol=1e-2)
    orb_sym = [symm.irrep_name2id(mol.groupname, ir) for ir in orb_sym]
    if np.array(dm).ndim == 3:
        spdm = np.sum(dm, axis=0)
    else:
        spdm = dm
    n_sites = len(spdm)
    spdm = spdm.flatten()
    nat_occ = np.zeros((n_sites, ))

    import block2 as b
    b.MatrixFunctions.block_eigs(spdm, nat_occ, b.VectorUInt8(orb_sym))
    rot = np.array(spdm.reshape((n_sites, n_sites)).T, copy=True)
    midx = np.argsort(nat_occ)[::-1]
    nat_occ = nat_occ[midx]
    rot = rot[:, midx]
    orb_sym = np.array(orb_sym)[midx]
    for isym in set(orb_sym):
        mask = np.array(orb_sym) == isym
        for j in range(len(nat_occ[mask])):
            mrot = rot[mask, :][:j + 1, :][:, mask][:, :j + 1]
            mrot_det = np.linalg.det(mrot)
            if mrot_det < 0:
                mask0 = np.arange(len(mask), dtype=int)[mask][j]
                rot[:, mask0] = -rot[:, mask0]
    nat_coeff = np.einsum('...pi,...ij->...pj', mc.mo_coeff, rot, optimize=True)
    print('nat occ =', nat_occ)
    print('nat orb_sym =', orb_sym)
    np.save("nat_coeff.npy", nat_coeff)
    np.save("nat_occ.npy", nat_occ)
    np.save("nat_orb_sym.npy", orb_sym)
"""

@handle_io_error
def write(fn, pmc, pmf):
    with open(fn, "w") as f:

        f.write(TIME_ST)

        if "dftd3" in pmf:
            f.write("from pyscf import dftd3\n")

        def xmethod(method, x2c, dftd3):
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
            r = "scf.sfx2c(%s)" % r if x2c else r
            r = "dftd3.dftd3(%s)" % r if dftd3 else r
            return r

        mme = xmethod(pmf["method"], "x2c" in pmf, "dftd3" in pmf)
        lde = pmc["load_mf"]
        if "/" not in lde:
            lde = "../" + lde

        if "spin" in pmc:
            f.write("spin = %s\n" % pmc["spin"])
        else:
            f.write("spin = None\n")

        f.write(MF_LOAD % (lde + "/mf.chk", mme))

        if "max_memory" in pmc:
            f.write("mf.max_memory = %s\n" % pmc["max_memory"])

        if "no_ccsd_t" in pmc:
            f.write("do_ccsd_t = False\n")

        if "do_spin_square" in pmc:
            f.write("do_spin_square = True\n")

        if "nat_with_pg" in pmc:
            f.write("nat_with_pg = True\n")

        if "save_amps" in pmc:
            f.write("save_amps = True\n")

        if "KS" in mme:
            if "U" in mme:
                f.write("mfhf = scf.UHF(mol)\n")
            else:
                f.write("mfhf = scf.RHF(mol)\n")
            if "x2c" in pmf:
                f.write("mfhf = scf.sfx2c(mfhf)\n")
            if "dftd3" in pmf:
                f.write("mfhf = dftd3.dftd3(mfhf)\n")
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

        if "xcc_nelec" in pmc or "xcc_ncas" in pmc:
            f.write(XCC)

        if "xcc_ncas" in pmc:
            f.write("xcc_ncas = %s\n" % pmc["xcc_ncas"])
        if "xcc_nelec" in pmc:
            f.write("xcc_nelec = %s\n" % pmc["xcc_nelec"])

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
