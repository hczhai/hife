
from .mol import TIME_ST, TIME_ED, handle_io_error

MF_LOAD = """
from pyscf import scf
import numpy as np
mfchk = "%s"
mol, mfx = scf.chkfile.load_scf(mfchk)
x2c = %s
d3 = %s
nactorb = None
nactelec = None
semi_canonical = False
cascc_mf_relax = False
"""

SEMI_CANON = """
def semi_canon(mf):
    print('doing semi canonicalization ...')

    if isinstance(mf, scf.uhf.UHF):
        ma, mb = mf.mo_coeff
        nocca = len(mf.mo_occ[0][mf.mo_occ[0] > 0])
        noccb = len(mf.mo_occ[1][mf.mo_occ[1] > 0])

        fockao_a = mf.get_fock()[0]
        fockmo_a = ma.T @ fockao_a @ ma
        foo = fockmo_a[:nocca, :nocca]
        fvv = fockmo_a[nocca:, nocca:]
        mo_coeff_occ_a = np.dot(ma[:, :nocca], np.linalg.eigh(foo)[1])
        mo_coeff_vir_a = np.dot(ma[:, nocca:], np.linalg.eigh(fvv)[1])
        mo_coeff_a = np.concatenate((mo_coeff_occ_a, mo_coeff_vir_a), axis=1)

        fockao_b = mf.get_fock()[1]
        fockmo_b = mb.T @ fockao_b @ mb
        foo = fockmo_b[:noccb, :noccb]
        fvv = fockmo_b[noccb:, noccb:]
        mo_coeff_occ_b = np.dot(mb[:, :noccb], np.linalg.eigh(foo)[1])
        mo_coeff_vir_b = np.dot(mb[:, noccb:], np.linalg.eigh(fvv)[1])
        mo_coeff_b = np.concatenate((mo_coeff_occ_b, mo_coeff_vir_b), axis=1)

        mf.mo_coeff = np.array([mo_coeff_a, mo_coeff_b])

    elif isinstance(mf, scf.rhf.RHF):
        nocc = len(mf.mo_occ[mf.mo_occ > 0])
        fockao = mf.get_fock()
        fockmo = mf.mo_coeff.T @ fockao @ mf.mo_coeff
        foo = fockmo[:nocc, :nocc]
        fvv = fockmo[nocc:, nocc:]
        mo_coeff_occ = np.dot(mf.mo_coeff[:, :nocc], np.linalg.eigh(foo)[1])
        mo_coeff_vir = np.dot(mf.mo_coeff[:, nocc:], np.linalg.eigh(fvv)[1])
        mf.mo_coeff = np.concatenate((mo_coeff_occ, mo_coeff_vir), axis=1)
    else:
        assert False
"""

CASCI = """
from pyscf import mcscf
import os

mcscf.casci.FRAC_OCC_THRESHOLD = %s

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
if d3:
    from pyscf import dftd3
    mf = dftd3.dftd3(mf)

mf.chkfile = "mf.chk"
mf.mo_coeff = coeff

mc = mcscf.%s(mf, nactorb, (nacta, nactb))
mc.conv_tol = %s
mc.max_cycle_macro = %s
mc.canonicalization = %s
mc.natorb = %s

mcfs = [mc.fcisolver]
"""

DMRG = """
from pyscf import dmrgscf, lib
import os

dmrgscf.settings.BLOCKEXE = os.popen("which %s").read().strip()
dmrgscf.settings.MPIPREFIX = "" if "PYSCF_MPIPREFIX" not in os.environ else os.environ["PYSCF_MPIPREFIX"]

mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=%s, tol=%s)
mc.fcisolver.runtimeDir = lib.param.TMPDIR
mc.fcisolver.scratchDirectory = lib.param.TMPDIR
mc.fcisolver.threads = int(os.environ["OMP_NUM_THREADS"])
mc.fcisolver.memory = int(mol.max_memory / 1000)

mcfs = [mc.fcisolver]
"""

DMRG_MIXSPIN = """
from pyscf import dmrgscf, lib
import os

dmrgscf.settings.BLOCKEXE = os.popen("which %s").read().strip()
dmrgscf.settings.MPIPREFIX = "" if "PYSCF_MPIPREFIX" not in os.environ else os.environ["PYSCF_MPIPREFIX"]

mcfs = [dmrgscf.DMRGCI(mol, maxM=%s, tol=%s) for _ in range(2)]
weights = [1.0 / len(mcfs)] * len(mcfs)

mcfs[0].spin = 0
mcfs[1].spin = spin

for i, mcf in enumerate(mcfs):
    mcf.runtimeDir = lib.param.TMPDIR + "/%%d" %% i
    mcf.scratchDirectory = lib.param.TMPDIR + "/%%d" %% i
    mcf.threads = int(os.environ["OMP_NUM_THREADS"])
    mcf.memory = int(mol.max_memory / 1000) # mem in GB

mc = mcscf.CASSCF(mf, nactorb, (nacta + nactb))
mcscf.state_average_mix_(mc, mcfs, weights)
"""

CASCC = """
import numpy as np
from pyscf import cc, gto

class CCSolver:
    def __init__(self, ccsd_t=False):
        self.ccsd_t = ccsd_t

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        mol = gto.M(verbose=4)
        mol.nelectron = sum(nelec)
        mol.spin = nelec[0] - nelec[1]
        mf = mol.RHF()
        mf._eri = h2
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.identity(norb)
        if mol.spin == 0:
            mf.mo_coeff = np.identity(norb)
            mf.mo_occ = np.zeros(norb)
            mf.mo_occ[:nelec[0]] += 1.0
            mf.mo_occ[:nelec[1]] += 1.0
        else:
            mf.mo_coeff = np.identity(norb)
            mf.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
            mf.mo_occ = np.zeros((2, norb))
            mf.mo_occ[0][:nelec[0]] += 1.0
            mf.mo_occ[1][:nelec[1]] += 1.0

        if cascc_mf_relax:
            mf.kernel()

        if semi_canonical:
            semi_canon(mf)

        self.cc = cc.CCSD(mf)
        self.cc.level_shift = %s
        self.cc.incore_complete = True
        self.cc.run()
        if self.ccsd_t:
            e_ccsd_t = self.cc.e_tot + self.cc.ccsd_t()
        else:
            e_ccsd_t = self.cc.e_tot
        return e_ccsd_t + ecore, dict(t1=self.cc.t1, t2=self.cc.t2)

    def make_rdm1(self, t12, norb, nelec):
        dms = self.cc.make_rdm1(**t12)
        if isinstance(dms, tuple):
            return dms[0] + dms[1]
        else:
            return dms

mc.fcisolver = CCSolver(ccsd_t=%s)
mcfs = [mc.fcisolver]
"""

CASUCC = """
import numpy as np
from pyscf import cc, gto
from libdmet.basis_transform import make_basis

scf_dmao = np.load("%s/mf_dmao.npy")
scf_dmlo = make_basis.transform_rdm1_to_lo_mol(scf_dmao, coeff, mf.get_ovlp())
dmcas = scf_dmlo[:, mc.ncore:mc.ncore + mc.ncas, mc.ncore:mc.ncore + mc.ncas]

print('idemponency of dmcas[0]: %%s' %% np.linalg.norm(dmcas[0].dot(dmcas[0]) - dmcas[0]))
print('idemponency of dmcas[1]: %%s' %% np.linalg.norm(dmcas[1].dot(dmcas[1]) - dmcas[1]))
print('trace of dmcas[0]: %%s' %% np.trace(dmcas[0]))
print('trace of dmcas[1]: %%s' %% np.trace(dmcas[1]))

class UCCSolver:
    def __init__(self, ccsd_t=False, dmcas=None):
        self.ccsd_t = ccsd_t
        self.dmcas = dmcas

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        mol = gto.M(verbose=4)
        mol.nelectron = sum(nelec)
        mol.spin = nelec[0] - nelec[1]
        mf = mol.UHF()
        mf._eri = h2
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.identity(norb)
        mf.max_cycle = %s
        mf.kernel(dm0=self.dmcas)

        if semi_canonical:
            semi_canon(mf)

        self.cc = cc.UCCSD(mf)
        self.cc.level_shift = %s
        self.cc.max_cycle = %s
        self.cc.incore_complete = True
        self.cc.run()
        if self.ccsd_t:
            e_ccsd_t = self.cc.e_tot + self.cc.ccsd_t()
        else:
            e_ccsd_t = self.cc.e_tot
        return e_ccsd_t + ecore, dict(t1=self.cc.t1, t2=self.cc.t2)

    def make_rdm1(self, t12, norb, nelec):
        dms = self.cc.make_rdm1(**t12)
        if isinstance(dms, tuple):
            return dms[0] + dms[1]
        else:
            return dms

mc.fcisolver = UCCSolver(ccsd_t=%s, dmcas=dmcas)
mcfs = [mc.fcisolver]
"""

CASSCF_MIXSPIN = """
from pyscf import fci
mch = mc
mch.fcisolver.spin = spin
mcfh = fci.addons.fix_spin(mch.fcisolver, 0.2, spin * (spin + 2) / 4.0)
mcl = mcscf.CASSCF(mf, nactorb, nacta + nactb)
mcl.fcisolver.spin = 0
mcfl = fci.addons.fix_spin(mcl.fcisolver, 0.2 , 0.0)
mc = mcscf.addons.state_average_mix(mc, [mcfl, mcfh], [0.5, 0.5])
mc.conv_tol = mch.conv_tol
mc.max_cycle_macro = mch.max_cycle_macro
mc.canonicalization = True
mc.natorb = True

mcfs = [mcl.fcisolver, mch.fcisolver]
"""

DRYRUN_FINAL = """
for mcf in mcfs:
    mcf.conv_tol = %s
from pyscf import dmrgscf
dmrgscf.dryrun(mc)
"""

CASCI_FINAL = """
for mcf in mcfs:
    mcf.conv_tol = %s
mc.%s()
for _ in range(%s):
    if not mc.converged:
        mc.%s()
dmao = mc.make_rdm1()
"""

ALL_FINAL = """
import numpy as np
coeff_inv = np.linalg.pinv(mc.mo_coeff)
dmmo = np.einsum('...ip,...pq,...jq->...ij', coeff_inv, dmao, coeff_inv, optimize=True)
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

@handle_io_error
def write(fn, pmc, pmf, is_casci=True):
    with open(fn, "w") as f:

        f.write(TIME_ST)

        lde = pmc["load_mf"]
        if "/" not in lde:
            lde = "../" + lde
        
        mf_lde = lde

        f.write(MF_LOAD % (lde + "/mf.chk", "x2c" in pmf, "dftd3" in pmf))

        if "cas_semi_canonical" in pmc:
            f.write("semi_canonical = True\n")
            f.write(SEMI_CANON)

        if "cascc_mf_relax" in pmc:
            f.write("cascc_mf_relax = True\n")

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
            pmc["frac_occ_tol"],
            "CASCI" if is_casci else "CASSCF",
            pmc["conv_tol"],
            pmc["max_cycle"],
            "False" if "no_canonicalize" in pmc else "True",
            "False" if "no_canonicalize" in pmc else "True"
        ))

        if "cas_uccsd" in pmc:
            f.write(CASUCC % (mf_lde,
                pmc.get("cas_ucc_mf_max_cycle", 0),
                pmc.get("level_shift", 0.0),
                pmc.get("cas_ucc_cc_max_cycle", 100),
                False, ))

        if "cas_uccsd_t" in pmc:
            f.write(CASUCC % (mf_lde,
                pmc.get("cas_ucc_mf_max_cycle", 0),
                pmc.get("level_shift", 0.0),
                pmc.get("cas_ucc_cc_max_cycle", 100),
                True, ))

        if "cas_ccsd" in pmc:
            f.write(CASCC % (pmc.get("level_shift", 0.0), False, ))

        if "cas_ccsd_t" in pmc:
            f.write(CASCC % (pmc.get("level_shift", 0.0), True, ))

        if "stackblock-dmrg" in pmc or "block2-dmrg" in pmc:
            if "mixspin" in pmc:
                f.write(DMRG_MIXSPIN % (
                    "block.spin_adapted" if "stackblock-dmrg" in pmc else "block2main",
                    pmc["maxm"], pmc["fci_conv_tol"]))
            else:
                f.write(DMRG % (
                    "block.spin_adapted" if "stackblock-dmrg" in pmc else "block2main",
                    pmc["maxm"], pmc["fci_conv_tol"]))

        if "dmrg-sch-sweeps" in pmc:
            f.write("for mcf in mcfs:\n")
            f.write("    mcf.scheduleSweeps = %s\n" % list(map(int, pmc["dmrg-sch-sweeps"].split(";"))))
            assert len(pmc["dmrg-sch-maxms"].split(";")) == len(pmc["dmrg-sch-sweeps"].split(";"))
            f.write("    mcf.scheduleMaxMs = %s\n" % list(map(int, pmc["dmrg-sch-maxms"].split(";"))))
            if ";" in pmc["dmrg-sch-tols"]:
                assert len(pmc["dmrg-sch-tols"].split(";")) == len(pmc["dmrg-sch-sweeps"].split(";"))
                f.write("    mcf.scheduleTols = %s\n" % list(map(float, pmc["dmrg-sch-tols"].split(";"))))
            else:
                f.write("    mcf.scheduleTols = %s\n" % ([float(pmc["dmrg-sch-tols"])] * len(pmc["dmrg-sch-sweeps"].split(";"))))
            if ";" in pmc["dmrg-sch-noises"]:
                assert len(pmc["dmrg-sch-noises"].split(";")) == len(pmc["dmrg-sch-sweeps"].split(";"))
                f.write("    mcf.scheduleNoises = %s\n" % list(map(float, pmc["dmrg-sch-noises"].split(";"))))
            else:
                f.write("    mcf.scheduleNoises = %s\n" % ([float(pmc["dmrg-sch-noises"])] * len(pmc["dmrg-sch-sweeps"].split(";"))))
            f.write("    mcf.maxIter = %s\n" % pmc["dmrg-max-iter"])
            f.write("    mcf.twodot_to_onedot = %s\n" % pmc["dmrg-tto"])
            if "dmrg-tol" in pmc:
                f.write("    mcf.tol = %s\n" % pmc["dmrg-tol"])
            if "dmrg-no-2pdm" in pmc:
                f.write("    mcf.twopdm = False\n")
            if "dmrg-1pdm" in pmc:
                f.write("    mcf.block_extra_keyword = ['%s']\n" % "onepdm")

        if "nonspinadapted" in pmc:
            f.write("for mcf in mcfs:\n")
            f.write("    mcf.nonspinAdapted = True\n")

        if "mixspin" in pmc and not ("stackblock-dmrg" in pmc or "block2-dmrg" in pmc):
            f.write(CASSCF_MIXSPIN)

        if "step_size" in pmc:
            f.write("mc.max_stepsize = %s\n" % pmc["step_size"])

        if "ci_response_space" in pmc:
            f.write("mc.ci_response_space = %s\n" % pmc["ci_response_space"])

        if "dryrun" in pmc:
            f.write(DRYRUN_FINAL % pmc["fci_conv_tol"])

            if "dmrg-rev-sweeps" in pmc:
                f.write("for mcf in mcfs:\n")
                f.write("    mcf.scheduleSweeps = %s\n" % list(map(int, pmc["dmrg-rev-sweeps"].split(";"))))
                assert len(pmc["dmrg-rev-maxms"].split(";")) == len(pmc["dmrg-rev-sweeps"].split(";"))
                f.write("    mcf.scheduleMaxMs = %s\n" % list(map(int, pmc["dmrg-rev-maxms"].split(";"))))
                if ";" in pmc["dmrg-rev-tols"]:
                    assert len(pmc["dmrg-rev-tols"].split(";")) == len(pmc["dmrg-rev-sweeps"].split(";"))
                    f.write("    mcf.scheduleTols = %s\n" % list(map(float, pmc["dmrg-rev-tols"].split(";"))))
                else:
                    f.write("    mcf.scheduleTols = %s\n" % ([float(pmc["dmrg-rev-tols"])] * len(pmc["dmrg-rev-sweeps"].split(";"))))
                if ";" in pmc["dmrg-rev-noises"]:
                    assert len(pmc["dmrg-rev-noises"].split(";")) == len(pmc["dmrg-rev-sweeps"].split(";"))
                    f.write("    mcf.scheduleNoises = %s\n" % list(map(float, pmc["dmrg-rev-noises"].split(";"))))
                else:
                    f.write("    mcf.scheduleNoises = %s\n" % ([float(pmc["dmrg-rev-noises"])] * len(pmc["dmrg-rev-sweeps"].split(";"))))
                f.write("    mcf.maxIter = %s\n" % pmc["dmrg-rev-iter"])
                f.write("    mcf.twodot_to_onedot = 0\n")
                f.write("    mcf.tol = 0.0\n")
                f.write("    mcf.twopdm = False\n")
                f.write("    mcf.block_extra_keyword = ['fullrestart', 'twodot', 'extrapolation']\n")
                f.write("    mcf.configFile = \"dmrg-rev.conf\"\n")

                f.write(DRYRUN_FINAL % pmc["fci_conv_tol"])
            
            if "dmrg-csf" in pmc:
                f.write("for mcf in mcfs:\n")
                f.write("    mcf.scheduleSweeps = [0]\n")
                f.write("    mcf.scheduleMaxMs = [%s]\n" % pmc["maxm"])
                f.write("    mcf.scheduleTols = [5E-6]\n")
                f.write("    mcf.scheduleNoises = [0.0]\n")
                f.write("    mcf.maxIter = 1\n")
                f.write("    mcf.twodot_to_onedot = 0\n")
                f.write("    mcf.tol = 0.0\n")
                f.write("    mcf.twopdm = False\n")
                f.write("    mcf.block_extra_keyword = ['trans_mps_to_singlet_embedding']\n")
                f.write("    mcf.block_extra_keyword += ['restart_copy_mps SEKET']\n")
                f.write("    mcf.block_extra_keyword += ['restart_sample %s']\n" % pmc["dmrg-csf"])
                f.write("    mcf.configFile = \"dmrg-csf.conf\"\n")

                f.write(DRYRUN_FINAL % pmc["fci_conv_tol"])

        else:
            f.write(CASCI_FINAL % (
                pmc["fci_conv_tol"],
                "kernel" if is_casci else "mc2step",
                pmc["nrepeat"],
                "kernel" if is_casci else "mc2step")
            )
            f.write(ALL_FINAL)

        f.write(TIME_ED)
