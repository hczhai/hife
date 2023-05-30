
from .mol import TIME_ST, TIME_ED, handle_io_error
from .cc import XCC

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

ncore, ncas = %s, %s
xcc_nelec = None
xcc_ncas = None

txx = time.perf_counter()
do_ccsd_t = True
do_st_extrap = False
"""

ST_LOAD_COEFF = """
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

mf.mo_coeff = coeff if coeff.ndim == 3 else [coeff, coeff]
mf.mo_occ = mo_occ if mo_occ.ndim == 2 else [mo_occ / 2, mo_occ / 2]
mf.mo_energy = None
mf.e_tot = mf.energy_tot()
print('ref energy = ', mf.e_tot)
"""

ST_LOAD_AMPS = """
import numpy as np
import os

for fname in ["mp2_t1.npy", "ccsd_t1.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        tt1 = np.load(lde + "/" + fname)
        break

for fname in ["mp2_t2.npy", "ccsd_t2.npy"]:
    if os.path.isfile(lde + "/" + fname):
        print("use: " + lde + "/" + fname)
        tt2 = np.load(lde + "/" + fname)
        break
"""

ST_CC = """
from pyscf import cc
nfrozen = %s
if xcc_ncas is not None:
    xna = (xcc_nelec + mol.spin) // 2
    xnb = (xcc_nelec - mol.spin) // 2
    mc = XUCCSD(mf, xcc_ncas, (xna, xnb), frozen=nfrozen)
else:
    mc = cc.UCCSD(mf, frozen=nfrozen)
mc.max_cycle = %s
eris = mc.ao2mo(mc.mo_coeff)
mc.kernel(eris=eris)
tt1, tt2 = mc.t1, mc.t2
print('ECCSD    = ', mc.e_tot)

if do_ccsd_t:
    e_t_all = mc.ccsd_t(eris=eris)
    print("ECCSD(T) = ", e_t_all + mc.e_tot)

    # from pyblock2.cc.uccsd import wick_t3_amps, wick_ccsd_t
    # t3 = wick_t3_amps(mc, mc.t1, mc.t2, eris=eris)
    # for t3x in t3:
    #     ged = t3x.shape[:3]
    #     xst, xed = ncore, ncore + ncas
    #     t3x[xst:, xst:, xst:, :xed - ged[0], :xed - ged[1], :xed - ged[2]] = 0
    # e_t_no_cas = wick_ccsd_t(mc, mc.t1, mc.t2, eris=eris, t3=t3)

    # require a custom version of pyscf
    # https://github.com/hczhai/pyscf/tree/ccsd_t_cas
    from pyscf.cc.uccsd_t import _gen_contract_aaa
    import inspect
    assert "cas_exclude" in inspect.getfullargspec(_gen_contract_aaa).args

    eris.cas_exclude = ncore, mc.t1[0].shape[-1] - (ncore + ncas - mc.t1[0].shape[0])
    print('\\ncas_exclude = ', eris.cas_exclude)
    e_t_no_cas = mc.ccsd_t(eris=eris)

    print("E(T) = ", e_t_no_cas, '\\n')
del mc
del eris
"""

ST_MP2 = """
from pyscf import mp
mc = mp.UMP2(mf, frozen=%s)
mf.converged = False
mc.max_cycle = %s
mc.kernel()

tt1, tt2 = mc.t1, mc.t2
e_t_no_cas = None
del mc
"""

ST = """
from pyblock2._pyscf.ao2mo import get_uhf_integrals
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
from pyblock2.driver.core import STTypes, SimilarityTransform
import numpy as np
import os

def fmt_size(i, suffix='B'):
    if i < 1000:
        return "%%d %%s" %% (i, suffix)
    else:
        a = 1024
        for pf in "KMGTPEZY":
            p = 2
            for k in [10, 100, 1000]:
                if i < k * a:
                    return "%%%%.%%df %%%%s%%%%s" %% p %% (i / a, pf, suffix)
                p -= 1
            a *= 1024
    return "??? " + suffix

_, n_elec, spin, ecore, h1e, g2e, orb_sym = get_uhf_integrals(mf, ncore=%s)
e_hf = mf.e_tot
del mf

try:
    import psutil
    mem = psutil.Process(os.getpid()).memory_info().rss
    print("pre-st memory usage = ", fmt_size(mem))
except ImportError:
    pass

dtotal = 0
for k, dx in [('t1', tt1), ('t2', tt2), ('h1e', h1e), ('g2e', g2e)]:
    for ddx in dx:
        print(k, "data memory = ", fmt_size(ddx.nbytes))
        dtotal += ddx.nbytes
print("total data memory = ", fmt_size(dtotal))

print('ecore = ', ecore)
print('orb_sym = ', orb_sym)

scratch = lib.param.TMPDIR

print("PART TIME (PRE)  = %%20.3f" %% (time.perf_counter() - txx))
txx = time.perf_counter()

driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SZ,
                    stack_mem=int(mol.max_memory * 1000 ** 2), n_threads=int(os.environ["OMP_NUM_THREADS"]))
driver.integral_symmetrize(orb_sym, h1e=h1e, g2e=g2e, iprint=1)
for ttx in tt1 + tt2:
    assert np.array(orb_sym).ndim == 1
    orb_syms = []
    for ip in ttx.shape[:ttx.ndim // 2]:
        orb_syms.append(orb_sym[:ip])
    for ip in ttx.shape[ttx.ndim // 2:]:
        orb_syms.append(orb_sym[-ip:])
    driver.integral_symmetrize(orb_syms, hxe=ttx, iprint=1)
dt, ecore, ncas, n_elec = SimilarityTransform.make_sz(h1e, g2e, ecore, tt1, tt2, scratch,
    n_elec, ncore=ncore, ncas=ncas, st_type=STTypes.%s, iprint=2)
del h1e, g2e, tt1, tt2

try:
    import psutil
    mem = psutil.Process(os.getpid()).memory_info().rss
    print("pre-dmrg memory usage = ", fmt_size(mem))
except ImportError:
    pass

print("PART TIME (ST)  = %%20.3f" %% (time.perf_counter() - txx))
txx = time.perf_counter()

print('neleccas =', n_elec, 'ncas =', ncas, 'spin = ', spin)

driver.initialize_system(
    n_sites=ncas, n_elec=n_elec, spin=spin,
    orb_sym=np.array(orb_sym)[..., ncore:ncore + ncas], pg_irrep=0
)

b = driver.expr_builder()
for expr, v in dt.items():
    print('expr = ', expr)
    b.add_sum_term(expr, np.load(v), cutoff=1E-13)
b.add_const(ecore)
print('ok')

for k in os.listdir(scratch):
    if k.endswith(".npy") and k.startswith("ST-DMRG."):
        os.remove(scratch + "/" + k)

print("PART TIME (TERM)  = %%20.3f" %% (time.perf_counter() - txx))
txx = time.perf_counter()

mpo = driver.get_mpo(b.finalize(), algo_type=MPOAlgorithmTypes.FastBipartite, iprint=2)

print("PART TIME (MPO)  = %%20.3f" %% (time.perf_counter() - txx))
txx = time.perf_counter()

ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
bond_dims = [400] * 4 + [800] * 8
noises = [1e-5] * 4 + [1e-6] * 4 + [0]
thrds = [1e-7] * 4 + [1e-9] * 4
e_st = driver.dmrg(
    mpo,
    ket,
    n_sweeps=20,
    dav_type="NonHermitian",
    bond_dims=bond_dims,
    noises=noises,
    thrds=thrds,
    iprint=2,
)

print('EST    = ', e_st)
if e_t_no_cas is not None:
    print('EST(T) = ', e_st + e_t_no_cas)
print("PART TIME (DMRG)  = %%20.3f" %% (time.perf_counter() - txx))

if do_st_extrap:
    ket = ket.deep_copy('GS-TMP')
    bond_dims = [600] * 4 + [500] * 4 + [400] * 4 + [300] * 4 + [200] * 4 + [100] * 4
    noises = [0] * 24
    thrds = [1e-12] * 24
    energy = driver.dmrg(mpo, ket, n_sweeps=24, bond_dims=bond_dims, noises=noises,
        dav_type="NonHermitian", tol=0, thrds=thrds, iprint=2)

for k in os.listdir(scratch):
    if '.PART.' in k:
        os.remove(scratch + "/" + k)
"""

ALL_FINAL = """
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

        f.write(MF_LOAD % (
            lde + "/mf.chk",
            mme,
            pmc["ncore"],
            pmc["ncas"]
        ))

        if "max_memory" in pmc:
            f.write("mf.max_memory = %s\n" % pmc["max_memory"])

        if "no_ccsd_t" in pmc:
            f.write("do_ccsd_t = False\n")

        if "do_st_extrap" in pmc:
            f.write("do_st_extrap = True\n")

        if "KS" in mme or "RHF" in mme:
            f.write("mfhf = scf.UHF(mol)\n")
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

            f.write(ST_LOAD_COEFF)

        if "load_amps" in pmc:
            lde = pmc["load_amps"]
            if "/" not in lde:
                lde = "../" + lde

            f.write("lde = '%s'\n" % lde)

            f.write(ST_LOAD_AMPS)
        elif "from_mp2" in pmc:
            f.write(ST_MP2 % (
                pmc["frozen"],
                pmc["max_cycle"]
            ))
        else:

            if "xcc_nelec" in pmc or "xcc_ncas" in pmc:
                f.write(XCC)

            if "xcc_ncas" in pmc:
                f.write("xcc_ncas = %s\n" % pmc["xcc_ncas"])
            if "xcc_nelec" in pmc:
                f.write("xcc_nelec = %s\n" % pmc["xcc_nelec"])

            f.write(ST_CC % (
                pmc["frozen"],
                pmc["max_cycle"]
            ))

        f.write(ST % (
            pmc["frozen"],
            pmc.get("st_type", "H_HT_HT2T2"))
        )

        f.write(ALL_FINAL)
        f.write(TIME_ED)
