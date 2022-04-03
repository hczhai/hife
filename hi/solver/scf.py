
from .mol import TIME_ST, TIME_ED, MOL, MOL_FINAL

MF = """
mf = %s
mf.chkfile = 'mf.chk'
mf.conv_tol = %s
"""

MF_SMEAR = """
mf = %s
mf = pbc_helper.smearing_(mf, sigma=%s, method='fermi', fit_spin=True)
mf.chkfile = 'mf.chk'
mf.conv_tol = %s
"""

MF_FINAL = """
mf.kernel(dm0=dm)
dm = mf.make_rdm1()
"""

ALL_FINAL = """
import numpy as np
np.save("mf_occ.npy", mf.mo_occ)
np.save("mo_coeff.npy", mf.mo_coeff)
np.save("mo_energy.npy", mf.mo_energy)
np.save("e_tot.npy", mf.e_tot)
np.save("mf_dmao.npy", dm)
"""

def write(fn, pmf):
    with open(fn, "w") as f:

        f.write(TIME_ST)

        gemo = open(pmf["geometry"]).readlines()
        assert int(gemo[0]) == len(gemo) - 2

        f.write(MOL % (
            "".join(gemo[2:]),
            pmf["basis"] if ":" in pmf["basis"] else "\"%s\"" % pmf["basis"],
            pmf["spin"],
            pmf["charge"]
        ))

        if "max_memory" in pmf:
            f.write("mol.max_memory = %s\n" % pmf["max_memory"])

        if "cart" in pmf:
            f.write("mol.cart = True\n")

        f.write(MOL_FINAL)

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
        f.write("dm = None\n")

        if "smearing" in pmf:
            f.write("from libdmet.routine import pbc_helper\n")
            sigmas = pmf["smearing"].split(";")
            if "smearing_max_cycle" in pmf:
                sm_max_cycle = pmf["smearing_max_cycle"].split(";")
            else:
                sm_max_cycle = [pmf["max_cycle"]] * len(sigmas)
            if "smearing_conv_tol" in pmf:
                sm_conv_tol = pmf["smearing_conv_tol"].split(";")
            else:
                sm_conv_tol = [pmf["conv_tol"]] * len(sigmas)
            if "smearing_method" in pmf:
                sm_mme = [xmethod(x, "x2c" in pmf) for x in pmf["smearing_method"].split(";")]
            else:
                sm_mme = [mme] * len(sigmas)
            for sg, mc, ct, sm in zip(sigmas, sm_max_cycle, sm_conv_tol, sm_mme):
                f.write(MF_SMEAR % (sm, sg, ct))
                if "KS" in sm:
                    f.write("mf.xc = '%s'\n" % pmf["func"])
                f.write("mf.max_cycle = %s\n" % mc)
                f.write(MF_FINAL + "\n")

        f.write(MF % (mme, pmf["conv_tol"]))
        if "KS" in mme:
            f.write("mf.xc = '%s'\n" % pmf["func"])
        if "max_cycle" in pmf:
            f.write("mf.max_cycle = %s\n" % pmf["max_cycle"])
        f.write(MF_FINAL)

        if "newton_conv" in pmf:
            f.write("mf = mf.newton()\n")
            f.write("mf.conv_tol = %s\n" % pmf["newton_conv"])
            f.write(MF_FINAL)

        f.write(ALL_FINAL)

        f.write(TIME_ED)
