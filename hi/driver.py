
import os
import time
from .utils import read_opts, read_json, write_json, optcopy
from .render import ScriptsRender
from shutil import copyfile

HIFEHOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_host():
    xhost = os.popen("hostname -f | grep '\\.' || hostname -a").read().strip()
    xhost = xhost.lower()
    keys = ['pauling', 'mac', 'cori']
    for k in keys:
        if k in xhost:
            return k
    keys = [('cm.cluster', 'hpc')]
    for k, v in keys:
        if k in xhost:
            return v
    return 'unknown'

host_cores = {'pauling': 28, 'mac': 4, 'hpc': 24, 'cori': 32 }
host_part = {'pauling': 'serial,parallel', 'hpc': 'any', 'mac': '', 'cori': '' }
host_def_queue = {'pauling': 'normal', 'cori': 'regular', 'hpc': 'normal', 'mac': ''}
extra_basis = { 'def2-sv(p)': 'def2-svpp.dat' }

def time_span_str(time, no_sec=False):
    time = int(time)
    xsec = time % 60
    xmin = (time // 60) % 60
    xhours = (time // 60 // 60) % 24
    xdays = time // 60 // 60 // 24
    rstr = []
    if not no_sec:
        rstr.append("%d sec" % xsec)
    if xmin != 0:
        rstr.append("%d min " % xmin)
    if xhours != 0:
        rstr.append("%d hr " % xhours)
    if xdays != 0:
        rstr.append("%d day " % xdays)
    return ''.join(rstr[::-1])

# solve expr: 1-5,9 => [ 1, 2, 3, 4, 5, 9 ]
def solve_expr(ms):
    mk = []
    for l in ms.split(","):
        if '-' not in l:
            mk.append(int(l))
        else:
            ll = l.split('-')
            mk += list(range(int(ll[0]), int(ll[1]) + 1))
    return mk

def cast_value(orig, x):
    if isinstance(orig, bool):
        return x in ["True", "true", "T", "t", "1"]
    elif isinstance(orig, float):
        return float(x)
    elif isinstance(orig, int):
        return int(x)
    elif isinstance(orig, str):
        return x
    elif isinstance(orig, list):
        if len(orig) < len(x.split(",")):
            orig = [''] * len(x.split(","))
        return [cast_value(orig[ix], x) for ix, x in enumerate(x.split(","))]
    else:
        raise RuntimeError("Unsupport data type: %s" % orig.__class__)

class BaseDriver:

    def __init__(self, argv):
        self.this = argv[0]
        self.args = argv[1:]
        if len(self.args) == 0:
            self.args = ["help"]
        self.task, self.args = self.args[0], self.args[1:]
        self.scripts_dir = HIFEHOME + "/scripts"
        self.scripts_templates_dir = HIFEHOME + "/scripts"
        self.scripts_spec_dir = HIFEHOME + "/scripts.spec"
        self.structures_dir = HIFEHOME + "/structures"
        self.basis_dir = HIFEHOME + "/basis"
        self.tasks = []

    def run(self):
        if "help" in self.task or self.task == "-h":
            self.help(self.args)
        else:
            found = False
            for task in self.tasks:
                if self.task == task:
                    getattr(self, task)(self.args)
                    found = True
                    break
            if not found:
                raise RuntimeError("Unknown task name: %s" % self.task)
    
    def help(self, args):
        print("HiFe version 0.1")

class HFDriver(BaseDriver):

    def __init__(self, argv):
        super().__init__(argv)
        self.tasks = ["init", "show", "clean", "mf", "cc", "mp",
            "set", "create", "submit", "log", "act", "orb",
            "ex", "select", "casci", "casscf", "mrpt", "avas",
            "stdmrg"]

    def run(self):
        super().run()
        if "help" in self.task or self.task == "-h":
            return
        if self.task in [ "sync", "show", "check", "clean" ]:
            return
        if self.task == "log" and "exact" in self.args:
            return
        self.to_dir(dox="local")
        if os.path.isfile("./CMD-HISTORY"):
            ftime = os.path.getmtime("./CMD-HISTORY")
        else:
            ftime = None
        with open("./CMD-HISTORY", "a") as f:
            if ftime is not None and time.time() - ftime > 3600:
                f.write("# AFTER %d HOURS\n" % int((time.time() - ftime) / 3600))
            f.write("hife %s %s\n" % (self.task, " ".join([
                a if ";" not in a and "(" not in a and "=" not in a and "|" not in a else
                ("\"%s\"" % a) for a in self.args])))

    def lr_dirs(self, cur=False):
        """get local and remote dirs
        if cur, also return current root dir"""
        ndir = None
        jdir = None
        if os.path.isfile("../../DIRECTORIES"):
            ndir = "../../DIRECTORIES"
            if cur:
                jdir = "../.."
        elif os.path.isfile("./DIRECTORIES"):
            ndir = "./DIRECTORIES"
            if cur:
                jdir = "."
        elif os.path.isfile("../DIRECTORIES"):
            ndir = "../DIRECTORIES"
            if cur:
                jdir = ".."
        else:
            return None
        d = [x for x in open(ndir, 'r').read().split('\n') if len(x) != 0][:2]
        if cur:
            jdir = os.path.abspath(jdir)
            return d + [jdir]
        else:
            return d

    def to_dir(self, dox=None):
        lr = self.lr_dirs()
        if lr == None:
            return False
        else:
            ldir, rdir = lr
            if dox == "local":
                os.chdir(ldir)
            elif dox == "remote":
                os.chdir(rdir)
            return True

    def show(self, args):
        lr = self.lr_dirs(cur=True)
        if lr is None:
            raise RuntimeError("DIRECTORIES not found!")
        ldir, rdir, cdir = lr
        def_pos = { "0": "dir" }
        opts = read_opts(args, def_pos, [])
        if "dir" not in opts:
            if ldir == cdir: opts["dir"] = "remote"
            else: opts["dir"] = "local"
        if opts["dir"] == "local":
            print(ldir)
        elif opts["dir"] == "remote":
            print(rdir)

    def pre_info(self):
        self.to_dir(dox="local")
        if os.path.isfile("./hife-parameters.json"):
            para_temp = read_json("./hife-parameters.json")
        else:
            raise RuntimeError("please run 'hife init ...' first!")
        xmodel = para_temp["hosts"]["model"]
        xhost = para_temp["hosts"]["name"]
        self.scripts_render = ScriptsRender(
            self.scripts_dir,
            self.scripts_templates_dir, 
            self.scripts_spec_dir, xhost, xmodel)
        return para_temp

    def mf(self, args):
        """DFT or HF calculation."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        lr = self.lr_dirs()
        def_pos = { "0": "stage" }
        opts = {
            "geometry": pre["create"]["name"] + ".xyz",
            "spin": pre["create"]["spin"],
            "charge": pre["create"]["charge"],
            "func": pre["create"]["func"],
            "basis": pre["create"]["basis"],
            "method": pre["create"]["method"],
            "max_memory": "100000",
            "max_cycle": "1000",
            "conv_tol": "1E-12"
        }
        optl = [ "smearing", "smearing_conv_tol", "x2c", "dftd3",
            "smearing_method", "smearing_max_cycle", "newton_max_cycle",
            "dimer_init", "dimer_idx", "dimer_spin", "dimer_type", "direct_newton",
            "newton_conv", "newton_conv_grad", "ecp", "pg" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "mf-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if ":dimer:" in opts["geometry"]:
            elem, _, rr = opts["geometry"].split('.xyz')[0].split(':') # def unit is ang
            elem = elem[0].upper() + elem[1:]
            geom = open(self.structures_dir + "/dimer.xyz", "r").read()
            geom = geom % (elem, -float(rr) / 2, elem, float(rr) / 2)
            geom = [x + '\n' for x in geom.split('\n')]
        elif os.path.isfile(lr[0] + "/" + opts["geometry"]):
            geom = lr[0] + "/" + opts["geometry"]
        elif os.path.isfile(self.structures_dir + "/" + opts["geometry"]):
            geom = self.structures_dir + "/" + opts["geometry"]
        else:
            raise RuntimeError("cannot find %s!" % opts["geometry"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        pre[sec_key]["geometry"] = geom
        write_json(pre, "./hife-parameters.json")

    def stdmrg(self, args):
        """ST-DMRG calculation."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage" }
        opts = { "frozen": "0", "max_cycle": "1000" }
        optl = [ "load_mf", "load_coeff", "load_amps",
            "spin", "max_memory", "ncore", "ncas", "from_mp2",
            "xcc_nelec", "xcc_ncas", "st_type" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "stdmrg-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        if "load_coeff" in opts and opts["load_coeff"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_coeff"])
        if "load_amps" in opts and opts["load_amps"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_amps"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        print("%s mol based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        if "load_coeff" in opts:
            print("%s orb based on %s" % (sec_key, pre[sec_key]["load_coeff"]))
        if "load_amps" in opts:
            print("%s amps based on %s" % (sec_key, pre[sec_key]["load_amps"]))
        write_json(pre, "./hife-parameters.json")

    def mp(self, args):
        """Moller–Plesset perturbation theory calculation."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage" }
        opts = {
            "max_cycle": "1000",
        }
        optl = [ "load_mf", "load_coeff", "do_spin_square",
            "frozen", "spin", "max_memory", "non_canonical",
            "nat_with_pg", "save_amps" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "mp-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        if "load_coeff" in opts and opts["load_coeff"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_coeff"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        print("%s mol based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        if "load_coeff" in opts:
            print("%s orb based on %s" % (sec_key, pre[sec_key]["load_coeff"]))
        write_json(pre, "./hife-parameters.json")

    def cc(self, args):
        """Coupled cluster calculation."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage" }
        opts = {
            "max_cycle": "1000",
            "bcc_conv_tol": 1E-5,
            "bcc_max_cycle": 10
        }
        optl = [ "load_mf", "load_coeff", "level_shift", "no_ccsd_t",
            "do_spin_square", "frozen", "spin", "bcc", "max_memory",
            "nat_with_pg", "save_amps", "xcc_nelec", "xcc_ncas",
            "semi_canonical" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "cc-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        if "load_coeff" in opts and opts["load_coeff"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_coeff"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        print("%s mol based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        if "load_coeff" in opts:
            print("%s orb based on %s" % (sec_key, pre[sec_key]["load_coeff"]))
        write_json(pre, "./hife-parameters.json")
    
    def act(self, args):
        """Orbital localization procedure."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage" }
        opts = {}
        optl = [ "load_mf" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "act-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        print("%s based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        write_json(pre, "./hife-parameters.json")

    def orb(self, args):
        """Generate orbital plot scripts."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage" }
        opts = {}
        optl = [ "load_mf", "load_coeff", "alpha",
            "beta", "from", "to" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf", "load_coeff" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "orb-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        if opts["load_coeff"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_coeff"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        print("%s mol based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        print("%s orb based on %s" % (sec_key, pre[sec_key]["load_coeff"]))
        write_json(pre, "./hife-parameters.json")

    def select(self, args):
        """Select active space, localize and sort active orbitals."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage", "1": "cas_list" }
        opts = {}
        optl = [ "load_mf", "load_coeff", "no_loc", "cas_list",
            "nactorb", "nactelec", "split_low", "split_high",
            "alpha", "beta", "uno", "average_occ", "loc_with_pg" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf", "load_coeff" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "select-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        if opts["load_coeff"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_coeff"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        if "cas_list" in pre[sec_key]:
            pre[sec_key]["cas_list"] = solve_expr(pre[sec_key]["cas_list"])
            print("%s active space %s = %d orbitals" % (sec_key,
                pre[sec_key]["cas_list"], len(pre[sec_key]["cas_list"])))
        print("%s mol based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        print("%s orb based on %s" % (sec_key, pre[sec_key]["load_coeff"]))
        write_json(pre, "./hife-parameters.json")

    def avas(self, args):
        """Select active space automatically using avas."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage", "1": "ao_labels" }
        opts = { "threshold": "0.2" }
        optl = [ "load_mf", "ao_labels" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf", "ao_labels" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "avas-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        print("%s based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        write_json(pre, "./hife-parameters.json")
    
    def casci(self, args, do_casci=True):
        """CASCI/CASSCF calculation."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage" }
        opts = {
            "max_cycle": "50",
            "fci_conv_tol": "1E-10",
            "conv_tol": "1E-8",
            "frac_occ_tol": "1E-6",
            "nrepeat" : "2"
        }
        optl = [ "load_mf", "load_coeff", "spin", "nactorb", "nactelec",
            "stackblock-dmrg", "block2-dmrg", "maxm",
            "step_size", "ci_response_space", "mixspin",
            "no_canonicalize", "dryrun", "dmrg-sch-sweeps", "dmrg-sch-maxms",
            "dmrg-sch-tols", "dmrg-sch-noises", "dmrg-max-iter", "dmrg-tto", "dmrg-tol",
            "dmrg-no-2pdm", "dmrg-1pdm", "dmrg-rev-sweeps", "dmrg-rev-maxms",
            "dmrg-rev-tols", "dmrg-rev-noises", "dmrg-rev-iter", "dmrg-csf",
            "cas_ccsd", "cas_ccsd_t", "cas_uccsd", "cas_uccsd_t",
            "cas_ucc_mf_max_cycle", "cas_ucc_cc_max_cycle", "cascc_mf_relax",
            "nonspinadapted", "level_shift", "cas_semi_canonical" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf", "load_coeff" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "%s-%s" % ("casci" if do_casci else "casscf", opts["stage"])
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        if opts["load_coeff"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_coeff"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        print("%s mol based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        print("%s orb based on %s" % (sec_key, pre[sec_key]["load_coeff"]))
        write_json(pre, "./hife-parameters.json")

    def casscf(self, args):
        self.casci(args, do_casci=False)
    
    def mrpt(self, args):
        """mrpt calculation."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        def_pos = { "0": "stage" }
        opts = { "fci_conv_tol": "1E-10", "frac_occ_tol": "1E-6" }
        optl = [ "load_mf", "load_coeff", "spin",
            "stackblock-dmrg", "block2-dmrg", "maxm",
            "nactorb", "nactelec", "method", "solver", "fix_spin",
            "dmrg-sch-sweeps", "dmrg-sch-maxms",
            "dmrg-sch-tols", "dmrg-sch-noises", "dmrg-max-iter", "dmrg-tto", "dmrg-tol",
            "dmrg-no-2pdm", "dmrg-1pdm", "dmrg-rev-sweeps", "dmrg-rev-maxms",
            "dmrg-rev-tols", "dmrg-rev-noises", "dmrg-rev-iter", "dmrg-csf" ] + list(opts.keys())
        opts.update(read_opts(args, def_pos, optl))
        for k in [ "stage", "load_mf", "load_coeff" ]:
            if k not in opts:
                raise RuntimeError("no %s argument found!" % k)
        sec_key = "mrpt-%s" % opts["stage"]
        if sec_key in pre:
            raise RuntimeError("key %s already used!" % sec_key)
        if opts["load_mf"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_mf"])
        if opts["load_coeff"] not in pre:
            raise RuntimeError("%s not found!" % opts["load_coeff"])
        pre[sec_key] = {}
        for k in optl:
            if k in opts:
                pre[sec_key][k] = opts[k]
        print("%s mol based on %s" % (sec_key, pre[sec_key]["load_mf"]))
        print("%s orb based on %s" % (sec_key, pre[sec_key]["load_coeff"]))
        write_json(pre, "./hife-parameters.json")

    def create(self, args):
        """Create job scripts."""
        pre = self.pre_info()
        self.to_dir(dox="local")
        lr = self.lr_dirs()
        sec_key = "%s-%s" % (args[0], args[1])
        opts = {
            "time": pre["create"]["time"],
            "nodes": pre["hosts"]["nodes"],
            "cores": pre["hosts"]["cores"], 
            "name": "%s.%s.hife" % (pre["create"]["name"][:4], args[1]),
            "mem": pre["hosts"]["mem"],
            "partition": pre["hosts"]["partition"],
            "queue": host_def_queue[pre["hosts"]["name"]],
            "arch": "haswell",
            "tmin": "",
            "serial": "0"
        }
        optl = [] + list(opts.keys())
        opts.update(read_opts(args[2:], {}, optl))
        for xlr in lr:
            xdir = xlr + "/runs/%s" % sec_key
            if not os.path.exists(xdir): 
                print("create %s" % xdir)
                os.makedirs(xdir)
        xdir = lr[0] + "/runs/%s" % sec_key
        rdir = lr[1] + "/runs/%s" % sec_key
        if "basis" in pre[sec_key]:
            pre[sec_key]["basis"] = pre[sec_key]["basis"].lower()
            for bname, bfile in extra_basis.items():
                if bname in pre[sec_key]["basis"]:
                    copyfile(self.basis_dir + "/" + bfile, "%s/%s" % (xdir, bfile))
                    pre[sec_key]["basis"] = pre[sec_key]["basis"].replace(bname, bfile)
        if "load_mf" in pre[sec_key] and "basis" in pre[pre[sec_key]["load_mf"]]:
            pre[pre[sec_key]["load_mf"]]["basis"] = pre[pre[sec_key]["load_mf"]]["basis"].lower()
            for bname, bfile in extra_basis.items():
                if bname in pre[pre[sec_key]["load_mf"]]["basis"]:
                    copyfile(self.basis_dir + "/" + bfile, "%s/%s" % (xdir, bfile))
                    pre[pre[sec_key]["load_mf"]]["basis"] = \
                        pre[pre[sec_key]["load_mf"]]["basis"].replace(bname, bfile)
        if args[0] == "mf":
            from .solver.scf import write
            write("%s/hife.py" % xdir, pre[sec_key])
        elif args[0] == "cc":
            from .solver.cc import write
            write("%s/hife.py" % xdir, pre[sec_key], pre[pre[sec_key]["load_mf"]])
        elif args[0] == "mp":
            from .solver.mp import write
            write("%s/hife.py" % xdir, pre[sec_key], pre[pre[sec_key]["load_mf"]])
        elif args[0] == "stdmrg":
            from .solver.stdmrg import write
            write("%s/hife.py" % xdir, pre[sec_key], pre[pre[sec_key]["load_mf"]])
        elif args[0] == "act":
            from .solver.active import write
            write("%s/hife.py" % xdir, pre[sec_key])
        elif args[0] == "orb":
            from .solver.orbs import write
            write("%s/hife.py" % xdir, pre[sec_key])
        elif args[0] == "select":
            from .solver.select import write
            write("%s/hife.py" % xdir, pre[sec_key])
        elif args[0] == "avas":
            from .solver.avas import write
            write("%s/hife.py" % xdir, pre[sec_key], pre[pre[sec_key]["load_mf"]])
        elif args[0] == "casci":
            from .solver.casci import write
            write("%s/hife.py" % xdir, pre[sec_key], pre[pre[sec_key]["load_mf"]], is_casci=True)
        elif args[0] == "casscf":
            from .solver.casci import write
            write("%s/hife.py" % xdir, pre[sec_key], pre[pre[sec_key]["load_mf"]], is_casci=False)
        elif args[0] == "mrpt":
            from .solver.mrpt import write
            write("%s/hife.py" % xdir, pre[sec_key], pre[pre[sec_key]["load_mf"]])
        else:
            raise NotImplementedError

        ropts = {
            "@MEM": opts["mem"],
            "@NCORES": str(opts["cores"]),
            "@NNODES": str(opts["nodes"]),
            "@TIME": opts["time"],
            "@NAME": opts["name"],
            "@PART": opts["partition"],
            "@TMPDIR": rdir,
            "@RESTART": "0",
            "@BLOCK2": "0",
            "@QUEUE": opts["queue"],
            "@ARCH": opts["arch"],
            "@TMIN": opts["tmin"] if opts["tmin"] != "" else opts["time"],
            "@SERIAL": opts["serial"]
        }
        optcopy(self.scripts_render.get("run.sh"), "%s/run.sh" % xdir, ropts)
        ropts["@RESTART"] = "1"
        optcopy(self.scripts_render.get("run.sh"), "%s/restart.sh" % xdir, ropts)
        if args[0] == "casci":
            ropts["@RESTART"] = "0"
            ropts["@BLOCK2"] = "1"
            ropts["@NAME"] = opts["name"].replace("hife", "dmrg")
            optcopy(self.scripts_render.get("run.sh"), "%s/block2-dmrg.sh" % xdir, ropts)
            ropts["@RESTART"] = "1"
            ropts["@NAME"] = opts["name"].replace("hife", "drev")
            optcopy(self.scripts_render.get("run.sh"), "%s/block2-dmrg-rev.sh" % xdir, ropts)
            ropts["@NAME"] = opts["name"].replace("hife", "dcsf")
            optcopy(self.scripts_render.get("run.sh").replace("dmrg-rev", "dmrg-csf"),
                "%s/block2-dmrg-csf.sh" % xdir, ropts)

    def ex(self, args):
        """Execute job scripts on this node."""
        self.to_dir(dox="local")
        lr = self.lr_dirs()
        opts = {}
        optl = [] + list(opts.keys())
        opts.update(read_opts(args[2:], {}, optl))
        sec_key = "%s-%s" % (args[0], args[1])
        os.chdir('%s/runs/%s' % (lr[0], sec_key))
        l = "run.sh"
        cmd = '%s \"%s\"' % ("bash", l)
        print("%s %s/%s" % ("execute", sec_key, l))
        print(os.popen(cmd).read().strip())
        os.chdir("../..")

    def submit(self, args):
        """Submit job scripts for executing on compute nodes."""
        jcmd = ["sbatch", "scancel"]
        self.to_dir(dox="local")
        lr = self.lr_dirs()
        opts = {}
        optl = [ "exclude", "restart", "reservation", "pyscf-path", "block2-path",
            "block2-dmrg", "block2-dmrg-rev", "block2-dmrg-csf" ] + list(opts.keys())
        opts.update(read_opts(args[2:], {}, optl))
        sec_key = "%s-%s" % (args[0], args[1])
        os.chdir('%s/runs/%s' % (lr[0], sec_key))
        if "block2-dmrg" in opts:
            l = "block2-dmrg.sh"
        elif "block2-dmrg-rev" in opts:
            l = "block2-dmrg-rev.sh"
        elif "block2-dmrg-csf" in opts:
            l = "block2-dmrg-csf.sh"
        elif "restart" in opts:
            l = "restart.sh"
        else:
            l = "run.sh"
        if "exclude" in opts:
            cmd = "sed -i '3 i\\#SBATCH --exclude=%s' %s" % (opts["exclude"], l)
            print(os.popen(cmd).read().strip())
        if "reservation" in opts:
            cmd = "sed -i '3 i\\#SBATCH --reservation=%s' %s" % (opts["reservation"], l)
            print(os.popen(cmd).read().strip())
        if "pyscf-path" in opts:
            cmd = "sed -i '/which python3/a \\export PYTHONPATH=%s:$PYTHONPATH' %s" % (
                opts["pyscf-path"].replace("/", "\\/"), l)
            print(os.popen(cmd).read().strip())
        if "block2-path" in opts:
            cmd = "sed -i '/which python3/a \\export PYTHONPATH=%s:$PYTHONPATH' %s" % (
                opts["block2-path"].replace("/", "\\/"), l)
            print(os.popen(cmd).read().strip())
        cmd = '%s \"%s\"' % (jcmd[0], l)
        print("%s %s/%s" % ("submit", sec_key, l))
        print(os.popen(cmd).read().strip())
        os.chdir("../..")

    def set(self, args):
        pre = self.pre_info()
        self.to_dir(dox="local")
        sec_key = "%s-%s" % (args[0], args[1])
        args = args[2:]
        if args[0] == "none":
            del pre[sec_key]
            print("[args-%s] %s" % (sec_key, args[0]))
        elif args[0] == "args":
            odv = pre[sec_key][args[1]]
            carg = cast_value(odv, args[2])
            pre[sec_key][args[1]] = carg
            print("[args-%s] %s = %s => %s" % (sec_key, args[1], odv, carg))
        elif args[0] == "args+":
            pre[sec_key][args[1]] = args[2]
            print("[args-%s] %s += %s" % (sec_key, args[1], args[2]))
        elif args[0] == "args-":
            del pre[sec_key][args[1]]
            print("[args-%s] %s => none" % (sec_key, args[1]))
        else:
            raise RuntimeError("Wrong set operation name %s!" % args[0])
        write_json(pre, "./hife-parameters.json")
    
    def log(self, args):
        pre = self.pre_info()
        lr = self.lr_dirs()
        self.to_dir(dox="local")
        exact = False
        if len(args) == 1:
            sec_key = args[0]
        elif len(args) >= 2:
            sec_key = "%s-%s" % (args[0], args[1])
            if len(args) >= 3 and args[2] == "exact":
                exact = True
        else:
            sec_key = None
        for k, v in pre.items():
            if sec_key is not None and sec_key not in k:
                continue
            if exact and sec_key != k:
                continue
            if '-' not in k:
                continue
            extra = ''
            xff = None
            ex, tx, niter = '?', '', 0
            acte, acto = 0, 0
            ssq = ''
            txst = None
            bdim = None
            tswp = 0
            xf = "%s/runs/%s/OUTFILE" % (lr[0], k)
            if os.path.isfile(xf):
                xff = "%s/runs/%s/%s" % (lr[0], k, open(xf, "r").readlines()[-1].strip())
                if not os.path.isfile(xff):
                    xdd = xff.replace("hife.out", "dmrg.out")
                    if os.path.isfile(xdd):
                        xff = xdd
                if os.path.isfile(xff):
                    xg = open(xff, "r").readlines()
                    for xgl in xg:
                        if xgl.startswith("NAO"):
                            extra += " nao = " + xgl.split('=')[-1].strip()
                        elif xgl.startswith("NELEC") and "BEFORE" not in xgl and "AFTER" not in xgl:
                            extra += " nelec = %d" % sum([int(x) for x in xgl.split('=')[-1].strip()[1:-1].split(', ')])
                        elif "-> CAS" in xgl:
                            extra += " " + xgl.strip()
                        elif "converged SCF energy" in xgl:
                            ex = xgl.split("=")[1].split()[0]
                            if "<S^2> = " in xgl:
                                ssq = xgl.split("<S^2> = ")[-1].split()[0]
                        elif "SCF not converged" in xgl:
                            ex = "!!! NO CONV !!!"
                        elif ex == "!!! NO CONV !!!" and "SCF energy" in xgl:
                            ex += xgl.split("=")[1].split()[0]
                        elif xgl.startswith("EMP2     ="):
                            ex = " ".join(xgl.split())
                        elif xgl.startswith("ECCSD    ="):
                            ex = " ".join(xgl.split())
                        elif xgl.startswith("ECCSD(T) ="):
                            ex += " " + " ".join(xgl.split())
                        elif xgl.startswith("EBCCSD   ="):
                            ex += " " + " ".join(xgl.split())
                        elif xgl.startswith("EST    ="):
                            ex += " " + " ".join(xgl.split())
                        elif xgl.startswith("EST(T) ="):
                            ex += " " + " ".join(xgl.split())
                        elif xgl.startswith("Time sweep ="):
                            tswp += float(xgl.split("=")[1].split()[0])
                            niter += 1
                        elif "| Bond dimension =" in xgl:
                            bdim = xgl.split("| Bond dimension =")[1].split()[0]
                        elif xgl.startswith("DMRG Energy =") and bdim is not None:
                            ex = "EDMRG(M=%s) = " % bdim + xgl.split()[-1]
                        elif xgl.startswith("CASCI E =") and "CASSCF" not in ex:
                            ex = " ".join(xgl.split()[:4])
                            if "S^2 = " in xgl:
                                ssq = xgl.split("S^2 = ")[-1].split()[0]
                        elif xgl.startswith("CASSCF energy ="):
                            ex += " ECASSCF = " + xgl.split()[-1]
                        elif xgl.startswith("E(WickICNEVPT2)"):
                            ex += " ICNEV = " + xgl.split()[2] + " PT = " + xgl.split()[5]
                        elif xgl.startswith("E(WickSCNEVPT2)"):
                            ex += " SCNEV = " + xgl.split()[2] + " PT = " + xgl.split()[5]
                        elif xgl.startswith("E(WickICMRREPT2)"):
                            ex += " ICMRR = " + xgl.split()[2] + " PT = " + xgl.split()[5]
                        elif xgl.startswith("E(ICMRLCC)"):
                            ex += " ICMRL = " + xgl.split()[2] + " PT = " + xgl.split()[5]
                        elif xgl.startswith("Nevpt2 Energy ="):
                            eref = float(ex.split()[-1])
                            ept = float(xgl.split()[3])
                            ex += " SCNEV = %.16g PT = %.16g" % (eref + ept, ept)
                        elif xgl.startswith("TOTAL TIME"):
                            tx = xgl.split("=")[-1].strip()
                            tx = time_span_str(float(tx)) + " (%s)" % tx
                        elif xgl.startswith("START  TIME"):
                            from datetime import datetime
                            txst = datetime.strptime(xgl.split("=")[-1].strip(), "%m/%d/%Y %H:%M:%S")
                            tx = (datetime.now() - txst).total_seconds()
                            tx = time_span_str(tx) + " (%.3f) (running)" % tx
                        elif "cycle=" in xgl:
                            nxx = int(xgl.split()[1])
                            niter = max(nxx, niter)
                        elif xgl.startswith("cycle =") or xgl.startswith("macro iter"):
                            nxx = int(xgl.split()[2])
                            niter = max(nxx, niter)
                        elif xgl.startswith("NACTORB ="):
                            acto = int(xgl.split()[2])
                            acte = int(xgl.split()[5])
            print()
            if tx == '' and tswp != 0:
                tx = time_span_str(tswp) + " (%.3f)" % tswp
            if ssq != '':
                extra += ' S^2 = %s' % ssq
            if k.startswith("mf-"):
                print("[%s] :: %s/%s/%s :: charge = %s spin = %s%s"
                    % (k, v["method"], v["func"], v["basis"], v["charge"], v["spin"], extra))
            elif k.startswith("cc-"):
                if "level_shift" in v:
                    print("[%s] :: load = %s level_shift = %s%s" % (k, v["load_mf"], v["level_shift"], extra))
                elif "frozen" in v:
                    print("[%s] :: load = %s frozen = %s%s" % (k, v["load_mf"], v["frozen"], extra))
                else:
                    print("[%s] :: load = %s%s" % (k, v["load_mf"], extra))
            elif k.startswith("mp-"):
                if "non_canonical" in v:
                    print("[%s] :: load = %s non_canonical = T%s" % (k, v["load_mf"], extra))
                else:
                    print("[%s] :: load = %s%s" % (k, v["load_mf"], extra))
            elif k.startswith("stdmrg-"):
                print("[%s] :: load = %s%s" % (k, v["load_mf"], extra))
            elif "load_mf" in v:
                xx = "[%s] :: mf = %s" % (k, v["load_mf"])
                if "load_coeff" in v:
                    xx += " coeff = %s" % v["load_coeff"]
                if "method" in v:
                    xx += " method = %s" % v["method"]
                if "solver" in v:
                    xx += " solver = %s" % v["solver"]
                if "spin" in v:
                    xx += " spin = %s" % v["spin"]
                if "maxm" in v:
                    xx += " maxm = %s" % v["maxm"]
                if "block2-dmrg" in v:
                    xx += " b2-dmrg"
                if "cas_ccsd" in v:
                    xx += " cas_ccsd"
                if "cas_ccsd_t" in v:
                    xx += " cas_ccsd(t)"
                if "cas_uccsd" in v:
                    xx += " cas_uccsd"
                if "cas_uccsd_t" in v:
                    xx += " cas_uccsd(t)"
                if "level_shift" in v:
                    xx += " level_shift = %s" % v["level_shift"]
                print("%s%s" % (xx, extra))
            if k.startswith("orb-"):
                print("   PLOT --- cd %s/runs/%s; jmol orbs.spt; cd -" % (lr[0], k))
            if k.startswith("select-") and "cas_list" in v:
                print("   ACT %s --- (%do, %de)" % (v["cas_list"], acto, acte))
            if k.startswith("avas-"):
                print("   ACT %s --- (%do, %de)" % (v["ao_labels"], acto, acte))
            xf = "%s/runs/%s/JOBIDS" % (lr[0], k)
            jid = open(xf, "r").readlines()[-1].strip() if os.path.isfile(xf) else '?'
            if xff is not None:
                print("   FILE = %s JOBID = %s" % (xff, jid))
                if k.split("-")[0] not in ["orb", "select", "act", "avas"]:
                    if len(ex) > 50:
                        print("   E = %s\n   NITER = %d T = %s" % (ex, niter, tx))
                    else:
                        print("   E = %s NITER = %d T = %s" % (ex, niter, tx))

    def init(self, args):
        self.to_dir(dox="local")
        opts = {"method": "uks", "charge": "0", "spin": "0", "func": "tpss",
                "basis": "def2-SV(P)", "mem": "120G", "time": "48:00:00" }
        optl = ["no-scratch"] + list(opts.keys())
        if os.path.isfile("./hife-parameters.json"):
            para_temp = read_json("./hife-parameters.json")
            opts.update(para_temp["create"])
        else:
            para_temp = read_json(self.scripts_dir + "/hife-parameters.json")
        def_pos = {"0": "name", "1": "method"}
        optst = read_opts(args, def_pos, optl)
        opts.update(optst)
        for k in [ "charge", "spin" ]:
            if k in opts:
                opts[k] = int(opts[k])
        if "name" not in opts:
            raise RuntimeError("name argument must be set!")
        path_pwd = None
        if "no-scratch" not in opts and not os.path.isfile("./DIRECTORIES"):
            path_pwd = os.path.abspath(os.curdir)
            xname = opts["name"].replace("(", "").replace(")", "").replace("|", "")
            xname = xname.replace(":", "-").replace("*", "").replace("@", "")
            xbasis = opts["basis"].replace("(", "").replace(")", "").replace("|", "")
            path_remote = "%s-%s-%s" % (xname, xbasis, opts["method"])
            path_id = 0
            if "SCRATCH" in os.environ:
                path_scr = os.environ["SCRATCH"]
            elif "WORKDIR" in os.environ:
                path_scr = os.environ["WORKDIR"]
            else:
                raise RuntimeError("SCRATCH/WORKDIR directory not found in environ!")
            while os.path.exists("%s/%s.%d" % (path_scr, path_remote, path_id)):
                path_id += 1
            path_full = "%s/%s.%d" % (path_scr, path_remote, path_id)
            os.makedirs(path_full)
            f = open('./DIRECTORIES', 'w')
            f.write('%s\n%s\n' % (path_pwd, path_full))
            f.close()
            copyfile('./DIRECTORIES', path_full + "/DIRECTORIES")
        para_temp["create"] = {
            "name": opts["name"],
            "spin": opts["spin"],
            "charge": opts["charge"],
            "time": opts["time"],
            "func": opts["func"],
            "basis": opts["basis"],
            "method": opts["method"]
        }
        xhost = get_host()
        if "cores" in opts:
            xcores = int(opts["cores"])
        elif "hosts" in para_temp:
            xcores = para_temp["hosts"]["cores"]
        else:
            xcores = host_cores[xhost]
        if "partition" in opts:
            xpart = opts["partition"]
        else:
            xpart = host_part[xhost]
        if "nodes" in opts:
            xnodes = int(opts["nodes"])
        elif "nodes" in para_temp:
            xnodes = para_temp["hosts"]["nodes"]
        else:
            xnodes = 1
        para_temp["hosts"] = { "name": xhost, "model": xhost,
                               "cores": xcores, "nodes": xnodes,
                               "mem": opts["mem"], "partition": xpart }
        write_json(para_temp, "./hife-parameters.json")

    def clean(self, args):
        cmd = "mv CMD-HISTORY CMD.bak"
        os.popen(cmd).read()
        cmd = "rm -r CMD-HISTORY DIRECTORIES hife-parameters.json runs"
        os.popen(cmd).read()
