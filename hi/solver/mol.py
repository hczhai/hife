
TIME_ST = """
import time
from datetime import datetime
txst = time.perf_counter()
print("START  TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
"""

TIME_ED = """
txed = time.perf_counter()
print("FINISH TIME = ", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
print("TOTAL TIME  = %20.3f" % (txed - txst))
"""

MOL = """
from pyscf import gto, scf
mol = gto.Mole()
mol.verbose = 4
mol.atom = '''\n%s\n'''
mol.basis = %s
mol.spin = %s
mol.charge = %s
"""

MOL_FINAL = """
mol.build()
print("NAO   = ", mol.nao)
print("NELEC = ", mol.nelec)
"""

def handle_io_error(f):
    def ff(*args, **kwargs):
        while True:
            try:
                r = f(*args, **kwargs)
            except BlockingIOError as err:
                print(err)
            else:
                break
        return r
    return ff
