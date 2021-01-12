import dill
from multiprocessing import Pool

# from https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def dill_map(cpu_count, fun, args):
    payloads = [dill.dumps((fun, arg)) for arg in args]
    with Pool(cpu_count) as pool:
        ret = pool.map(run_dill_encoded, payloads)
    return ret