import sys
# from functools import cache
from typing import Callable, Any

import plotly.express as px
import pandas as pd
import pyRAPL
from pyRAPL.outputs import DataFrameOutput
from pyJoules.handler.pandas_handler import PandasHandler
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.rapl_device import RaplDramDomain
from loguru import logger


def cache(function):
    cached_runs = {}

    def wrapper(n):
        if n in cached_runs:
            return cached_runs[n]
        else:
            res = function(n)
            cached_runs[n] = res
            return res

    return wrapper


@cache
def factorial_cache(n):
    return n * factorial(n - 1) if n else 1


def factorial(n):
    return n * factorial(n - 1) if n else 1


def run_pyjoules_test(
        start_n: int,
        max_n: int,
        repeats: int,
        interval: int,
        test_func: Callable[[int], Any]
):
    current_n = start_n
    pyjoules_results = []
    for _ in range(repeats):
        pandas_handler = PandasHandler()
        with EnergyContext(
                handler=pandas_handler,
                domains=[
                    RaplPackageDomain(0),
                    RaplDramDomain(0)],
                start_tag='start') as ctx:
            while current_n < max_n:
                logger.debug(f"Testing value {current_n}")
                ctx.record(tag=str(current_n))
                test_func(current_n)
                ctx.record(tag='start')
                current_n += interval
            current_n = start_n

        run_res = pandas_handler.get_dataframe()
        run_res = run_res[run_res['tag'] != 'start']
        run_res['tag'] = run_res['tag'].astype(int)
        pyjoules_results.append(run_res)

    return pyjoules_results


def run_pyrapl_test(
        start_n: int,
        max_n: int,
        repeats: int,
        interval: int,
        test_func: Callable[[int], Any]
):
    pyRAPL.setup(devices=[pyRAPL.Device.PKG, pyRAPL.Device.DRAM], socket_ids=None)
    result_range = [x for x in range(start_n, max_n, interval)]
    current_n = start_n
    pyrapl_results = []
    for _ in range(repeats):
        run_result = DataFrameOutput()
        while current_n < max_n:
            logger.debug(f"Testing value {current_n}")
            with pyRAPL.Measurement('test', output=run_result):
                test_func(current_n)
            current_n += interval
        current_n = start_n

        result_data = run_result.data
        result_data['tag'] = result_range
        pyrapl_results.append(result_data)

    return pyrapl_results


def get_package_power():
    with open('/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as pfile:
        return int(pfile.readline())


def get_dram_power():
    with open('/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:2/energy_uj', 'r') as pfile:
        return int(pfile.readline())


def run_adhoc_test(
        start_n: int,
        max_n: int,
        repeats: int,
        interval: int,
        test_func: Callable[[int], Any]
):
    results = []
    current_n = start_n
    for _ in range(repeats):
        run_results = []
        while current_n < max_n:
            logger.debug(f"Testing value {current_n}")
            before_package = get_package_power()
            before_dram = get_dram_power()
            test_func(current_n)
            after_package = get_package_power()
            after_dram = get_dram_power()
            run_results.append((after_package - before_package, after_dram - before_dram, current_n))
            current_n += interval
        current_n = start_n
        results.append(pd.DataFrame(run_results, columns=['pkg', 'dram', 'tag']))

    return results


if __name__ == "__main__":
    max_n = 20000
    start_n = 1000
    runs = 100
    repeats = 10
    interval = (max_n - start_n) // runs
    sys.setrecursionlimit(max_n + 10)
    from dataflow.inefficient_data_processing import *

    results = run_adhoc_test(start_n, max_n, repeats, interval, factorial_cache)
    results = pd.concat(results, keys=[f'run_{i}' for i in range(repeats)])
    results = results.reset_index(names=['run', ''])
    print(results)
    fig = px.line(results, x='tag', y='pkg', color='run', range_y=[0, 2000000])
    fig.add_bar(x=results.tag, y=results.dram)
    fig.show()
