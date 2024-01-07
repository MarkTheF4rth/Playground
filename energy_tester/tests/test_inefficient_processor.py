import functools
from collections import defaultdict
import pandas as pd
import pytest
import plotly.express as px
import plotly.graph_objects as go
from ..dataflow.efficient_data_processing import *
from ..dataflow.inefficient_data_processing import get_brute_force_average
import cProfile

def get_package_power():
    with open('/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj', 'r') as pfile:
        return int(pfile.readline())


def get_dram_power():
    with open('/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:2/energy_uj', 'r') as pfile:
        return int(pfile.readline())


class TestInefficientProcessor:
    results = defaultdict(lambda: [])

    @classmethod
    def teardown_class(cls):
        results = [(name, pd.DataFrame(result, columns=['pkg', 'dram', 'tag'])) for name, result in cls.results.items()]
        fig = go.Figure()

        for name, result in results:
            fig.add_trace(go.Scatter(x=result.index, y=result.pkg, name=name + 'cpu'))
            fig.add_trace(go.Bar(x=result.index, y=result.dram, name=name + 'mem'))

        fig.update_yaxes(title_text="Logarithmic Energy Usage (μJ)", type="log")
        fig.update_xaxes(title_text="Test Number")
        fig.update_layout(barmode='stack')
        fig.show()

    @pytest.fixture(scope="function")
    def test_energy_usage(self, request):
        before_package = get_package_power()
        before_dram = get_dram_power()
        yield
        after_package = get_package_power()
        after_dram = get_dram_power()
        self.results[request.node.originalname].append(
            (after_package - before_package, after_dram - before_dram, request.node.name))

    # @pytest.mark.parametrize('n', list(range(100, 100000, 1000)))
    # def test_creation(self, n, test_energy_usage):
    #     df = create_data(n)
    #     assert df.shape[0] == n

    @pytest.mark.parametrize('df', [create_data(n) for n in range(10, 10000, 1000)])
    def test_average(self, df, test_energy_usage):
        print('STARTING TEST')
        print(locals())
        print(globals())
        print('RUNNING PROFILER')
        cProfile.runctx("get_average(df)", globals=globals(), locals=locals())

    # @pytest.mark.parametrize('df', [create_data(n) for n in range(100, 100000, 1000)])
    # def test_average(self, df, test_energy_usage):
    #     df = get_brute_force_average(df)
