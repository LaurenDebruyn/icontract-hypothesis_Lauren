import importlib
# import inspect
import os
from pathlib import Path
# import regex as re
# import textwrap
# import hypothesis  # noqa
# from typing import *  # noqa

import pandas as pd
# from icontract_hypothesis_Lauren.strategy_factory import StrategyFactory
# from icontract_hypothesis_Lauren.tests.test_input.dictionaries import dictionaries_functions
# from icontract_hypothesis_Lauren.tests.test_input.integers import integers_functions
# from icontract_hypothesis_Lauren.tests.test_input.lists import lists_functions
# from icontract_hypothesis_Lauren.tests.test_input.regex import regex_functions
# from icontract_hypothesis_Lauren.tests.test_input.text import text_functions
# from icontract_hypothesis_Lauren.tests.test_input.tuples import tuples_functions


# modules_to_test = [
#     dictionaries_functions,
#     integers_functions,
#     lists_functions,
#     regex_functions,
#     text_functions,
#     tuples_functions,
# ]
#
# test_folder = Path("icontract_hypothesis_Lauren/tests/test_input/")
#
# integers_functions_file = test_folder / "integers" / "integers_functions.py"
#
#
# RE_ASSUME = re.compile(r'\s\s\s\sassume\((.|\n)*\)')
# RE_FILTER = re.compile(r'(.|\n)*\.filter\((.|\n)*\).*')
#
#
# def test_function(function_name: str, strategy: str) -> bool:
#     test_code = textwrap.dedent(f"""{strategy}
#     @hypothesis.given(args=strategy_{function_name})
#     def test_{function_name}(args):
#         {function_name}(*args)
#
#     test_{function_name}()
#     """)
#     try:
#         exec(test_code)
#         return True
#     except Exception as e:
#         print(e)
#         return False
#
#
# # TODO rename
# def test_module(module: Any) -> pd.DataFrame:
#     assert(inspect.ismodule(module))
#     functions = [f[1].__name__ for f in inspect.getmembers(module) if inspect.isfunction(f[1])]
#     modules = []
#     function_names = []
#     runs = []
#     assume_free = []
#     filter_free = []
#     for function_name in functions:
#         modules.append(module.__name__.split('.')[-1])
#         function_names.append(function_name)
#         function = getattr(module, function_name)
#         # generate strategy
#         sf = StrategyFactory(function)
#         try:
#             strategy_str = sf.generate_composite_strategy()
#             # check if it runs correctly
#             runs_correctly = test_function(function_name, strategy_str)
#             runs.append(runs_correctly)
#             # check if it uses 'assume'
#             uses_assume = any(RE_ASSUME.match(line) for line in strategy_str.split("\n"))
#             assume_free.append(not uses_assume)
#             # check if it uses 'filter'
#             uses_filter = True if RE_FILTER.match(strategy_str) else False
#             filter_free.append(not uses_filter)
#         except Exception:
#             runs.append(False)
#             assume_free.append(False)
#             filter_free.append(False)
#         # print(f"{function_name}  {runs_correctly}  {uses_assume}  {uses_filter}")
#     df = pd.DataFrame(data={'module': modules, 'function_name': function_names, 'runs': runs, 'assume_free': assume_free, 'filter_free': filter_free})
#     return df
#
#
# def calculate_metrics_detailed(modules: List[Any]):
#     assert(all(inspect.ismodule(module) for module in modules))
#     df = pd.DataFrame()
#     for module in modules:
#         df = df.append(test_module(module), ignore_index=True)
#
#     for module_name in df['module'].unique():
#         sub_df = df.loc[df['module'] == module_name]
#         print(textwrap.dedent(
#             f"""\
#             -- metrics {module_name} --
#             total # tests: {len(sub_df)}
#             # successful runs: {sub_df['runs'].sum()}
#             # assume free: {sub_df['assume_free'].sum()}
#             # filter free: {sub_df['filter_free'].sum()}
#             """))
#
#
# def calculate_metrics(modules: List[Any]):
#     assert (all(inspect.ismodule(module) for module in modules))
#     df = pd.DataFrame()
#     for module in modules:
#         df = df.append(test_module(module), ignore_index=True)
#     print(df)
#
#     print(textwrap.dedent(
#         f"""\
#         -- metrics general overview --
#         total # tests: {len(df)}
#         # successful runs: {df['runs'].sum()}
#         # assume free: {df['assume_free'].sum()}
#         # filter free: {df['filter_free'].sum()}
#         """))


def test_ethz():
    ethz_path: Path = Path("corpus/python_by_contract_corpus/ethz_eprog_2019/")
    dirs = [ethz_path / d for d in os.listdir(ethz_path) if os.path.isdir(ethz_path / d) and d.startswith('exercise')]
    for d in dirs:
        files = [f for f in os.listdir(d) if os.path.isfile(d / f) and not f == '__init__.py']
        package = str(d).replace('/', '.')
        for file in files:
            module_name = '.' + file.rstrip('.py')
            print(f"{package} {module_name}")
            importlib.import_module(module_name, package)

test_ethz()
# print(importlib.import_module('python-by-contract-corpus.python_by_contract_corpus.ethz_eprog_2019.exercise_12.problem_04'))
# print(importlib.import_module(name='.problem_04', package='python-by-contract-corpus.python_by_contract_corpus.ethz_eprog_2019.exercise_12'))
