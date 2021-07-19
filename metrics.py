import importlib
import inspect
import os
from pathlib import Path
import regex as re
import textwrap
import pkgutil
import hypothesis  # noqa
import hypothesis.errors
from typing import *  # noqa
from python_by_contract_corpus import aoc2020, ethz_eprog_2019


import pandas as pd
from icontract_hypothesis_Lauren.strategy_factory import StrategyFactory
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

test_folder = Path("icontract_hypothesis_Lauren/tests/test_input/")

integers_functions_file = test_folder / "integers" / "integers_functions.py"


RE_ASSUME = re.compile(r'\s\s\s\sassume\((.|\n)*\)')
RE_FILTER = re.compile(r'(.|\n)*\.filter\((.|\n)*\).*')


def test_function(function_name: str, strategy: str) -> bool:
    test_code = textwrap.dedent(f"""{strategy}
    @hypothesis.given(args=strategy_{function_name})
    def test_{function_name}(args):
        {function_name}(*args)

    test_{function_name}()
    """)
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(e)
        return False


# TODO rename
def test_module(module: Any) -> pd.DataFrame:
    assert(inspect.ismodule(module))
    functions = [f[1].__name__ for f in inspect.getmembers(module) if inspect.isfunction(f[1])]
    modules = []
    function_names = []
    runs = []
    assume_free = []
    filter_free = []
    failed_health_check = []
    exceptions = []
    for function_name in functions:
        # modules.append(module.__name__.split('.')[-1])
        modules.append(module.__name__)
        function_names.append(function_name)
        function = getattr(module, function_name)
        # generate strategy
        sf = StrategyFactory(function)
        try:
            strategy_str = sf.generate_composite_strategy()
            # check if it runs correctly
            runs_correctly = test_function(function_name, strategy_str)
            runs.append(runs_correctly)
            # check if it uses 'assume'
            uses_assume = any(RE_ASSUME.match(line) for line in strategy_str.split("\n"))
            assume_free.append(not uses_assume)
            # check if it uses 'filter'
            uses_filter = True if RE_FILTER.match(strategy_str) else False
            filter_free.append(not uses_filter)
            failed_health_check.append(False)
            exceptions.append(None)
        except hypothesis.errors.FailedHealthCheck as e:
            runs.append(False)
            assume_free.append(False)
            filter_free.append(False)
            failed_health_check.append(True)
            exceptions.append(e)
        except Exception as e:
            runs.append(False)
            assume_free.append(False)
            filter_free.append(False)
            failed_health_check.append(False)
            exceptions.append(f"{type(e)} :     {e}")
        # print(f"{function_name}  {runs_correctly}  {uses_assume}  {uses_filter}")
    df = pd.DataFrame(
        data={
            'module': modules,
            'function_name': function_names,
            'runs': runs,
            'assume_free': assume_free,
            'filter_free': filter_free,
            'failed_health_check': failed_health_check,
            'exceptions': exceptions
        }
    )
    return df


def generate_metrics_df(modules: List[Any]) -> pd.DataFrame:
    assert(all(inspect.ismodule(module) for module in modules))
    df = pd.DataFrame()
    for module in modules:
        df = df.append(test_module(module), ignore_index=True)
    return df


def calculate_metrics_detailed(modules: List[Any]):
    assert(all(inspect.ismodule(module) for module in modules))
    df = generate_metrics_df(modules)

    for module_name in df['module'].unique():
        sub_df = df.loc[df['module'] == module_name]
        print(textwrap.dedent(
            f"""\
            -- metrics {module_name} --
            total # tests: {len(sub_df)}
            # successful runs: {sub_df['runs'].sum()}
            # assume free: {sub_df['assume_free'].sum()}
            # filter free: {sub_df['filter_free'].sum()}
            # failed health check: {sub_df['failed_health_check'].sum()}
            """))


def calculate_metrics(modules: List[Any]):
    assert (all(inspect.ismodule(module) for module in modules))
    df = generate_metrics_df(modules)

    print(textwrap.dedent(
        f"""\
        -- metrics general overview --
        total # tests: {len(df)}
        # successful runs: {df['runs'].sum()}
        # assume free: {df['assume_free'].sum()}
        # filter free: {df['filter_free'].sum()}
        # failed health check: {df['failed_health_check'].sum()}
        """))


def get_ethz_modules():
    modules = []
    for exercise_loader, exercise_name, _ in pkgutil.iter_modules(ethz_eprog_2019.__path__):
        _exercise_module = exercise_loader.find_module(exercise_name).load_module(exercise_name)
        for problem_loader, problem_name, _ in pkgutil.iter_modules(_exercise_module.__path__):
            if problem_name.startswith('problem'):
                problem_name = f"{exercise_name}.{problem_name}"
                _problem_module = problem_loader.find_module(problem_name).load_module(problem_name)
                modules.append(_problem_module)
    return modules


def get_aoc2020_modules():
    modules = []
    for problem_loader, problem_name, _ in pkgutil.iter_modules(aoc2020.__path__):
        _problem_module = problem_loader.find_module(problem_name).load_module(problem_name)
        modules.append(_problem_module)
    return modules

# csv = generate_metrics_df(get_aoc2020_modules()).to_csv('17_07_2021_aoc2020.csv')
calculate_metrics(get_aoc2020_modules())
calculate_metrics(get_ethz_modules())
