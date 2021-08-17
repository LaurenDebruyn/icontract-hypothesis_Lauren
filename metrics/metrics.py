import importlib
import inspect
import os
import sys
import typing
from pathlib import Path

import icontract.errors
# import icontract_hypothesis
import python_by_contract_corpus.ethz_eprog_2019.exercise_02.problem_02
import regex as re
import textwrap
import pkgutil
import hypothesis  # noqa
import hypothesis.errors
from typing import *  # noqa

from icontract._checkers import find_checker, _unpack_pre_snap_posts, add_precondition_to_checker, decorate_with_checker
from python_by_contract_corpus import aoc2020, ethz_eprog_2019
from python_by_contract_corpus.aoc2020 import day_10_adapter_array
from python_by_contract_corpus.ethz_eprog_2019 import exercise_05, exercise_06, exercise_08, \
    exercise_09, exercise_11, exercise_12

from datetime import date
import decimal

from python_by_contract_corpus.ethz_eprog_2019.exercise_02.problem_02 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_02.problem_03 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_02.problem_05_01 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_02.problem_05_02 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_03.problem_01 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_03.problem_02 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_03.problem_03 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_03.problem_04 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_04.problem_01 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_04.problem_02 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_04.problem_03 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_05.problem_03 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_05.problem_04 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_06.problem_01 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_06.problem_04 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_06.problem_05 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_07.problem_02 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_07.problem_04 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_08.problem_01 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_08.problem_02 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_08.problem_03 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_08.problem_05 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_09.problem_02 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_09.problem_03 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_09.problem_04 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_11.problem_01 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_11.problem_02 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_12.problem_01 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_12.problem_03 import *
from python_by_contract_corpus.ethz_eprog_2019.exercise_12.problem_04 import *

from python_by_contract_corpus.ethz_eprog_2019.exercise_05 import problem_03, problem_04
from python_by_contract_corpus.ethz_eprog_2019.exercise_06 import problem_04
from python_by_contract_corpus.ethz_eprog_2019.exercise_08 import problem_03, problem_05
from python_by_contract_corpus.ethz_eprog_2019.exercise_09 import problem_03
from python_by_contract_corpus.ethz_eprog_2019.exercise_11 import problem_01, problem_02
from python_by_contract_corpus.ethz_eprog_2019.exercise_12 import problem_01, problem_03, problem_04

# from python_by_contract_corpus.aoc2020 import day_3_toboggan_trajectory, day_8_handheld_halting, day_10_adapter_array, \
#     day_11_seating_system, day_12_rain_risk, day_13_shuttle_search, day_14_docking_data, day_16_ticket_translation, \
#     day_17_conway_cubes, \
#     day_18_operation_order, day_19_monster_messages, day_20_jurassic_jigsaw, day_21_allergen_assessment, \
#     day_22_crab_combat, day_23_crab_cups, day_24_lobby_layout
from python_by_contract_corpus.aoc2020.day_1_report_repair import *
from python_by_contract_corpus.aoc2020.day_2_password_philosophy import *
from python_by_contract_corpus.aoc2020.day_3_toboggan_trajectory import *
from python_by_contract_corpus.aoc2020.day_4_passport_processing import *
from python_by_contract_corpus.aoc2020.day_5_binary_boarding import *
from python_by_contract_corpus.aoc2020.day_6_custom_customs import *
from python_by_contract_corpus.aoc2020.day_7_handy_haversacks import *
from python_by_contract_corpus.aoc2020.day_8_handheld_halting import *
from python_by_contract_corpus.aoc2020.day_9_encoding_error import *
from python_by_contract_corpus.aoc2020.day_10_adapter_array import *
from python_by_contract_corpus.aoc2020.day_11_seating_system import *
from python_by_contract_corpus.aoc2020.day_12_rain_risk import *
from python_by_contract_corpus.aoc2020.day_13_shuttle_search import *
from python_by_contract_corpus.aoc2020.day_14_docking_data import *
from python_by_contract_corpus.aoc2020.day_15_rambunctious_recitation import *
from python_by_contract_corpus.aoc2020.day_16_ticket_translation import *
from python_by_contract_corpus.aoc2020.day_17_conway_cubes import *
from python_by_contract_corpus.aoc2020.day_18_operation_order import *
from python_by_contract_corpus.aoc2020.day_19_monster_messages import *
from python_by_contract_corpus.aoc2020.day_20_jurassic_jigsaw import *
from python_by_contract_corpus.aoc2020.day_21_allergen_assessment import *
from python_by_contract_corpus.aoc2020.day_22_crab_combat import *
from python_by_contract_corpus.aoc2020.day_23_crab_cups import *
from python_by_contract_corpus.aoc2020.day_24_lobby_layout import *
from python_by_contract_corpus.aoc2020.day_25_combo_breaker import *

from itertools import chain
import tabulate

import pandas as pd
from icontract_hypothesis_Lauren.strategy_factory import StrategyFactory, hook_into_icontract_and_hypothesis


hypothesis.settings.register_profile("custom", max_examples=100, verbosity=hypothesis.Verbosity.quiet)
hypothesis.settings.load_profile("custom")

USE_OLD_VERSION_IC = False

if not USE_OLD_VERSION_IC:
    hook_into_icontract_and_hypothesis(localns=locals(), globalns=globals())

if USE_OLD_VERSION_IC and not 'icontract-hypothesis' not in sys.modules:
    importlib.import_module('icontract-hypothesis')

COUNTER = 0

test_folder = Path("icontract_hypothesis_Lauren/tests/test_input/")

integers_functions_file = test_folder / "integers" / "integers_functions.py"


RE_ASSUME = re.compile(r'\s\s\s\shypothesis.assume\((.|\n)*\)')
RE_FILTER = re.compile(r'(.|\n)*\.filter\((.|\n)*\).*')


def test_function(function_name: str, strategy: str, multiple_arguments: bool) -> None:
    if multiple_arguments:
        test_code = textwrap.dedent(f"""{strategy}
@hypothesis.given(args=strategy_{function_name}())
def test_{function_name}(args):
    {function_name}(*args)

test_{function_name}()
""")
    else:
        test_code = textwrap.dedent(f"""{strategy}
@hypothesis.given(args=strategy_{function_name}())
def test_{function_name}(args):
    {function_name}(args)

test_{function_name}()
""")
    exec(test_code)


def prepare_function(func: Callable) -> Callable:
    checker = find_checker(func)
    if not checker:
        checker = decorate_with_checker(func)
    preconditions, _, _ = _unpack_pre_snap_posts(checker)

    sign = inspect.signature(func).replace(return_annotation=inspect.Signature.empty)
    new_func_str = textwrap.dedent(f"""
def {func.__name__}{sign}:
    global COUNTER
    COUNTER += 1
    pass
""").replace('~', '')
    exec(new_func_str)
    new_func = decorate_with_checker(locals()[func.__name__])
    for contract_group in preconditions:
        for contract in contract_group:
            add_precondition_to_checker(new_func, contract)
    globals()[func.__name__] = new_func
    return new_func


# TODO rename
def test_module(module: Any) -> pd.DataFrame:
    assert(inspect.ismodule(module))
    if module.__name__ not in sys.modules:
        importlib.import_module(module.__name__)
    functions = [f[1].__name__ for f in inspect.getmembers(module) if inspect.isfunction(f[1])]

    global COUNTER
    print(f"MUT: {module.__name__}")

    modules = []
    function_names = []
    runs = []
    assume_free = []
    filter_free = []
    nb_samples = []
    failed_health_check = []
    violation_error = []
    exceptions = []
    for function_name in functions:
        modules.append(module.__name__)
        function_names.append(function_name)
        print(f"FUT: {function_name}")

        COUNTER = 0

        # add the local classes to the 'globals()' namespace
        local_members = {name: obj for name, obj in inspect.getmembers(module)}
        for cls_name, cls in local_members.items():
            globals()[cls_name] = cls

        function = prepare_function(getattr(module, function_name))
        globals().update({function_name: function})

        strategy_str = ""
        try:
            add_run = False
            if USE_OLD_VERSION_IC:
                strategy_str = str(icontract_hypothesis.infer_strategy(function))
                icontract_hypothesis.test_with_inferred_strategy(function,
                                                                 localns=locals(),
                                                                 globalns=globals())
                add_run = True
            else:
                sf = StrategyFactory(function, localns=locals(), globalns=globals())
                strategy_str = sf.generate_composite_strategy()
                if strategy_str:
                    # an empty strategy means that there was nothing we could generate
                    #  for example when no type hints were given
                    #  these cases are not taken into account
                    # check if it runs correctly
                    if len(sf._get_type_hints()) > 1:
                        test_function(function_name, strategy_str, multiple_arguments=True)
                    else:
                        test_function(function_name, strategy_str, multiple_arguments=False)
                    add_run = True
                else:
                    modules.pop()
                    function_names.pop()

            if add_run:
                runs.append(True)
                # check if it uses 'assume'
                uses_assume = any(RE_ASSUME.match(line) for line in strategy_str.split("\n"))
                assume_free.append(not uses_assume)
                # check if it uses 'filter'
                uses_filter = True if RE_FILTER.match(strategy_str) else False
                filter_free.append(not uses_filter)
                nb_samples.append(COUNTER)
                failed_health_check.append(False)
                violation_error.append(False)
                exceptions.append(None)
        except hypothesis.errors.MultipleFailures as e:
            runs.append(True)
            if strategy_str:
                # check if it uses 'assume'
                uses_assume = any(RE_ASSUME.match(line) for line in strategy_str.split("\n"))
                assume_free.append(not uses_assume)
                # check if it uses 'filter'
                uses_filter = True if RE_FILTER.match(strategy_str) else False
                filter_free.append(not uses_filter)
            else:
                assume_free.append(False)
                filter_free.append(False)
            nb_samples.append(COUNTER)
            failed_health_check.append(True)
            violation_error.append(False)
            exceptions.append(None)
        except hypothesis.errors.FailedHealthCheck as e:
            runs.append(False)
            # check if it uses 'assume'
            uses_assume = any(RE_ASSUME.match(line) for line in strategy_str.split("\n"))
            assume_free.append(not uses_assume)
            # check if it uses 'filter'
            uses_filter = True if RE_FILTER.match(strategy_str) else False
            filter_free.append(not uses_filter)
            nb_samples.append(COUNTER)
            failed_health_check.append(True)
            violation_error.append(False)
            exceptions.append(e)
        except icontract.errors.ViolationError as e:
            runs.append(False)
            # check if it uses 'assume'
            uses_assume = any(RE_ASSUME.match(line) for line in strategy_str.split("\n"))
            assume_free.append(not uses_assume)
            # check if it uses 'filter'
            uses_filter = True if RE_FILTER.match(strategy_str) else False
            filter_free.append(not uses_filter)
            nb_samples.append(COUNTER)
            failed_health_check.append(False)
            violation_error.append(True)
            exceptions.append(e)
        except Exception as e:
            runs.append(False)
            # check if it uses 'assume'
            uses_assume = any(RE_ASSUME.match(line) for line in strategy_str.split("\n"))
            assume_free.append(not uses_assume)
            # check if it uses 'filter'
            uses_filter = True if RE_FILTER.match(strategy_str) else False
            filter_free.append(not uses_filter)
            nb_samples.append(COUNTER)
            failed_health_check.append(False)
            violation_error.append(False)
            exceptions.append(f"{type(e)} :     {e}")
    df = pd.DataFrame(
        data={
            'module': modules,
            'function_name': function_names,
            'runs': runs,
            'assume_free': assume_free,
            'filter_free': filter_free,
            'nb_samples': nb_samples,
            'failed_health_check': failed_health_check,
            'violation_error': violation_error,
            'exceptions': exceptions
        }
    )
    return df


def generate_metrics_df(modules: Generator[Any, Any, None]) -> pd.DataFrame:
    df = pd.DataFrame()
    for module in modules:
        df = df.append(test_module(module), ignore_index=True)
    return df


def calculate_metrics_detailed(modules: Generator[Any, Any, None]):
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
            # nb samples: {sub_df['nb_samples'].sum()}
            # failed health check: {sub_df['failed_health_check'].sum()}
            # violation error: {sub_df['violation_error'].sum()}
            """))


def calculate_metrics(modules: Generator[Any, Any, None]):
    df = generate_metrics_df(modules)

    print(textwrap.dedent(
        f"""\
        -- metrics general overview --
        total # tests: {len(df)}
        # successful runs: {df['runs'].sum()}
        # assume free: {df['assume_free'].sum()}
        # filter free: {df['filter_free'].sum()}
        # nb samples: {df['nb_samples'].sum()}
        # failed health check: {df['failed_health_check'].sum()}
        # violation_error: {df['violation_error'].sum()}
        """))


def argument_types(function: Any) -> Dict[str, int]:
    type_hint_counter = dict()
    for type_hint in typing.get_type_hints(function).values():
        if typing.get_origin(type_hint):
            if isinstance(typing.get_origin(type_hint), typing._SpecialForm):
                type_hint_origin = typing.get_origin(type_hint)._name
            else:
                type_hint_origin = typing.get_origin(type_hint).__name__

            if type_hint_origin in type_hint_counter:
                type_hint_counter[type_hint_origin] = type_hint_counter[type_hint_origin] + 1
            else:
                type_hint_counter[type_hint_origin] = 1
            for sub_type_hint in typing.get_args(type_hint):
                if typing.get_origin(sub_type_hint):
                    if isinstance(typing.get_origin(sub_type_hint), typing._SpecialForm):
                        sub_type_hint_origin = typing.get_origin(sub_type_hint)._name
                    else:
                        sub_type_hint_origin = typing.get_origin(sub_type_hint).__name__
                    if sub_type_hint_origin in type_hint_counter:
                        type_hint_counter[sub_type_hint_origin] = type_hint_counter[sub_type_hint_origin] + 1
                    else:
                        type_hint_counter[sub_type_hint_origin] = 1
                    for sub_sub_type_hint in typing.get_args(sub_type_hint):
                        if isinstance(typing.get_origin(sub_sub_type_hint), typing._SpecialForm):
                            sub_sub_type_hint = sub_sub_type_hint._name
                        else:
                            sub_sub_type_hint = sub_sub_type_hint.__name__
                        if sub_sub_type_hint in type_hint_counter:
                            type_hint_counter[sub_sub_type_hint] = type_hint_counter[sub_sub_type_hint] + 1
                        else:
                            type_hint_counter[sub_sub_type_hint] = 1
                else:
                    if isinstance(typing.get_origin(sub_type_hint), typing._SpecialForm):
                        sub_type_hint = sub_type_hint._name
                    else:
                        sub_type_hint = sub_type_hint.__name__
                    if sub_type_hint in type_hint_counter:
                        type_hint_counter[sub_type_hint] = type_hint_counter[sub_type_hint] + 1
                    else:
                        type_hint_counter[sub_type_hint] = 1
        else:
            if isinstance(typing.get_origin(type_hint), typing._SpecialForm):
                type_hint = type_hint._name
            else:
                type_hint = type_hint.__name__
            if type_hint in type_hint_counter:
                type_hint_counter[type_hint] = type_hint_counter[type_hint] + 1
            else:
                type_hint_counter[type_hint] = 1
    return type_hint_counter


def argument_type_statistics(modules: Generator[Any, Any, None]):
    type_hint_counter = dict()
    for module in modules:
        functions = [f[1] for f in inspect.getmembers(module) if inspect.isfunction(f[1])]
        for function in functions:
            type_hint_counter_function = argument_types(function)
            for k, v in type_hint_counter_function.items():
                if k in type_hint_counter:
                    type_hint_counter[k] = type_hint_counter[k] + v
                else:
                    type_hint_counter[k] = v
    # table_data = [
    #     [f"{k.__module__}.{k.__name__}", v] if inspect.isclass(k) else [f"{k.__module__}.{k}", v]
    #     for k, v in type_hint_counter.items()
    # ]
    table_data = [
        [k, v]
        for k, v in type_hint_counter.items()
    ]
    # print(tabulate.tabulate(table_data))
    return pd.DataFrame(table_data)


def argument_type_per_function(modules: Generator[Any, Any, None]) -> pd.DataFrame:
    df = pd.DataFrame()
    for module in modules:
        functions = [f[1] for f in inspect.getmembers(module) if inspect.isfunction(f[1])]
        for function in functions:
            type_hints = set(argument_types(function).keys())
            new_row_data = dict()
            new_row_data['module'] = module.__name__
            new_row_data['function_name'] = function.__name__
            for type_hint in type_hints:
                new_row_data[type_hint] = 1
            new_row = pd.Series(data=new_row_data)
            df = df.append(new_row, ignore_index=True)
    return df.fillna(0)


def ethz_module_generator():
    for exercise_loader, exercise_name, _ in pkgutil.iter_modules(ethz_eprog_2019.__path__):
        _exercise_module = exercise_loader.find_module(exercise_name).load_module(exercise_name)
        for problem_loader, problem_name, _ in pkgutil.iter_modules(_exercise_module.__path__):
            if problem_name.startswith('problem'):
                problem_name = f"{exercise_name}.{problem_name}"
                if f"python_by_contract_corpus.ethz_eprog_2019.{problem_name}" in sys.modules:
                    yield sys.modules[f"python_by_contract_corpus.ethz_eprog_2019.{problem_name}"]
                else:
                    _problem_module = problem_loader.find_module(problem_name).load_module(problem_name)
                    yield _problem_module


def aoc2020_module_generator():
    for problem_loader, problem_name, _ in pkgutil.iter_modules(aoc2020.__path__):
        if f"python_by_contract_corpus.aoc2020.{problem_name}" in sys.modules:
            _problem_module = sys.modules[f"python_by_contract_corpus.aoc2020.{problem_name}"]
        else:
            _problem_module = problem_loader.find_module(problem_name).load_module(problem_name)
        yield _problem_module


# csv = generate_metrics_df(get_aoc2020_modules()).to_csv('24_07_2021_aoc2020.csv')
# calculate_metrics(get_aoc2020_modules())
# calculate_metrics(ethz_module_generator())
# csv = generate_metrics_df(get_ethz_modules()).to_csv('24_07_2021_ethz.csv')

# argument_type_statistics(chain(ethz_module_generator(), aoc2020_module_generator())).to_csv('argument_type_statistics.csv')
# print(argument_type_statistics(chain(ethz_module_generator(), aoc2020_module_generator())))
# argument_type_per_function(chain(ethz_module_generator(), aoc2020_module_generator())).to_csv("type_hints_per_function.csv")
# csv = generate_metrics_df(ethz_module_generator()).to_csv('ethz_new_IC_2.csv')
# csv = generate_metrics_df(aoc2020_module_generator()).to_csv('aoc2020_new_IC.csv_2.csv')
# print(test_module(day_10_adapter_array)[['function_name', 'nb_samples', 'exceptions']].to_string())
# icontract_hypothesis.test_with_inferred_strategy(python_by_contract_corpus.ethz_eprog_2019.exercise_02.problem_02.draw)
# generate_metrics_df(chain(ethz_module_generator(), aoc2020_module_generator())).to_csv("IC_results_2.csv")
# print(generate_metrics_df(ethz_module_generator()))
calculate_metrics(chain(ethz_module_generator(), aoc2020_module_generator()))
