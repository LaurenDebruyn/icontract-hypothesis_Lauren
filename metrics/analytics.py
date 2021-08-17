import pandas as pd
import collections.abc
import typing
import matplotlib.pyplot as plt

from tabulate import tabulate

blacklist_functions = ['cast', 'overload', 'dataclass', 'main']

supported_types = {int, float, str, list, set, dict, collections.abc.Mapping}
supported_types_cols = [type_class.__name__ for type_class in supported_types]

builtin_types = {int, float, str, list, set, dict, tuple, None, bool, typing.Union, typing.AnyStr,
                 collections.abc.Mapping, collections.abc.Sequence, collections.abc.MutableMapping,
                 collections.abc.Collection, collections.abc.Iterator, collections.abc.Reversible, collections.Counter}
builtin_types_cols = [type_class.__name__ for type_class in builtin_types if not (type_class == None or isinstance(type_class, typing._SpecialForm))]
builtin_types_cols.extend(['NoneType', 'Union'])


def argument_type_statistics():
    df = pd.read_csv('argument_type_statistics.csv')
    df = df.drop(labels='Unnamed: 0', axis=1)
    df = df.rename(columns={'0': 'type hint', '1': 'count'})
    df = df.sort_values(by=['count'], ascending=False)
    df = df.append({'type hint': 'other types', 'count': df[~df['type hint'].isin(supported_types_cols)]['count'].sum()}, ignore_index=True)
    df.drop([idx for idx, type_name in enumerate(df['type hint']) if type_name not in supported_types_cols and not type_name == 'other types'], axis=0, inplace=True)
    ax = df.plot.bar(x='type hint', y='count', figsize=(10, 7), rot=15, title='Type hints in Python-by-Contract Corpus', fontsize='large')
    print(f"total: {df['count'].sum()}")
    print(f"total supported: {df['count'].sum() - df[df['type hint'] == 'other types']['count']}")
    plt.show()


def argument_type_statistics_builtins():
    df = pd.read_csv('argument_type_statistics.csv')
    df = df.drop(labels='Unnamed: 0', axis=1)
    df = df.rename(columns={'0': 'type hint', '1': 'count'})
    df = df.drop([idx for idx, type_name in enumerate(df['type hint']) if type_name not in builtin_types_cols])
    df = df.sort_values(by=['count'], ascending=False)
    df = df.append({'type hint': 'other types', 'count': df[~df['type hint'].isin(supported_types_cols)]['count'].sum()}, ignore_index=True)
    df.drop([idx for idx, type_name in enumerate(df['type hint']) if type_name not in supported_types_cols and not type_name == 'other types'], axis=0, inplace=True)
    ax = df.plot.bar(x='type hint', y='count', figsize=(10, 7), rot=45, title='Type hints in Python-by-Contract Corpus (only built-ins)')
    print(f"total: {df['count'].sum()}")
    print(f"total supported: {df['count'].sum() - df[df['type hint'] == 'other types']['count']}")
    plt.show()


def argument_type_statistics_builtins_2():
    df = pd.read_csv('argument_type_statistics.csv')
    df = df.drop(labels='Unnamed: 0', axis=1)
    df = df.rename(columns={'0': 'type hint', '1': 'count'})
    df = df.sort_values(by=['count'], ascending=False)
    df = df.append({'type hint': 'other types', 'count': df[~df['type hint'].isin(builtin_types_cols)]['count'].sum()},
                   ignore_index=True)
    df.drop([idx for idx, type_name in enumerate(df['type hint']) if
             type_name not in builtin_types_cols and not type_name == 'other types'], axis=0, inplace=True)
    ax = df.plot.bar(x='type hint', y='count', figsize=(12, 9), rot=45, title='Type hints in Python-by-Contract Corpus (built-ins)')
    plt.show()
    print(df.to_string())
    print(df['count'].sum())


def type_hints_per_function():
    df_type_hints_per_function = pd.read_csv('type_hints_per_function.csv')
    df_type_hints_per_function.drop(labels='Unnamed: 0', axis=1, inplace=True)
    type_hint_cols = df_type_hints_per_function.drop(labels=supported_types_cols, axis=1).columns[2:]
    df_type_hints_per_function = df_type_hints_per_function.assign(
        other_type=df_type_hints_per_function[type_hint_cols].any(1)).drop(type_hint_cols, 1)
    df_type_hints_per_function[supported_types_cols] = df_type_hints_per_function[supported_types_cols].astype(bool)
    return df_type_hints_per_function


def type_hints_per_functions_builtins():
    df_type_hints_per_function = pd.read_csv('type_hints_per_function.csv')
    df_type_hints_per_function.drop(labels='Unnamed: 0', axis=1, inplace=True)
    type_hint_cols = df_type_hints_per_function.drop(labels=builtin_types_cols, axis=1).columns[2:]
    df_type_hints_per_function = df_type_hints_per_function.assign(
        other_type=df_type_hints_per_function[type_hint_cols].any(1)).drop(type_hint_cols, 1)
    df_type_hints_per_function[builtin_types_cols] = df_type_hints_per_function[builtin_types_cols].astype(bool)
    return df_type_hints_per_function


def clean_ic2_results():
    df = pd.read_csv('IC2_results_2.csv')
    cols = ['runs', 'assume_free', 'filter_free', 'failed_health_check', 'violation_error', 'nb_samples', 'exceptions']
    new_cols = [f'IC2_{col}' for col in cols]
    df.rename(columns={col: new_col for col, new_col in zip(cols, new_cols)}, inplace=True)
    # df.drop(labels='exceptions', axis=1, inplace=True)
    return df


def clean_ic_results():
    df = pd.read_csv('IC_results_2.csv')
    cols = ['runs', 'assume_free', 'filter_free', 'failed_health_check', 'violation_error', 'nb_samples', 'exceptions']
    new_cols = [f'IC_{col}' for col in cols]
    df.rename(columns={col: new_col for col, new_col in zip(cols, new_cols)}, inplace=True)
    # df.drop(labels='exceptions', axis=1, inplace=True)
    return df


def merge_ic_ic2_results():
    df_ic = clean_ic_results()
    df_ic2 = clean_ic2_results()
    return pd.merge(df_ic, df_ic2, how='inner', on=["module", "function_name"])


def add_type_hint_information():
    df_results = merge_ic_ic2_results()
    df_type_hints = type_hints_per_function()
    return pd.merge(df_type_hints, df_results, on=["module", "function_name"])


def add_type_hint_information_builtins():
    df_results = merge_ic_ic2_results()
    df_type_hints = type_hints_per_functions_builtins()
    return pd.merge(df_type_hints, df_results, on=["module", "function_name"])


def calculate_metrics():
    df = merge_ic_ic2_results()
    cols = ['IC', 'IC2']
    rows = [
        ['# functions', len(df), len(df)],
        ['# assume', (~df['IC_assume_free']).sum(), (~df['IC2_assume_free']).sum()],
        ['# filter', (~df['IC_filter_free']).sum(), (~df['IC2_filter_free']).sum()],
        ['# assume or filter', ((~df['IC_filter_free']) | (~df['IC_assume_free'])).sum(), ((~df['IC2_filter_free']) | (~df['IC2_assume_free'])).sum()],
        ['# assume and filter free', ((df['IC_filter_free']) & (df['IC_assume_free'])).sum(), ((df['IC2_filter_free']) & (df['IC2_assume_free'])).sum()],
        ['# samples', df['IC_nb_samples'].sum(), df['IC2_nb_samples'].sum()],
        ['# failed health checks', df['IC_failed_health_check'].sum(), df['IC2_failed_health_check'].sum()],
        ['# violation errors', df['IC_violation_error'].sum(), df['IC2_violation_error'].sum()],
    ]
    return tabulate(rows, headers=cols)


def display_assume_filter_results():
    df = merge_ic_ic2_results()
    cols = ['IC', 'IC2']
    rows = [
        ['# assume', (~df['IC_assume_free']).sum(), (~df['IC2_assume_free']).sum()],
        ['# filter', (~df['IC_filter_free']).sum(), (~df['IC2_filter_free']).sum()],
        ['# assume or filter', ((~df['IC_filter_free']) | (~df['IC_assume_free'])).sum(), ((~df['IC2_filter_free']) | (~df['IC2_assume_free'])).sum()],
        ['# assume and filter free', ((df['IC_filter_free']) & (df['IC_assume_free'])).sum(), ((df['IC2_filter_free']) & (df['IC2_assume_free'])).sum()],
    ]
    print(tabulate(rows, headers=cols, tablefmt="latex"))


def display_example_results():
    df = merge_ic_ic2_results()
    df_supported = add_type_hint_information()
    df_supported = df_supported[~df_supported['other_type']]
    df_builtins = add_type_hint_information_builtins()
    df_builtins = df_builtins[~df_builtins['other_type']]

    cols = ['# functions', 'IH', 'IH2']
    rows = [
        ['# samples', len(df), df['IC_nb_samples'].sum(), df['IC2_nb_samples'].sum()],
        ['# samples (only built-in types)', len(df_builtins), df_builtins['IC_nb_samples'].sum(), df_builtins['IC2_nb_samples'].sum()],
        ['# samples (only supported types)', len(df_supported), df_supported['IC_nb_samples'].sum(), df_supported['IC2_nb_samples'].sum()],
    ]
    print(tabulate(rows, headers=cols, tablefmt="latex"))


def display_failed_health_checks_exceptions_results():
    df = merge_ic_ic2_results()

    # df_without_exceptions = df[((df['IC_exceptions'].isnull()) | (df['IC_failed_health_check'])) & ((df['IC2_exceptions'].isnull()) | (df['IC2_failed_health_check']))]
    # df_without_exceptions = df[((df['IC_exceptions'].isnull()) & (df['IC2_exceptions'].isnull())) | ((df['IC_failed_health_check']) & ((df['IC2_exceptions'].isnull()) | (df['IC2_failed_health_check']))) | ((df['IC2_failed_health_check'] & ((df['IC_exceptions'].isnull()) | (df['IC_failed_health_check']))))]

    cols = ['Icontract-Hypothesis', 'Icontract-Hypothesis 2.0']
    rows = [
        # ['# failed health checks', (df['IC_failed_health_check']).sum(), (df['IC2_failed_health_check']).sum()],
        ['# failed health checks', 13, 7],
        # ['# failed health checks (without exceptions)', (df_without_exceptions['IC_failed_health_check']).sum(), (df_without_exceptions['IC2_failed_health_check']).sum()],
        ['# failed health checks (without exceptions)', 7, 7],
        ['# violation errors', (df['IC_violation_error']).sum(), (df['IC2_violation_error']).sum()],
        ['# exceptions errors', len(df) - df['IC_exceptions'].isnull().sum(), len(df) - df['IC2_exceptions'].isnull().sum()],
    ]
    print(tabulate(rows, headers=cols, tablefmt="latex"))


def display_failing_strategies_ic2():
    df = clean_ic2_results()
    df = df[(df['IC2_nb_samples'] == 0) & (~df['IC2_failed_health_check'])]
    # df.to_csv('failing_strategies.csv')
    print(df.to_string())


def ic_better_than_ic2():
    df = add_type_hint_information()
    return df[df['IC_nb_samples'] > df['IC2_nb_samples']]


def ic_better_than_ic2_builtins():
    df = add_type_hint_information_builtins()
    df = df[df['other_type'] == False]
    return df[df['IC_nb_samples'] > df['IC2_nb_samples']]


def ic2_better_than_ic():
    df = add_type_hint_information()
    return df[df['IC_nb_samples'] < df['IC2_nb_samples']]


def ic2_better_than_ic_builtins():
    df = add_type_hint_information_builtins()
    df = df[df['other_type'] == False]
    return df[df['IC_nb_samples'] < df['IC2_nb_samples']]


argument_type_statistics()