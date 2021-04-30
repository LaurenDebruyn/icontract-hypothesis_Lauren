import ast
import os.path
import textwrap
from re import sub


def generate_test_suite_from_file(filepath: str) -> None:
    if filepath.startswith("./"):
        module_path = filepath[2:-3].replace("/", ".")
    elif filepath.startswith("../"):
        module_path = filepath[3:-3].replace("/", ".")
    else:
        module_path = filepath.replace("/", ".")
    module_name = module_path.split(".")[-1]

    # function to convert string to camelCase
    def camelcase(string: str) -> str:
        string = sub(r"[_-]+", " ", string).title().replace(" ", "")
        return string

    with open(filepath) as file:
        node = ast.parse(file.read())

    functions = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

    test_file = ""

    test_file += f'import {module_path}\n'
    test_file += 'import unittest\n'
    test_file += 'from icontract_hypothesis_Lauren.generate_symbol_table import generate_symbol_table\n'

    test_file += f'\n\nclass {camelcase(module_name.capitalize())}Test(unittest.TestCase):\n'

    for function in functions:
        test_func = textwrap.dedent(
            f"""\
            def test_{function}(self) -> None:
                try:
                    generate_symbol_table({module_path}.{function})
                except Exception:
                    raise self.failureException
            """
        ).strip()
        test_func_indented = ''
        for line in test_func.split('\n'):
            test_func_indented += f'    {line}\n'
        test_file += f'\n{test_func_indented}'

    if not os.path.isdir('tests'):
        os.mkdir('tests')
    f = open(f'tests/test_{module_name}.py', "w")
    f.write(test_file)
    f.close()
