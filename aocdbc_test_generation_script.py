from os import listdir
from os.path import isfile, join

from icontract_hypothesis_Lauren.generate_test_suite_from_file import generate_test_suite_from_file

path = '../python_by_contract_corpus/correct_programs/aoc2020'

files = [f for f in listdir(path) if isfile(join(path, f)) and f.startswith('day')]

print('\n'.join(files))

for file in files:
    filepath = f"{path}/{file}"
    generate_test_suite_from_file(filepath)
