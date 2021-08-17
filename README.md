# Contract-Based Testing in Python with Icontract and Hypothesis

This project has been created as part of Lauren De bruyn's master's thesis at KU Leuven.

Icontract-Hypothesis Lauren (called Icontract-Hypothesis 2.0 in the master's thesis) continues on the work of Marko Ristin's project [Icontract-Hypothesis](https://github.com/mristin/icontract-hypothesis).

## Structure of the repository

**generate_property_table.py**
* The module that holds data structures for the Property Table and algorithms to turn Icontract contracts into property tables.

**property_table_to_strategies.py**
* The module that holds the data structures for the Symbolic Strategies and the algorithms for the strategy matching.

**strategy_factory.py**
* The module responsible for creating textual and actual composite Hypothesis strategies.
* Also holds methods to get property tables and separate strategies.

**/metrics**
* This folders contains all the data that is used for the evaluation in the master's thesis.
* Also the code used to obtain the data and the visualizations can be found in this folder.

**/tests**
* Tests for the three main modules can be found in this folder.