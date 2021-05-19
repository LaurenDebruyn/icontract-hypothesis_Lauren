##  Icontract -> IR/Symbol table/... (name can still change)

### What is supported?
* .. + | - | * | / | % .. (only in right-hand side of equations)
    * x1 < x2 + 3
* .. < | <= | > | >= | == ..
    * len(lst) == 4
* .. in ..
    * item in lst
* all( .. for .. in .. )
    * all( item > 0 for item in lst
* any( .. for .. in .. )
    * any( item > 0 for item in lst )
* nested quantifiers
    * all( all( .. for .. in .. ) for .. in .. )
* module.function[.function]*( .. )
    * regex.match(r’.*’, s)
* var.attribute(.attribute)( .. )
    * lst.contains(item)
* var[idx: int] >|>=|<|<= .. (comparisons with tuples/lists with single slice)
    * t[0] < 0
* len(..) compare ..
    * len(lst) > 0
* comparing tuples (left hand side are all variables)
    * (a, b) > (0, 0)
* dictionaries (keys and values same type!)
    * .. for .. in dictionar.values()
    * .. for .. in dictionar.keys()
### What is not supported?
* .. + | - | * | / | % .. (in left-hand side of equations)
    * x1 + 4 < x2
* all( .. for .. in .. if ..)
    * all( item > 0 for item in lst if item < 100 )
* .. or ..
    * not lst or lst.contains(item)
* .. if .. else ..
    * item > 0 if bool else item < 0
* .. for .., .. in enumerate(..)
    * item < idx for idx, item in enumerate(lst)
* left-hand boundaries
    * 0 < x
* tuples/lists with slices
    * tuple[0:2] > (0,0)
* comparisons with functions that are not len()
    * do_something(n1) < 0
* not ..
    * not n1 < 0
    * not s.startswith(‘abc’)
    * -> can all be rewritten (I assume)
* dictionaries
    * everything that is not covered in [What is supported?](#what-is-supported)
### What do I still want to support if I have enough time?
* Dictionaries with different key and value type
### Which types are supported?
* int
* string
* float
* List[int]
* List[string]
* tuple
* dictionaries (partially)


# Design decisions

## Symbolic strategies

We could use the SearchStrategies as they are defined in Hypothesis(._internatls.numbers.IntegerStrategy),
but this would limit us to only passing actual values (here integers) to the strategy.
An example where this would fail us if we have a formula n1 < n2, this would not be expressible.

A second option is to represent everything as string, but this quickly becomes hard to handle.
Especially validation and extensibility are hard in this approach.

The third option, which we think is the best one, is to introduce a new strategy for every Hypothesis strategy.
These strategies would act as a placeholder for the actual strategies, while allowing symbolic variables.
We therefore call these strategies *symbolic strategies*.
Symbolic strategies will allow us to easily add new bounds, values, ... and easily transform them into strings
that we can use for code generation.
Next to storing the information on the arguments of the actual strategy,
we will also use them to store extra information that will be necessary to generate complete strategies.
In a first stage, this will be limited to storing all the filters that have to be applied to this strategy.
This can be extended in later stages to example map's and flatmap's.

## What happens with contracts we cannot parse?

### not in property table
***TODO***

### not in strategy
***TODO***

## Approach for handling 'link', 'universal_quantifier' and 'existential_quantifier'

1. generate_strategies(table: Table)  
This will remain the same. For each function argument exists one unique BASE row.
We will start from these rows and incorporate the other rows recursively.

2. infer_strategy(row: Row)
    1. Is going to be extended with an extra case for lists (other types that should be supported will be added here).
    2. Each case will be extended to cater for the different possibilities: links, universal quantifiers and existential quantifiers.
    3. At the end of the function are we going to iterate over all related rows and merge them all into one strategy.
    How are we going to this? We will pass the parent strategy on and this strategy will be extended with the information from the new rows.
    This problem can be divided into three classes:
        * links
            * similar to how the base strategies work, only we will now extend a strategy instead of creating a new one
        * universal quantifiers
            * again three different options (note that in this case, there will always be a parent strategy)
                * the parent has already a sub-strategy of the same type (int, string, list, ...), then this sub-strategy will be simply extended
                * the parent has already a sub-strategy, but of a different type, then a new strategy will be created and the two strategies are merged
                * the parent has not yet a sub-strategy, then a new strategy will be created 
        * existential quantifiers 
            * these will simply be converted into filters to the parent strategy


# Feature ideas

## Specifying specific examples
[Hypothesis - reproducing failures](https://hypothesis.readthedocs.io/en/latest/reproducing.html)


