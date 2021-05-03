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
