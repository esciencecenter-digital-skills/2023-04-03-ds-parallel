![](https://i.imgur.com/iywjz8s.png)

# Collaborative Document

2023-04-04 Parallel Python (day 3).

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [https://codimd.carpentries.org/9dSwX5EqSGObKgLTwJf_mA](https://codimd.carpentries.org/9dSwX5EqSGObKgLTwJf_mA)

Collaborative Document day 1: [tinyurl.com/parallel-python-april-2023](https://tinyurl.com/parallel-python-april-2023)

Collaborative Document day 2: [tinyurl.com/parallel-python-april-2023-2](https://tinyurl.com/parallel-python-april-2023-2)

Collaborative Document day 3: [tinyurl.com/parallel-python-april-2023-3](https://tinyurl.com/parallel-python-april-2023-3)

Collaborative Document day 4: [tinyurl.com/parallel-python-april-2023-4](https://tinyurl.com/parallel-python-april-2023-4) 

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## 🎓 Certificate of attendance

If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, raise your hand in zoom. Click on the icon labeled "Reactions" in the toolbar on the bottom center of your screen,
then click the button 'Raise Hand ✋'. For urgent questions, just unmute and speak up!

You can also ask questions or type 'I need help' in the chat window and helpers will try to help you.
Please note it is not necessary to monitor the chat - the helpers will make sure that relevant questions are addressed in a plenary way.
(By the way, off-topic questions will still be answered in the chat)


## 🖥 Workshop website

[link](https://esciencecenter-digital-skills.github.io/2023-04-03-ds-parallel/)

🛠 Setup

[gh:esciencecenter-digital-skills/parallel-python-workshop](https://github.com/esciencenter-digital-skills/parallel-python-workshop/)

## 👩‍🏫👩‍💻🎓 Instructors

Johan Hidding, Jaro Camphuijsen

## 🧑‍🙋 Helpers

Flavio Hafner, Laura Ootes

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda

| Time  | Topic                         |
| ----- | ----------------------------- |
| 09:00 | Welcome, icebreaker and recap |
| 09:15 | Delayed evaluation            |
| 10:30 | Coffee break                  |
| 10:45 | Map and reduce                |
| 12:00 | Tea break                     |
| 12:15 | Asyncio Part 1 ?              |
| 12:45 | Wrap-up                       |
| 13:00 | END                           |

## 🔧 Exercises
### Exercise 1
**Given this workflow **
```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, -3)
```
**Visualize and compute y_p and z_p separately, how often do you think x_p is evaluated? Run the workflow to check your answer.** 

**For the second workflow:**
```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
z_p.visualize(rankdir="LR")
```
**We pass the yet uncomputed promise x_p to both y_p and z_p. Now, only compute z_p, how often do you expect x_p to be evaluated? Run the workflow to check your answer.
Do you understand the difference between the two workflows?**

#### Answers


- 1st case: x_p is evaluated twice (once for y_p once for z_p), second case: x_p is calculated only once (it's in the same dependency chain) :+1::+1::+1::+1::+1::+1::+1:
- x_p gets evaluated twice in both examples

### Exercise 2
**Can you describe what the gather function does in terms of lists and promises? hint: Suppose I have a list of promises, what does gather allow me to do?**


- allows you to allocate exactly the amount of memory you will need for a list? Because you delay making the list until you know what goes into it
- you can put the promises in the dependency chain, such that you can reuse the same results from the list if you want to do a more complex calculation, you can use the list as a kind of buffer for the results
- Gather makes sure that all promises are done (each promise can be done in parallel but gather is the waiting point for all promises) 
- Gather allows all provided promises to be computed using a single line of code (and dask hopefully computes them in parallel) and provides the results in a list+1
- don't keep large lists in memory but compute them only once needed?
- Allows for all promises to be evaluated in parallel then passes the list of promises (return or downstream) when everything is evaluated (some kinda synchronization)
- gather creates a list of promises, so that the promises can be computed in parallel by dask :+1: 
- Consumes promises to provide an overview of multiple workflows; could show dependency or lack thereof allowing for parallel computation


- using lists, the gather function tries to communicate between different elements of a list (threads)

#### Solution
It turns a list of promises into a promise of a list.

### Exercise 3
**Write a delayed function that computes the mean of its arguments. Use it to esimates pi several times and returns the mean of the results.**

```python
mean(1, 2, 3, 4).compute()
```
**Make sure that the entire computation is contained in a single promise.**


#### Answers

##### Room 1


```python
@delayed
def mean(*args):
    return sum(args) / len(args)
    
pi_p = mean(*(calc_pi(10**7) for i in range(10)))
pi_p.compute()

```



##### Room 2
```python
import random
from dask import delayed
import numba

@delayed
def mean(*args):
    return sum(args) / len(args)

@delayed
@numba.njit(nogil=True)
def calc_pi(N):
    """Computes the value of pi using N random samples."""
    # Area of the square: 4r^2; Area of the circle: \pi r^2
    M = 0
    for i in range(N):
        # Take a sample.
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if x**2 + y**2 < 1 : M += 1
    return 4 * M / N

N = 10**7
C = 10

mean(*[calc_pi(N) for n in range(C)]).compute()
```

##### Room 3
```python
@delayed
def mean(*args):
    return sum(args)/len(args)

pi_arr = [calc_pi(100000) for i in range(100)]
mean(*pi_arr).compute()
```






##### Room 4


```python

```


##### Room 5


```python
@delayed
def mean_func(*args):
    l = list(args)
    return sum(l)/len(l)

mean_func(*(calc_pi(10**6) for n in range(10))).visualize()
mean_func(*(calc_pi(10**6) for n in range(10))).compute()
```

### Exercise 4
**Open the [Dask documentation on bags](https://docs.dask.org/en/latest/bag-api.html). Discuss the map, filter, flatten and reduction methods.**
- **What do they do?**
- **How would you use it?**

**NB. Why is reduction a bit special?**

##### room 1
map: maps function (e.g. lambda) over bags
filter: Filter out elements in collection with certain condition (predicate function), similar like boolean mask.
flatten: Create one long list from vectors or matrices 
reduction: an N to 1 operation where you first apply a function to a part of the data and then an aggregate function where you apply another or the same function to the chunks separately

NB: you can do the other operations with reduction

##### room 2

map: Just executes a function to elements of different lists consecutively
filter: it filters elements from a collection that satisfy a certain criteria (in the example given, where the elements in a sequence or list are even)
flatten: if you have a collection of collections, it flattens it once. So a list of [[1, 2], 3] will return [1,2,3], but if it would have been [[1,2, [3]]] it will return [1,2,[3]]
reduction: applies the first function (perpartition) to all partitions of the list/sequence, and the aggregate function will be applied to the result of that function on those partitions. 

##### room 3

map: apply function iteratively on iterable
filter: apply indexing condition on iterable
flatten: reduce dimension of array to 1
reduction: map plus aggregrate

##### room 4

##### room 5
map: apply a function elementwise / along a row / along a column in a dask dataframe
reduction: apply an aggregate function to a data partition / bag / group of data and reduce the size of the data to say a dask Series
flatten: apply a flattening algorithm to sparse lists / matrix
filter: filters elements based on the input conditions

Applications: apply user functions to data, flatten to simplify data and increase speed, filter to filter out chunks of data, etc. 

### Exercise 5
**Without executing it, try to forecast what would be the output of bag.map(pred).compute().**

Thijs - [True, True, True, False, True] :+2::+1::+1:

Bob - [True, True, True, False, True]+1
Berend - Would return True, True, True, False, True

### Exercise 6
***We previously discussed some generic operations on bags. In the documentation, lookup the pluck method. How would you implement this if pluck wasn’t there?***

hint: Try pluck on some example data.

```python
from dask import bag as db

data = [
   { "name": "John", "age": 42 },
   { "name": "Mary", "age": 35 },
   { "name": "Paul", "age": 78 },
   { "name": "Julia", "age": 10 }
]

bag = db.from_sequence(data)
...
```

#### Room 1

#### Room 2

```python
from dask import bag as db

data = [
   { "name": "John", "age": 42 },
   { "name": "Mary", "age": 35 },
   { "name": "Paul", "age": 78 },
   { "name": "Julia", "age": 10 }
]

bag = db.from_sequence(data)

# Return the value n of x
def pluck_value(x, key):
    return x.get(key)

bag.map(pluck_value, 'name').compute()
```

#### Room 3
we can use Filter 
#### Room 4

#### Room 5
b = db.from_sequence([{'name': 'Alice', 'credits': [1, 2, 3]},
                      {'name': 'Bob',   'credits': [10, 20]}]
def pluck(x,key):
    return x[key]

db.map(pluck,b,"credits").compute()

### Exercise 7
***Use map and mean functions on Dask bags to compute π.***

```python
import random

def calc_pi(N):
    """Computes the value of pi using N random samples"""
    M = 0
    for i in range(N):
        # take a sample
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y < 1.: M+=1
    return 4 * M /N
```
Hint 1: Think about what your input data that you can put in a bag and on which you can apply the calc_pi function using bag.map().
Hint 2: The `repeat` method in numpy can come in useful to generate that input data: `from numpy import repeat`

#### Room 1
```python
import random
from numpy import repeat
def mean(*args):
    return sum(args)/len(args)
def calc_pi_simple(N):
    """Computes the value of pi using N random samples."""
    c = 0
    r= 1
    for i in range(N):
        # take a sample
        x = random.uniform(-1*r,r)
        y = random.uniform(-1*r,r)
        if x**2 + y**2 <=1:
            c +=1
    pi = 4*c/N
        
    return pi
import dask.bag as db
my_bag = db.from_sequence(repeat(10**3, 10))
my_bag.reduction(calc_pi_simple, mean, split_every=1).compute()
```python

#### Room 2
```python
n_runs = 10
n_sample_list = (10**7 for _ in range(n_runs))

bag = db.from_sequence(n_sample_list)
bag.map(calc_pi).compute()

bag.reduction(calc_pi, mean, split_every=n_runs).compute()

(gave an error...)
``` 
#### Room 3
import numpy as np
bag = db.from_sequence(np.arange(1,10)*(10**6), partition_size=4)

bag.map(calc_pi).compute()


#### Room 4

(gave error)
```python
import random

def in_circle():
    """C"""
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if x*x + y*y < 1.: 
        return 4/10**7
    else:
        return 0

bag = db.from_sequence([1 for i in range(10**7)])
est = bag.reduction(in_circle, sum)

```

#### Room 5
(But the .compute() does not work)
```python
def mean(*args):
    return np.mean(args)
    
bag = db.from_sequence([10**6 for i in range(10)],
                      partition_size = 2)
bag.reduction(calc_pi, mean).compute()
bag.reduction(calc_pi, mean).visualize()


## 🧠 Collaborative Notes

### What we've learned upto now

* Concepts
    - chunking
    - dependency diagram
    - natural parallelism
* Benchmarking
    - timing your code
    - checking memory consumption
* First make things fast
    - numpy vectorization
    - numba just-in-time
* Then make things run parallel
    - threading
        - OS dependent internal mechanism for running functions concurrently (i.e. `fork`)
        - `-` need to lift the GIL somehow
        - `+` low on resources
        - `+` shared memory
    - multiprocessing
        - Starts a new process (like from the shell) and sets up communication channels in other ways
        - `-` large overhead
        - `-` need for data serialisation
        - `-` complicates shared memory
        - `-` less flexible: for instance no lambdas
        - `+` circumvents the GIL


### Delayed evaluation
```python
from dask import delayed
```
Delay is a way to delay what you want to do. Sometimes it is better to delay a process.

```python
@delayed
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")
    return result
```

```python
add(1, 2)
```
This returns a delay function. It stores the requested function call inside a promise. The function is not actually executed yet, instead we are promised a value that can be computed later. In this way you are building up a workflow of functions. Once compute is called, Dask can then figure out which processes can be executed in parallel, it will determine the most efficient way to run your workflow.

```python
# We can check that x_p is now a Delayed value.
type(add(1, 2))
```

```python
# create a promise variable
x_p = add(1, 2)
```

```python
x_p.compute()
```

```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
```

```python
# install missing graphviz
pip install graphviz
```

(if the visualisation gives an error, graphviz has to be installed first)
```python
# visualize your workflow
z_p.visualize(rankdir="LR")
```

### Decorators
```python
import random
import numba 
from multiprocessing import Process, Queue

@numba.njit(nogil=True)
def calc_pi(N):
    """Computes the value of pi using N random samples"""
    
    M = 0
    for i in range(N):
        # simulate impact coordinates
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        # True if impact inside the circle
        if x**2 + y**2 < 1.0:
            M += 1
    
    return 4 * M / N 
```

```python
N = 10**7
pi_p = delayed(calc_pi)(N)
```

```python
pi_p.compute()
```

```python
@delayed
def add(*args):
    return sum(args)
```

```python
add(1, 2, 3, 4)
```

```python
numbers = [1, 2, 3, 4]
add(*numbers)
```

```python
# try with and without delayed
@delayed
def gather(*args):
    return list(args)
```

```python
gather(1, 2, 3, 4)
```

```python
x_p = add(1, 2)
y_p = add(2, 3)

# show the visualisation of the workflow
gather(x_p, y_p).visualize()
```


```python
# to get the actual results
gather(x_p, y_p).compute()
```

```python
plist = [x_p, y_p]

gather(*plist).compute()
```

```python
x_p = gather(*(add(n, n) for n in range(10)))
x_p.visualize()
```

```python
x_p.compute()
# shows you that the results in the list are ordered
```

```python
@delayed
def mean(*args):
    return sum(args) / len(args)

mean(1, 2, 3, 4).compute()
```

```python
pi_work = (delayed(calc_pi)(10**7) for _ in range(10))
```

```python
mean(*pi_work)
```

```python
pi_p.visualize()
```

```python
%%time

pi_p.compute()
```

```python
%%time

calc_pi(10**8)
```

```python

def repeat_and_mean(f, n, *args):
    return mean(*(delayed(f)(*args) for _ in range(n)))
```

```python
repeat_and_mean(calc_pi, 10, 10**7).compute()
```

### Map and reduce
#### Map

| Dask module      | Abstraction          | Keywords                            | Covered |
|:---------------- |:-------------------- |:----------------------------------- |:------- |
| `dask.array`     | `numpy`              | Numerical analysis                  | ✔️      |
| `dask.bag`       | `itertools`          | Map-reduce, workflows               | ✔️      |
| `dask.delayed`   | functions            | Anything that doesn't fit the above | ✔️      |
| `dask.dataframe` | `pandas`             | Generic data analysis               | ❌      |
| `dask.futures`   | `concurrent.futures` | Control execution, low-level        | ❌      |

Dask bags let you compose functionality using several primitive patterns: the most important of these are:
```python
map
filter
groupy
flatten
reduction
```
Dask bags documentation
https://docs.dask.org/en/latest/bag-api.html

#### Map
```python
# Create the bag containing the elements we want to work with
import dask.bag as db
bag = db.from_sequence(["mary", "had", "a", "little", "lamb"], partition_size = 2)
```

```python
bag.visualize()
```

```python
# see what is in the bag
bag.compute()
```

Map applies a function one-to-one on a list of arguments 
```python
# Create a function for mapping
def f(x):
    return x.upper()

# create the map and compute it
bag.map(f).compute()
```

```python
bag.map(f).visualize()
```

#### Filter
Filter is used to select elements based on a defined condition
```python
# Return True is x contains 'a', False if not
def pred(x):
    return 'a' in x

bag.filter(pred).compute()
```

```python
bag.map(pred).compute()
```

#### Reduction
```python
def count_chars(x):
    per_word = [len(w) for w in x]
    
    return sum(per_word)

bag.reduction(count_chars, sum).visualize()
```

```python
bag.reduction(count_chars, sum).compute()
```

```python
from dask import bag as db

data = [
    { "name": "John", "age": 42},
    { "name": "Mary", "age": 35},
    { "name": "Paul", "age": 78},
    { "name": "Julia", "age": 10}
]

bag = db.from_sequence(data)

# Return the value n of x
def pluck_value(x, key):
    return x.get(key)

bag.map(pluck_value, 'name').compute()
```

```python
bag.pluck("name").compute()
```

```python
bag.map(lambda x: x["name"]).compute()
```

```python
import random
from numpy import repeat

def calc_pi(N):
    """Computes the value of pi using N random samples"""
    M = 0
    for i in range(N):
        # take a sample
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y < 1.: M+=1
    return 4 * M /N

bag = db.from_sequence(repeat(10**5, 100), partition_size=10)
pies = bag.map(calc_pi)
estimate = pies.mean()
```

```python
estimate.visualize()
```

```python
estimate.compute()
```

Check the difference in wall time
```python
%%time
estimate.compute()
```

```python
%%time
calc_pi(10**7)
```


## 📚 Resources

- [Why $\pi$ is in the normal distribution](https://youtu.be/cy8r7WSuT1I)

- [Line profiler](https://pypi.org/project/line-profiler/)
"line_profiler will profile the time individual lines of code take to execute. The profiler is implemented in C via Cython in order to reduce the overhead of profiling.
Also included is the script kernprof.py which can be used to conveniently profile Python applications and scripts either with line_profiler or with the function-level profiling tools in the Python standard library."


- [**Running notebooks with nbconvert**](https://nbconvert.readthedocs.io/en/latest/usage.html#convert-notebook)
```bash 
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 --execute my_notebook.ipynb
```
 
 - [Decorators](https://pythonguide.readthedocs.io/en/latest/python/decorator.html)


## Feedback 
### Positive
- Good pace, good session :+1::+1::+1:
- Very interesting content, nice pace
- nice to see how rapidly you implement feedback from the previous days
- Multiple approaches covered, very nice!
- Useful tools and approaches to learning
- Learned some nice new methods to use

### To be improved 
- sometimes exercises felt a bit unclear, maybe try jumping around the breakout rooms to stimulate the discussions
- to be honest breakout rooms don't seem to be working as many people are approaching this as an individual exercise
    - (other responder) so perhaps try to stimulate the discussion somehow (I do like the breakout rooms!)
- for me, with the relatively simplistic examples, it feels difficult to implement this for my own code, because I don't really know when one method would/should be used over another. 
- Perhaps extrapolate how the functionality can be used in practise. 
- A few examples of real life implementations and how one tool might offer benefits over 
- The lecture is very interactive this way, but perhaps it would also be really nice to have slides (with examples in the slides) to have also some more theory background, rather than you telling it during the implementation of the code (while you are writing it).
