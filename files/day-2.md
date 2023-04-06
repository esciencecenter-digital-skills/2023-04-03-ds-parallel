![](https://i.imgur.com/iywjz8s.png)

# Collaborative Document

2023-04-04 Parallel Python (day 2).

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [https://codimd.carpentries.org/9dSwX5EqSGObKgLTwJf_mA](https://codimd.carpentries.org/9dSwX5EqSGObKgLTwJf_mA)

Collaborative Document day 1: [tinyurl.com/parallel-python-april-2023](https://tinyurl.com/parallel-python-april-2023)

Collaborative Document day 2: [tinyurl.com/parallel-python-april-2023-2](https://tinyurl.com/parallel-python-april-2023-2)

Collaborative Document day 3: [tinyurl.com/parallel-python-april-2023-3](https://tinyurl.com/parallel-python-april-2023-3)

Collaborative Document day 4: [link](<url>) 

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

| Time  | Topic                            |
| ----- | -------------------------------- |
| 09:00 | Welcome,icebreaker and recap     |
| 09:15 | Computing Pi, Numpy, Dask Array  |
| 10:30 | Coffee break                     |
| 10:45 | Numba, Running in Python Threads |
| 12:00 | Tea break                        |
| 12:15 | Python Multiprocessing           |
| 12:45 | Wrap-up                          |
| 13:00 | END                              |

## 🔧 Exercises

### Challenge: implement Pi algorithm 
![](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/fig/calc_pi_3_wide.svg)

Use only standard Python and the function `random.uniform`. The function should have the following interface:

```python
import random
def calc_pi(N):
    """Computes the value of pi using N random samples."""
    ...
    for i in range(N):
        # take a sample
        ...
    return ...

```


#### Solution 

Define the function
```python
import random 

def calc_pi(N):
    """Computes the value of pi using N random samples."""
    
    M = 0 
    for i in range(N):
        # Simulate impact coordinates 
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        # True if impact inside the circle
        if x**2 + y**2 < 1.0: # want to compare with float, not integer
            M += 1
        
    return 4 * M / N  

```


```python
%timeit calc_pi(10**6)
```

### Challenge: use dask library to make the numpy calculation parallel

Write `calc_pi_dask` to make the Numpy version parallel. Compare speed and memory performance with the Numpy version. NB: Remember that [dask.array](https://docs.dask.org/en/stable/array.html) mimics the numpy API.


#### Solution 

```python
import dask.array as da 

def calc_pi_dask(N):
    # Simulate impact coordinates 
    pts = da.random.uniform(-1, 1, (2, N), chunks=N//100) # chunks = chunk size (not number of chunks!)
    # Count number of impacts inside circle
    M = da.count_nonzero((pts**2).sum(axis=0) < 1)
    return 4 * M / N 
```

```python
print(calc_pi_dask(10**6).compute())
%timeit calc_pi_dask(10**6).compute()
```

Questions: 
- how substitutable are numpy and dask?
    - not everything from numpy is implemented in dask, but the most commonly used functionalities are. 
    - have to check for each use case 
- why not `.compute()` within the function `calc_pi_dask()`?
    - in general, dask computations should be delayed as much as possible. 
- why is there no speed-up from dask vs numpy?
    - we are not using `N` large enough; using `10**7` should make the speed-up visible
    - there is also some overhead involved 
- what are the chunks doing? how do we optimize?
    - see the link at the bottom of the collaborative document for best practices with dask 

Other comments
- it is easy to switch from numpy to dask 
- but it is sometimes opaque and thus difficult to understand what it does
- and the speed-up is not even that large
- there is no free lunch -- if you want to get more speed-up, you need to put in the effort


### Challenge: numbify `calc_pi`
Create a Numba version of `calc_pi`. Time it.


#### Solution 

```python
@numba.jit
def calc_pi_numba_native(N):
    M = 0
    for i in range(N):
        # Simulate impact coordinates
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # True if impact happens inside the circle
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N
```

```python
@numba.jit
def calc_pi_numba_numpy(N):
    # Simulate impact coordinates
    pts = np.random.uniform(-1, 1, (2, N))
    # Count number of impacts inside circle
    M = np.count_nonzero((pts**2).sum(axis=0) < 1)
    return 4 * M / N

```


```python
%timeit calc_pi_numba_native(10**6)
```

```python
%timeit calc_pi_numba_numpy(10**6)
```

The plausible reason why numba+numpy is slower than numba+native python is that compiling the numpy part takes additional time.


### Exercise: try threading on a Numpy function

Many Numpy functions unlock the GIL. Try to sort two randomly generated arrays using numpy.sort in parallel.

#### Solution 

```python
high = 10**6 
rnd1 = np.random.random(high)
rnd2 = np.random.random(high)
%timeit -n 10 -r 10 np.sort(rnd1) #  n=10 loops with 10 repetitions each
```

```python
%%timeit -n 10 -r 10
t1 = Thread(target=np.sort, args=(rnd1, ))
t2 = Thread(target=np.sort, args=(rnd2, ))

t1.start()
t2.start()

t1.join()
t2.join()
```

this will take about equally long in both cases, but we are doing twice as much work in the parallel part. Else, we could pass half the inputs to each thread in the parallel part.

### Exercise: reimplement `calc_pi` to use a queue to return the result



#### Solution

We extend the pimp.py file as follows:

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


def calc_pi_helper(n):
    pi = calc_pi(n)
    print(f"pi ~ {pi}")
    queue.put(pi)


n = 10**7

if __name__ == "__main__":
    queue = Queue()
    p1 = Process(target=calc_pi_helper, args=(n, queue))
    p2 = Process(target=calc_pi_helper, args=(n, queue))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    
    results = []
    while not queue.empty():
        results.append(queue.get())
    print(f"Mean pi estimation: {sum(results) / len(results)}")

```

Note: keep you functions pure -- do not touch the calc_pi function; write a wrapper around it to suit it to your need. Specifically, here we use the queue outside of the calc_pi function.

Then run the script in the notebook:
```python
!python pimp.py
```


## 🧠 Collaborative Notes


### Computing pi 

```python
import random 
random.uniform(-1, 1)
```

![](https://carpentries-incubator.github.io/lesson-parallel-python/fig/calc_pi_3_wide.svg)


Now we try with numpy 

```python
import numpy as np 

def calc_pi_numpy(N):
    # Simulate impact coordinates 
    pts = np.random.uniform(-1, 1, (2, N))
    # Count number of impacts inside circle
    M = np.count_nonzero((pts**2).sum(axis=0) < 1)
    return 4 * M / N 
```


```python
print(calc_pi_numpy(10**6))
%timeit calc_pi_numpy(10**6)
```
We should get a speed-up compared to the naive function from before. 

Discussion: is this better?
- memory
- one cannot always vectorize
- number overflow? -- it seems that we will run out of memory more quickly than reaching overflow
- less intuitive; harder to understand by others (or myself in the future)
- for larger number, using memory is big issue. because reading from memory is overhead of using CPU.



### Using Numba to accelerate Python code

```python
import numba 

@numba.jit
def sum_range_numba(a):
    """Compute the sum of the numbers in the range [0, a)."""
    x = 0 
    for i in range(a):
        x += i
    return x 
```

Questions: 
- how does the `numba.jit` decorator work?
    - The decorator takes a function as an input, and changes it to behave differently. Decorators are used across python for different purposes, for instance also for testing with `pytest`. 
        - here, `numba.jit(f)` translates the function `f` into machine code. 
    - we would get the same result with `numba.jit(sum_range_numba(...))`
- why not just decorate all functions with `numba.jit`?
    - python does not check function inputs -- this gives a lot of freedom and makes it easy to learn. but for machine code, the type of function inputs needs to be declared 
    - the function needs to be compiled when called for the first time, and this takes time. And for every new input type (ie, float vs int), the function needs to be compiled at the first run. 

[Documentation on jit](https://numba.pydata.org/numba-doc/latest/user/jit.html)

```python
%timeit sum(range(10**7))
```

```python
%timeit np.arange(10**7).sum()
```

```python
%timeit sum_range_numba(10**7)
```

Coming back to the question from above: recompiling for new input types. Here we try out a new argument type with the same function

```python
%time sum_range_numba(10.**7)
%time sum_range_numba(10.**7)
```

We see that the first run takes longer than the second one.

```python
%time sum_range_numba(10.**7)
%time sum_range_numba(10.**7)
```
But now, the function is compiled and the timings between the two calls are similar. 

Take-home
- it does not make sense to parallelize something that is slow itself
- measuring == knowing: always profile your code to see which parallelization method works best


## Threading and Multiprocessing


```python
import numba
import random 

@numba.jit 
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
%timeit 
calc_pi(10**7)
```

```python
from threading import (Thread)
```

```python
%%time 
n = 10**7
t1 = Thread(target=calc_pi, args=(n//2,))
t2 = Thread(target=calc_pi, args=(n//2,))

t1.start()
t2.start()

t1.join() #wait until thread is done
t2.join()

```

We are not getting faster, despite using two threads. Why? -- Global Interpreter Lock. It makes programming in Python safer, but creates problems for parallel computations.


```python
import time 
```

```python
%%time 
n = 10**7
t1 = Thread(target=time.sleep, args=(1,))
t2 = Thread(target=time.sleep, args=(1,))

t1.start()
t2.start()

t1.join() #wait until thread is done
t2.join()
```


Now we release the GIL with `numba.jit` by pasing the argument `nopython=True`, or using `numba.njit()` directly

```python
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
%%time 
n = 10**7
t1 = Thread(target=calc_pi, args=(n//2,))
t2 = Thread(target=calc_pi, args=(n//2,))

t1.start()
t2.start()

t1.join() #wait until thread is done
t2.join()

```
**(aside: explaining the exercise)**

```python
import numpy as np
rnd = np.random.normal(0, 1, 500_000)

```

```python
rnd
```


```python
np.sort(rnd)
```

**(aside closed)**


```python
%%time 
np.sort(rnd[:len(rnd)//2])
np.sort(rnd[len(rnd)//2:])

```


```python
%%time 

t1 = Thread(target=np.sort, args=(rnd[:len(rnd)//2],))
t2 = Thread(target=np.sort, args=(rnd[len(rnd)//2:],))

t1.start()
t2.start()

t1.join() #wait until thread is done
t2.join()
```

We see that numpy does release the GIL and multithreading is twice as fast as sequential computation.


```python
from multiprocessing import Process

```


```python
n = 10**7

def calc_pi_helper(n):
    pi = calc_pi(n)
    print(f"pi ~ {pi}")

p1 = Process(target=calc_pi_helper, args=(n,))
p2 = Process(target=calc_pi_helper, args=(n,))

p1.start()
p2.start()

p1.join()
p2.join()
```

if it does not work:

put the following code into a script `pimp.py` in the same directory as the notebook you are running.
```python
import random
import numba 
from multiprocessing import Process

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


def calc_pi_helper(n):
    pi = calc_pi(n)
    print(f"pi ~ {pi}")


n = 10**7

if __name__ == "__main__":
    p1 = Process(target=calc_pi_helper, args=(n,))
    p2 = Process(target=calc_pi_helper, args=(n,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
```

then run the script as follows from within the notebook

```python
!time python pimp.py
```

compare to non-parallel execution:
```python
%%time
calc_pi(2*10**8)
```

using a queue 
```python
from multiprocessing import Queue 
Could you share the code using the queue?

```

```python
= Queue()

```

```python
queue.put(5)
```


```python
queue.get()
```

Usually we can use some higher-level interfaces to do parallel computations.


```python
from multiprocessing import Pool
```


```python
help(Pool)
```

have to re-define the function here 
```python
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
Pool().map(calc_pi, [n, n, n, n])
```

Tomorrow:
- delayed evaluation 


## Feedback 

### Positive
- nice exercises as an introduction to the libraries :+1:
- I felt the level was very good today :+1:
- nice pace! Learned new things today :+1:
- We covered many ways to paralleize. 
- Good points on what can or cannot be pararellised and why to do it. :+1:
- Good pace, nice examples and useful links to documentation :+1: 
- Pace was good, I liked the examples.
- Thanks for being patient with the questions and explaining things in detail 




### To be improved 

- Perhaps an overview of the pros/cons of each parallel method? :+3:
- I'm missing in this collab document, a recap of the explanations/reasonings for choices made in the example code that is part of this document (at least those explanations given during the lesson). I can add this to my own notebook as comments, but it would be nice to have this archived here as well.
- I had to step out near the end for 15 minutes and when I got back, the collab doc didn't have Johan's code in there yet. This made it more difficult to get back up to speed. Is it possible to put the code in the collab doc faster?
- Some overview/structured document with what we have covered so far would be helpful, as we shortly interact with many different methods and it can be a bit confusing
- may be an example of how to decide which part of code is suitable to parallel computing. (I know profiling)
- needs to be more explicit about which case to use each parallel method, dask seems strictly worse than numba-unvectorised, so why use that
- an overview of what downsides correspond to setting the arguments in the numba.jit function to true or false for example (so nopython=True/False, and nogil=False/True). It was not entirely clear for me now what would happen/what could go wrong if you would set those to false.
- got lost when multiprocessed functions didn't display in Jupyter and had to be run as an external script
- I lost in all the functions at some point, perhaps an overview of the structure at the beginning is benificial.

- not very important, but maybe the host/organisers can visit now and then into the groups 
- A brief description of how different computer setups might affect the results since it seems very signifiacnt, and then how to tweak these
- Maybe it would be nice to have an overview on what trade-offs you should do in deciding what you want to use for speeding up code: numba,dask,multi-threading,multi-processing,etc., and the settings in each method (nopython,nogil,chunks,etc.)

## 📚 Resources


- [Dask best practices](https://docs.dask.org/en/stable/best-practices.html)
- [Numba documentation](https://numba.readthedocs.io/en/stable/)
    - all the numpy functions that are supported in numba: [documentation](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)

- The fancy btop Johan is using: [btop](https://github.com/aristocratos/btop)
