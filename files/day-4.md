![](https://i.imgur.com/iywjz8s.png)

# Collaborative Document

2023-04-06 Parallel Python (day 4).

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

| Time  | Topic                                  |
| ----- | -------------------------------------- |
| 09:00 | Welcome, icebreaker and recap          |
| 09:15 | Introduction to coroutines and asyncio |
| 10:45 | Coffee break                           |
| 11:00 | Computing fractals in parallel         |
| 12:00 | Tea break                              |
| 12:15 | Presentations of group work            |
| 12:45 | Post-workshop Survey                   |
| 13:00 | END                                    |


## 🔧 Exercises

### Exercise 1
Can you write a generator that generates all even numbers? Try to reuse integers(). Extra: Can you generate the Fibonacci numbers?


#### Solution 


##### Room 1
```python

def fibonacci():
    a = 1
    b = 1
    while True:
        yield a
        
        c = a + b
        a = b
        b = c

```

##### Group 2
```python
def even():
    a = 0
    while True:
        yield a
        a += 2
        
def fib():
    a=0
    b=1
    while True:
        yield a
        a,b = b,a+b
```

##### Group 3
```python
def fibonacci():
    a = 1
    b = 1
    while True:
        yield a 
        a,b = b,a+b


```

##### Group 4
```python
def fibonacci():
    a = 0
    b = 1
    while True:
        temp = b
        b = a
        a += temp
        yield a

```

##### Group 5
```python
def fibonacci():
    two_vals = [0,1]
    while True:
        next_fib = sum(two_vals)
        two_vals = [two_vals[-1], next_fib]
        yield next_fib
        
f = fibonacci()

next(f)
```





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
        - OS dependent internal mechanism for running functions concurrently
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
* Use abstractions to handle complexity
    - Delayed/lazy functions to compose workflows
    - Bags and Data flow patterns
        - map
        - reduce
        - filter


## 🧠 Collaborative Notes

### Asyncio

What are coroutines?
- different kind of abstractions than functions 
- what are functions?

![](https://mermaid.ink/img/pako:eNp1kL1Ow0AQhF9luSYgHApEdUUogpCoKRCSm8U3JpbtXXM_NlGUd-dMYgqkdKvb-Wb25mAqdTDWBHwlSIWnhj8996UQcRWbkSNoy10HPz-dpvVmc_ucJK9VLG01dY72mmjowAHEEiZ46mFp2nGkJlB9_X3zOBssWLZYn8wsvSMUFHd_YNY_3N9dinubLScOv8TgMTaawhm9GPFCTmUVqRWdCpqwGkGCMYeFQVsIfaBWj6uZF81f1nm30BCXk3TpxeFfM6YwPXzPjctFHmZJafJ1PUpj8-jYt6Up5Zh1nKK-7qUyNvqEwqTBZZ9z6cbW3AUcfwB5sYta?type=png)

Functions: black-box that transforms input to output; does not depend on anything else. This is called a pure function



```python
import random 
random.randint(1, 7)
# not a pure function: hidden state in it
```
Hidden states are tricky when you do parallel programming. Better to have a pure function that always gives the same result.


```python
import math 
math.sin(1.2)

```

But suppose you want something that *does* have state. Ie, functions that have their own little memory and that "remembers" what we did before.



```python
def integers():
    a = 1
    while True:
        yield a 
        a += 1

```



```python
i = integers()

```

can query `i` repeatedly and loop over them with `next`
```python
next(i)

```


```python
for i in integers():
    print(i)
    if i > 10: # important, otherwise runs forever
        break

```

or use itertools
```python
from itertools import islice

```


```python
list(islice(integers(), 0, 10))

```

what does `islice` do? -- does the same as slicing, but on iterators. for instance, the code below would not work with an iterator: 

```python
x = list(range(100))
x[4:14]
# alternative: 
slice(4, 14)
```


Illustration of how coroutines work: 

![](https://mermaid.ink/img/pako:eNqtkT1Pw0AMhv-KuYWBdIDxhjAEqepUJAaWLG7OTU-9j-DzlUZV_zsXhQgqMbJZ9vPafu2L6qIhpVWij0yhoxeLPaNvAwB2Yk8oBA06Rzyl5mhV1w-bINQTJw2vjjARJEEW6GIOYkP_Cy70D_x-QAGbQA4Egc4CIfsd8fPEL9SkmLUaHv-r0dNUCLG4iSdiWNIUDAwcF8uG_jC9bm5Hb-49pMg8VjDGDJ_EBPvIfRShALiLWe5u1qjr1brRsD1WRetcOVUcgM42zZdSlfLEHq0pf7hMylYVW55apUtokI-tasO1cJglvo2hU1o4U6XyYMqu3z9Teo8u0fUL2jOgcw?type=png)


one solution for even ints:
```python
def even_integers():
    for i in integers():
        yield i * 2

```

another:
```python
list(islice(filter(lambda i: i % 2 == 0, integers()), 0, 10))

```

There are also coroutines that accept values (the previous ones did not accept any inputs).

```python
def printer():
    while True:
        x = yield
        print(x)

```


```python
p = printer()

```



```python
p.send("hello") # will throw TypeError. 
```
function is not yet ready to receive anything. we need to first call `next()` on it:


```python
p = printer()
next(p)

```

```python
p.send("hello")
p.send("goodbye")
```


Now try to also print the line number.

Expectd output:

```python
p.send("Mercury")
p.send("Venus")
p.send("Earth")

1 Mercury
2 Venus
3 Earth

```

Hint: use f-strings: 

```python
x = 3
print(f"{x}")

```

Solution
```python
def printer():
    line_number = 1
    while True:
        x = yield
        print(f"{line_number} {x}")
        line_number += 1

```


```python
p = printer()
next(p)
p.send("Mercury")
p.send("Venus")
p.send("Earth")

```

We can also write a decorator that makes sure that `next()` is being called (and we do not have initialize the printer with `next()` each time). (will put in collaborative document later)


```python
def printer():
    for line_number in integers(): 
        x = yield
        print(f"{line_number} {x}")

```
We do not need to initialize the `integers()` generator because when we initialize `integers()`, it already `yield`s something, while the `printer()` does not. 


```python
p = printer()
next(p) # alternative: p.send(None)
p.send("Mercury")
p.send("Venus")
p.send("Earth")

```

What is called generators in python is what everyone calls coroutine. Coroutines are their own little universe, and so we can see already that they can run in their own thread and do their work independently of each other (and yield stuff). This is called *collaborative multitasking*. In contrast, threading is not collaborative because the operating system determines when the threads start working.
`asyncio` implements the coordination for us---despite its name: it is *not* asynchronous, but highly synchronous. 

Origin: async comes from Javascript. 
In concurrent programing, there is no place to start the program (as for instance in python when we see the `if __name__ == "__main__"`) statement. This is for instance the case for webservices that serve various requests from different sources that are independent from each other. In other words, async is a lightweight threading system.




```python
import asyncio 

```

```python
async def counter(name): # in python, the keyword `async`  defines a coroutine
    for i in range(5):
        print(f"{name:<10} {i:03}")
        await asyncio.sleep(0.2) # here the coroutine gives away control to the puppet master. it says: "I am not interesting in doing anything right now, but give me back control in 0.2 seconds."
```


```python
await counter("Venus")

```

executes consecutively:
```python
await counter("Earth")
await counter("Moon")

```

to make it execute concurrently
```python
await asyncio.gather(
    counter("Earth"),
    counter("Moon") # no await for the single processes. we await the `gather` only
)
```
under the hood: the puppetmaster says
- first, "Earth you go"
- then, "Mooon, you go"
- then, "Earth, you go"
etc 


Exercise: let the Earth and Moon processes wait for different amounts of time. 

```python
async def counter(name, delay=0.2): 
    for i in range(5):
        print(f"{name:<10} {i:03}")
        await asyncio.sleep(delay)

```


```python
await asyncio.gather(
    counter("Earth", 0.2),
    counter("Moon", 0.1)
)

```
because Earth waits for longer, it cycles alone after some time. 

Nothing is happening in parallel yet. Instead, the processes are run interchangeably (Earth, Moon, Earth, etc).



**Using ayncio in a script (not in a jupyter notebook)**
We need to define a main function. To await something, we need to be inside an asynchronous coroutine, which is started by the `async.run(main)` call below:

```python
# some main file

import asyncio 

... 

async def main():
    ...
    
    
if __name__ == "__main__":
    asyncio.run(main)

```

**timing**
line magics do not work for asyncio. 

this will not work
```python
%%time 

await counter("Earth")

```

So, we need other ways to time our code. The following snippet does this:

```python
from dataclasses import dataclass
from typing import Optional
from time import perf_counter
from contextlib import asynccontextmanager


@dataclass
class Elapsed:
    time: Optional[float] = None


@asynccontextmanager
async def timer():
    e = Elapsed()
    t = perf_counter()
    yield e
    e.time = perf_counter() - t
```


```python
async with timer() as t: # asynchronous context manager
    await asyncio.sleep(0.2)
print(f"that took {t.time} seconds.")
```


compute pi again -- this is just copied from what we did the other days

```python
import random
import numba


@numba.njit(nogil=True)
def calc_pi(N):
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
async with timer() as t:
    result = await asyncio.to_thread(calc_pi, 10**7) 

```

```python
t.time

```

```python
result

```


#### Exercise: gather multiple outcomes
We’ve seen that we can gather multiple coroutines using asyncio.gather. Now gather several calc_pi computations, and time them.



```python
async with timer() as t:
    result = await asyncio.gather(
        # collection of awaitables
        asyncio.to_thread(calc_pi, 10**7),
        asyncio.to_thread(calc_pi, 10**7)) # if we wait this, the result will be a list 

```


```python
result

```


```python
t.time

```
we should get a very similar number, even though we are doing twice as much computations.



```python
async def calc_pi_split(N, M):
    list_of_awaitables = [asyncio.to_thread(calc_pi, N) for _ in range(M)]
    awaitable_of_list = asyncio.gather(*list_of_awaitables)
    result = await awaitable_of_list
    return result 
```



```python
async with timer() as t:
    pi_estimate = await calc_pi_split(10**7, 10)
    print(f"Value of pi: {pi_estimate}")
    
print(f"that took {t.time} seconds")
```


how much speed up do we get? compare the above timing with doing everything in a single thread:
```python
async with timer() as t:
    result = await asyncio.to_thread(calc_pi, 10**8)
    
print(f"that took {t.time} seconds")

```
In my case about 3-times speedup.
For me ~4.5 times


## Final exercise: computing fractals

$\sqrt{-1} = i$


Complex numbers in python are denoted with `j`:

```python
1j 

```


```python
type(1j)

```



```python
1j*1j

```
- real part is -1, imaginary part is 0


```python
(1+1j)*1j

```

Mandelbrot set: set of numbers for which the following iteration diverges:


```markdown
$$z_{n+1} = z_n^2 + c$$

$$c \in \mathbb{C}$$

$$|z_n| > 2 $$ 

```

[wikipedia](https://en.wikipedia.org/wiki/Mandelbrot_set)


Parallelize the following piece of code with any/multiple methods that we discussed in this course.


```python
import numpy as np 


width = 512
height = 512
max_iter = 1024 # when do we stop looking for divergence?
center = -1.1195+0.2718j
extent = 0.005+0.005j
scale = max((extent / width).real, (extent / height).imag)

result = np.zeros((height, width), int)
for j in range(height):
    for i in range(width):
        ## calculate one pixel in the image
            # circle around center by going half of width to left and right, and half of height up and down
        c = center + ((i - width // 2) + (j - height // 2)*1j) * scale
        z = 0
        for k in range(max_iter):
            z  = z**2 + c
            if (z * z.conjugate()).real > 4.0:
                break
        result[j, i] = k

```



```python
from matplotlib import pyplot as plt 

fig, ax = plt.subplots(1, 1, figsize = (5, 5))
plot_extent = (width + 1j*height) * scale
z1 = center - plot_extent / 2
z2 = z1 + plot_extent

ax.imshow(
    result**(1/3), 
    origin="lower", 
    extent=(z1.real, z2.real, z1.imag, z2.imag)
    )
ax.set_xlabel("$\Re(c)$")
ax.set_ylabel("$\Im(c)$")
```

The code above checks combination of `(i, j)` sequentially. How can we parallelize it and make it faster?


### Solutions


#### Group 1



```python
import numba
import numpy as np

@numba.njit(nogil=True)
def get_k(p,center,scale,max_iter):
    c = center + (p[0]-width //2 + (p[1]-height//2)*1j)*scale
    z = 0
    for k in range(max_iter):
        z = z**2 + c

        if (z*z.conjugate()).real > 4.0:
            break
    return k

@delayed
def get_fractal_v(width,height, extent, max_iter=512):
    center = 0 +0.0j
    scale = max((extent/width).real, (extent/height).imag)
    
    # we can work in parallel :)
    # sounds good
    base_result=da.moveaxis(da.array(da.meshgrid(da.arange(0,512),da.arange(0,512),indexing='ij')),0,-1)

    result = da.apply_along_axis(func1d=get_k,arr=base_result, axis=2, center=center,scale=scale,max_iter=max_iter)
    return result

```


#### Group 2



```python
import matplotlib.pyplot as plt
import numpy as np
import numba

@numba.jit(nogil=True)
def mandelbrot(height, width, max_iter=1024):
    center = 0+0j
    extent = 3+3j
    scale = max((extent / width).real, (extent / height).imag)

    #result = np.zeros((height, width), dtype=int)
    result = [[0 for col in range(height)] for row in range(width)]
    for j in range(height):
        for i in range(width):
            ## calculate one pixel in the image
                # circle around center by going half of width to left and right, and half of height up and down
            c = center + ((i - width // 2) + (j - height // 2)*1j) * scale
            z = 0
            for k in range(max_iter):
                z  = z**2 + c
                if (z * z.conjugate()).real > 4.0:
                    break
            result[j][i] = k
    return result
width = 512
height = 512

results_set = mandelbrot(height, width)
```


#### Group 3



```python
@numba.njit()
def fractal_pixel(i,j, width = width, height = height, max_iter = max_iter,
            center = center, extent = extent, scale = scale):
    c = center + (i - width // 2 + (j - height //2)*1j) * scale
    z = 0
    for k in range(max_iter):
        z = z**2 + c
        if (z * z.conjugate()).real > 4.0:
            break
    return k

```


#### Group 4

```python
h = np.arange(512)-256
w = np.arange(512)-256
mg_h,mg_w  = np.meshgrid(h, w)

@numba.jit(nopython=True, nogil=True)
def pixel(width, height, center, scale):
    c = center + (mg_w + mg_h*1j) * scale
    return c
c = pixel(w, h, center, scale)
```



Further options
- [`numba.vectorize`](https://numba.pydata.org/numba-doc/latest/user/vectorize.html#the-vectorize-decorator): can pass entire numpy arrays
    - this could be parallelized with dask array, for instance
- use numba njit, and wrap around a function that calls the numba function
    - can be parallelized in different ways
        - split the picture into subdomains
        - numba with `parallel=True` and replacing `range` with `prange`

The solution that Johan presented:

```python


```



## Feedback

### General 
- We need a sequel to this course "Advanced parallel computing" to dive a bit deeper :+1::+1::+1: 
- The prerequisites on the website are a bit difficult to interpret (what is fammiliar/comfortable, does 'understand how numpy works' mean you have a working knowledge or have actually read the code etc), perhaps some example codes or something to 'test your level' could be useful to help people decide if the course is for them

### Positive
- Well coordinated course, knowledge and patient hosts :+1: 
- Very nice and patient teachers, many different approaches are shown and there is a lot of honesty (for example: python is not always the best approach etc). You really feel like you can ask any and all questions. Also the colab doc is super useful :+1: 
- Awsome course, great session as always. I've enjoyed the pace and the material. There was sufficient depth and a host of varying techniques for real world challenges. :+1: 
- This course gave a nice overview of some methods that can be used when parallelizing your code, and which methods will be worth to look into further, as (unfortunately) the course did not always go deep into the underlying 'processes' that happen when your computer executes python code for example.. 
- 

### To be improved
- breakout rooms could be guided with host/co-hosts presence
-- (another participant): I agree with this but felt it went much better the last day with people hopping around
- Perhaps extra 'study' material to see applications of co-routines, asynchroneous programming etc. of varying examples. It's a lot of techniques and due to time constraints can be only briefly touched upon but I'd like to solidify the knowledge by extra hands-on exercises to get a better understanding when to use a solution type
- Some more time for individual excercises would have been great (for example in the Mandelbrot exercise). Now it felt a bit like the 'normal' exercises were very simple and clearly outlined compared to the final exercise which was a lot more challenging and felt a bit rushed. Maybe a 'daily challenge'? (could even be an optional homework exercise that can be discussed the next day) :+1::+1: 
- I would have wanted to have some more time on the final exercise; the Mandelbrot set, because then you get to 'finally' apply everything yourself (I mean figure out which method is best, rather than in the breakout rooms where you have to practice what you just saw/learnt). 


## 📚 Resources

- ways to put limitations on async processes: 
[Synchronization Primitives](https://docs.python.org/3/library/asyncio-sync.html). Ie, `Lock`, `Semaphore`, `Barrier`
- other libraries related to this course: 
    - higher-level abstractions for multiprocessing/multithreading: 
        - [joblib](https://joblib.readthedocs.io/en/latest/)
        - [ray](https://www.ray.io/) 
        - each case is different, so have to try out
    - [jax](https://jax.readthedocs.io/en/latest/): alternative to numba, and also heavily used in machine learning
    - [xarray](https://docs.xarray.dev/en/stable/) works well together with dask 
- domain-specific software often uses the things we discussed in this course (and the above libraries) under the hood. for instance, in geoscience:
    - [iris](https://scitools-iris.readthedocs.io/en/latest/)
    - [esmvaltool](https://esmvaltool.org/)
