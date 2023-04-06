![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document

2023-04-03 Parallel Python

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/parallel-python-april-2023-4)

Collaborative Document day 1: [link](https://tinyurl.com/parallel-python-april-2023)

Collaborative Document day 2: [link](https://tinyurl.com/parallel-python-april-2023-2)

Collaborative Document day 3: [link](https://tinyurl.com/parallel-python-april-2023-3)

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

[link](https://esciencecenter-digital-skills.github.io/parallel-python-workshop/)

## 👩‍🏫👩‍💻🎓 Instructors

Johan Hidding, Jaro Camphuijsen

## 🧑‍🙋 Helpers

Flavio Hafner, Laura Ootes  

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city



## 🗓️ Agenda

### Day 1

| Time  | Topic                  |
| ----- | ---------------------- |
| 09:00 | Welcome and icebreaker |
| 09:15 | Introduction           |
| 10:30 | Break                  |
| 10:45 | Measuring performance  |
| 12:00 | Coffee break           |
| 12:15 | Recap and Computing Pi |
| 12:45 | Wrap-up                |
| 13:00 | END                    |

## 💡 Questions
l free to post any questions here 



## 🔧 Exercises
### Exercise 1
**Why did you sign up for this course? What problem are you trying to solve?**

- I want to compute independent calculations faster (mainly linear algebra for radio array calibration), where many forward calculations are used to circumvent having to do large matrix inversions

- I will perform computations with lots of spatial data time series (not sure yet which methods)
- I want to understand the high level libraries available in Python, which are different from my background of CFD, where mainly MPI reigns supreme. 
- Fast database searching - gene sequences
- I want to refresh my knowledge on parallel computing in Python. I am self-learned in parallel programming so I'm curious where I can improve. I already work with large datasets (>80 TB)and use parallelzation often to support scaling up research. 
- I work with satellite data and I need to calculate the ground movement for lots of points  
- make better use of computational resources I have, learn to understand parallel coding better (I learned how to parallelise on GPU but never learned how to parallelise on CPU, and don't understand the differences). I develop AutoML algorithms for satellite imagery, so lots of data, but not all algorithms are parallelisable on GPU.
- I work with images and I want to speed-up pre-processing and processing of large batches of data (I know how to apply some parallel programming already, but I want to know new tools/options)
- I do reliability evaluation for power system, It uses Monte Carlo sampling which can be done in parallel because they are independent.
- Prajwal: I had to some computation across grid cells of 0.05 degrees where I realised I need to make it faster. 
- Deborah: Spatial application of crop model
- Running MRI processing software on Snellius - e.g. Freesurfer, FSL, MagetBrain
- We use Fortran a lot because it is much faster to deal with large data, but Python sometimes is quite slower than Fortran, so I would like to find a way to optimize Python.
- I have to process national-wide airborne LiDAR 3D point cloud data (really huge data) for ecological applications. We use Spider on SurfSara in Python, I want to speed up the current workflow for data processing as well.
- Apply computations to satellite data (large data sets) in parallel
- Learn about parallel programming
- Data processing of deep learning,  I also work on satallite data, I can access to a remote server with GPU, and want to use some techniques to speed up the training and make full use of the resources.
- Ashim - computation with a lot of 3D lidar wind measurement Data, combining data from individual sensors and speeding up this code (reduced from 300s to 86s per file, but I guess I can improve) 
- I need to simulate PDFs of estimators (numerically) and compute very low probabilities (integrity risks) from these distributions. So I need to do a lot of simulations to get an accurate estimate for this integrity risk
- Lattice Boltzmann and DEM coupled simulations for porous media 
- I will simulate biological data similar to microscopy images and process them to increase the reolution and the quality. Which meand I will apply similar process for each frames in a video. 

### Exercise 2
***Can you think of a task in your domain that is parallelizable? Can you also think of one that is fundamentally non-parallelizable?***

- naturally: Radar volume measurement processing;  
- naturally: apply a pre-processing function (e.g., a median filter) to a batch of images (the result for each image is independent from the rest); difficult to parallelize: pairwise alignment of 2D slices in a 3D volume (each 2D slice position depends on the position of all the previous slices)

- History dependent iterative solvers are hard to parallelize

- naturally: epoch-by-epoch position estimations
- hard: Kalman filter, 

- Natural: Monte Carlo. Difficult: a time marching problem (e.g., a diffusion/convection problem)

- Natural: Forward predictions of visibilities based on point source/Gaussian model and known gains. Non-parallelizable: Hogbom cleaning for deconvolution (iterative)
- naturally: splitting batch of images for DL over 2 GPUs. Hard: bayesian optimisation ? (algorithms that model a probabilistic distribution can be hard to parallelise I believe?)

- Computing differences between two images: natural, k means clustering: difficult?

- Natural: Deseasonalising and detrending time-series across grids vs Hard: Computing spatial autocorrelation. 

- Natural: Doing the same computation for a lot of pixel in a satellite image. Hard: spatial interpolation with kriging


- peak finding in spectral data (maybe natural)



### Exercise 3
We have the following recipe:

1. (1 min) Pour water into a soup pan, add the split peas and bay leaf and bring it to boil.
2. (60 min) Remove any foam using a skimmer and let it simmer under a lid for about 60 minutes.
3. (15 min) Clean and chop the leek, celeriac, onion, carrot and potato.
4. (20 min) Remove the bay leaf, add the vegetables and simmer for 20 more minutes. Stir the soup occasionally.
5. (1 day) Leave the soup for one day. Reheat before serving and add a sliced smoked sausage (vegetarian options are also welcome). Season with pepper and salt.

Imagine you’re cooking alone.

- Can you identify potential for parallelisation in this recipe?
- And what if you are cooking with the help of a friend help? Is the soup done any faster?
- Draw a dependency diagram. Using [mermaid live](https://mermaid.live/edit#pako:eNpVjstqw0AMRX9FaNVC_ANeFBq7zSbQQrPzZCFsOTMk80CWCcH2v3ccb1qtxD3nCk3Yxo6xxP4W760lUTjVJkCe96ay4gb1NJyhKN7mAyv4GPgxw_7lEGGwMSUXLq-bv18lqKbjqjGodeG6bKh69r8Cz1A3R0oa0_kvOd3jDB-N-7b5_H9ihXPrs-mp7KloSaAieSq4Q8_iyXX5_WlNDKplzwbLvHYkV4MmLNmjUePPI7RYqoy8wzF1pFw7ugj5LVx-AfLqVWg)

#### Group 1
[![](https://mermaid.ink/img/pako:eNqNlE1v2zAMhv8KoZMHtMPWfRx8GNAmaXvYIWiGnXJhLDoRIouGPhoEbf_7KDmpiy7DkkMihM9LvhRpP6mGNalatZZ3zQZ9hF_TpQP5XFezro97mKP7AJeXP54_1zDn5GGHkTwYFxkQAqceenSATsPKG7cGE0FCKzb24zPcVHPCkuOQ9ZjqWmuIG4LQWxH0hCHT_4BWuAdL2I7ITUGuavhJMVcMpuvEVXJavhGs0dCynFacInz_BJ1xKVKucVctCputjr5e8z1Qx48k7ewlAXaQQgal0W1R5QSD4q4oDvg7k9PqN63Xhv4q8KWGiTDDdTUb7ovSEm1FNakm8k9PGgZ1OEPakJVWsMny_9PsDLvz0Aa953ge23PEyCM7KexxeIfupqdib-yfjB8NnxYfLZ6Mvpo6RKcl-rWGYfxlPa5kMdjTm-2YVYu80ePcRlk0ftjYDHDTYBBvaO0-ywZ4VuBveSvxsBWFzrXYEWjM8G01YbYy5rHK7VH4QBvCCCtqs69A_lHWTzT31ULOpzW5Y1lQaxqJh463-QdTwDVl5Tt6IY8aO9iZuJHHTtbNl2EGtPkq79WFktvp0Gh5Kzxl7VJJHx0tVS1HTS0mG5dq6V4ExRR5sXeNqqNPdKFSr-XtMDW49tipukUb6OUPfPpeEg?type=png)](https://mermaid.live/edit#pako:eNqNlE1v2zAMhv8KoZMHtMPWfRx8GNAmaXvYIWiGnXJhLDoRIouGPhoEbf_7KDmpiy7DkkMihM9LvhRpP6mGNalatZZ3zQZ9hF_TpQP5XFezro97mKP7AJeXP54_1zDn5GGHkTwYFxkQAqceenSATsPKG7cGE0FCKzb24zPcVHPCkuOQ9ZjqWmuIG4LQWxH0hCHT_4BWuAdL2I7ITUGuavhJMVcMpuvEVXJavhGs0dCynFacInz_BJ1xKVKucVctCputjr5e8z1Qx48k7ewlAXaQQgal0W1R5QSD4q4oDvg7k9PqN63Xhv4q8KWGiTDDdTUb7ovSEm1FNakm8k9PGgZ1OEPakJVWsMny_9PsDLvz0Aa953ge23PEyCM7KexxeIfupqdib-yfjB8NnxYfLZ6Mvpo6RKcl-rWGYfxlPa5kMdjTm-2YVYu80ePcRlk0ftjYDHDTYBBvaO0-ywZ4VuBveSvxsBWFzrXYEWjM8G01YbYy5rHK7VH4QBvCCCtqs69A_lHWTzT31ULOpzW5Y1lQaxqJh463-QdTwDVl5Tt6IY8aO9iZuJHHTtbNl2EGtPkq79WFktvp0Gh5Kzxl7VJJHx0tVS1HTS0mG5dq6V4ExRR5sXeNqqNPdKFSr-XtMDW49tipukUb6OUPfPpeEg)


#### Group 2
[![](https://mermaid.ink/img/pako:eNqF0E1LxDAQBuC_Msy5W1DXSw6K23pcWNSb9TA00zZsPkqS7iKl_920WdCbOYXwzJvhnbF1klFgp921HchH-KgbC-m8fEKIPMKdgJObPFwpsi_hRJ60Zg0DU1S2f_66cdjtnuAwbzP3At7YuAtD58gsf0WVxYOASjNZICuhHdz4mywgjFpFiAPDhftQpvmcUG0JdU7YCyApV1EWEJQx7G8fHf5nGdYbfM3wUUBaKK0c3DQuWGCShpRM5cwrbzAtZLhBka6S_LnBxq6Opujev22LIvqJC5xGmaqqFfWeDIqOdEivLFV0_pjb3kpffgCRYXmY?type=png)](https://mermaid.live/edit#pako:eNqF0E1LxDAQBuC_Msy5W1DXSw6K23pcWNSb9TA00zZsPkqS7iKl_920WdCbOYXwzJvhnbF1klFgp921HchH-KgbC-m8fEKIPMKdgJObPFwpsi_hRJ60Zg0DU1S2f_66cdjtnuAwbzP3At7YuAtD58gsf0WVxYOASjNZICuhHdz4mywgjFpFiAPDhftQpvmcUG0JdU7YCyApV1EWEJQx7G8fHf5nGdYbfM3wUUBaKK0c3DQuWGCShpRM5cwrbzAtZLhBka6S_LnBxq6Opujev22LIvqJC5xGmaqqFfWeDIqOdEivLFV0_pjb3kpffgCRYXmY)


#### Group 3
![](https://mermaid.live/edit#pako:eNplkU2OAiEQRq9SYaWJXqAXk_g364nOznZRQtkSaYpAtcbY3n2gNcY4bICX9xUF3JRmQ6pSB8cXfcQo8LusPeQx2_4wR7igUATrhSExBwjodzCdfsFiNGfrYPyw59tNcFYgECZAb2CPV3CEh3f56S4GtHpslqNFJ2J9My60d0SnHlbbmTFwpoYE947S7ukOiiZH0aLuXyUGzN6y_2AaY2T5gIEFhV9wNXTTlwMDhZAvW9pP6HLu-03p13QkLHC0oXim8b84QnJWUw63fCoTdgkbKon0SqiJaim2aE1-9VshtZIjtVSrKi8NxlOtan_PHnbCm6vXqpLY0UR1weTPWFpsIraqOqBLdP8DbayMFg)

#### Group 4
[![](https://mermaid.ink/img/pako:eNqNlE2L2zAQhv_KYFhoYQNtDz3kUNg4m-xH0jV4YQ9yDhN7HIvYkpHlhBDnv3dsxSGmJV0bjHnfRxppNKOjF-uEvLGX5nofZ2gsvE8jBfw8i2e1MZRIUrZawWj0Cx7EB1oyKzgTnTgRAWG1upZ8McED5IQ7GhpTsSDcDsc_Cp9yMhLjoT4TgbZoNU8x0OfiTUmthuKTiNEYbQfRQhFiXeGGBmoQiJLKkrcxYDG3ov2w6vSHTm8CXRvYt9tu4EVUui6hxEv0yWcg_ybkmBfHrLXMG3gVvtZb-PkNCtkD0w5YiJjTqgBVAnGmyz7E4013dtOd33Sf_u0689Wt2lChdwRrd-hpA0sRyqIgAz-ut7BwNCYJ7GhDFtc5VQyfIy2dXVnJCfotuvIBmxF0qUq1Aa0IEjycp-uR71da2C8oI7QNzKQSYTvaECY9EwSfgNpS-A_Wvy1-d8fN4WdGVrZA1y3NnCwUvORDA5Mvcw0V562UavP1MsRVj39ctGS7V6m2p4vblU3zpqhp-wZLe0n7xXzf64YbSAYZx_nLzAzx2JlIcZziiBsEfHRV7917fDoFyoRb_9gqkceZLijyxvybUIp1biMvUidGsbY6PKjYG1tT071XlwlX8VTixmDh8dx5xSrfFFabpbtOulvl9Aft21Vu?type=png)](https://mermaid.live/edit#pako:eNqNlE2L2zAQhv_KYFhoYQNtDz3kUNg4m-xH0jV4YQ9yDhN7HIvYkpHlhBDnv3dsxSGmJV0bjHnfRxppNKOjF-uEvLGX5nofZ2gsvE8jBfw8i2e1MZRIUrZawWj0Cx7EB1oyKzgTnTgRAWG1upZ8McED5IQ7GhpTsSDcDsc_Cp9yMhLjoT4TgbZoNU8x0OfiTUmthuKTiNEYbQfRQhFiXeGGBmoQiJLKkrcxYDG3ov2w6vSHTm8CXRvYt9tu4EVUui6hxEv0yWcg_ybkmBfHrLXMG3gVvtZb-PkNCtkD0w5YiJjTqgBVAnGmyz7E4013dtOd33Sf_u0689Wt2lChdwRrd-hpA0sRyqIgAz-ut7BwNCYJ7GhDFtc5VQyfIy2dXVnJCfotuvIBmxF0qUq1Aa0IEjycp-uR71da2C8oI7QNzKQSYTvaECY9EwSfgNpS-A_Wvy1-d8fN4WdGVrZA1y3NnCwUvORDA5Mvcw0V562UavP1MsRVj39ctGS7V6m2p4vblU3zpqhp-wZLe0n7xXzf64YbSAYZx_nLzAzx2JlIcZziiBsEfHRV7917fDoFyoRb_9gqkceZLijyxvybUIp1biMvUidGsbY6PKjYG1tT071XlwlX8VTixmDh8dx5xSrfFFabpbtOulvl9Aft21Vu)

#### Group 5
![](https://mermaid.ink/img/pako:eNptzjsOgzAQBNCrWFtDkV9DEQkwXIB0mMLCS0DCGJm1oghx9xhMma1Gb6bYFVqjEBLoRvNpe2mJvbiYmL-0XoybGxbHz6xeCGd2aUKT7cbyutrxemJ6IA94O5EfWAS8n5j_w-LAMuDDI0Sg0Wo5KP_duo8EUI8aBSQ-KuykG0mAmDY_lY5M9Z1aSMg6jMDNShLyQb6t1AG3HzZ2RkY?type=png)]

#### Group 6
- Parallizable tasks are: chopping, removing foam, remove leaf/add veggies
- If the first person can work efficiently, the friend will not help cook the soup faster. In essence, everything can be done while waiting for the soup to cook (60 minutes)
- 


[![](https://mermaid.ink/img/pako:eNpNkk1v2zAMhv8KodMGuMFadDvkMGBNstMOQzNgB6cHWqZjLbJo6COBUfe_j7Kdpr7YJsWXD1_qVWmuSa1VY_miW_QR_mwPDuT5Uf7m5OGCkTwYFxkQAqceenQFYF1DbAlCb02EnjAAuhoqHMASNvOPN-4Ikpbaio19gbu77-NflEjDHu6hMy5FGuHp0zN1fCapGiSFHaSQS6XhyXSd9M9ylmIWC3MkuTrHwZp6UsOKU4RvXxbRsPq8jJGbwqbcCJabdHTL_cRuiU4FaLLkDeoC2BmW0TR6z3E62nPEyKuXWWozSY33X689xm25gGe56-w3c850pIiVpdmchTzTPggne3qHhX00fjY0W8xaYxAYtHa4dn-au98mHJdFbefEwy0Bu_IX4YI1Ceae7AhqHFbwTC1hhIqajBDInyezhTCDi-nWaBLcjk_5hSngkQD2smR2cDGxlYX3_bKWgDZeGXeLQ4-PH2B-ljv3j4d3GDmrCiVOdGhquXqvufagJN3RQa3ls6YGk40HdXBvchRT5P3gtFpHn6hQqa_lUm4NHj12at2gDfT2H9xu8LA?type=png)](https://mermaid.live/edit#pako:eNpNkk1v2zAMhv8KodMGuMFadDvkMGBNstMOQzNgB6cHWqZjLbJo6COBUfe_j7Kdpr7YJsWXD1_qVWmuSa1VY_miW_QR_mwPDuT5Uf7m5OGCkTwYFxkQAqceenQFYF1DbAlCb02EnjAAuhoqHMASNvOPN-4Ikpbaio19gbu77-NflEjDHu6hMy5FGuHp0zN1fCapGiSFHaSQS6XhyXSd9M9ylmIWC3MkuTrHwZp6UsOKU4RvXxbRsPq8jJGbwqbcCJabdHTL_cRuiU4FaLLkDeoC2BmW0TR6z3E62nPEyKuXWWozSY33X689xm25gGe56-w3c850pIiVpdmchTzTPggne3qHhX00fjY0W8xaYxAYtHa4dn-au98mHJdFbefEwy0Bu_IX4YI1Ceae7AhqHFbwTC1hhIqajBDInyezhTCDi-nWaBLcjk_5hSngkQD2smR2cDGxlYX3_bKWgDZeGXeLQ4-PH2B-ljv3j4d3GDmrCiVOdGhquXqvufagJN3RQa3ls6YGk40HdXBvchRT5P3gtFpHn6hQqa_lUm4NHj12at2gDfT2H9xu8LA)

### Exercise 4
Why is the Dask solution more memory efficient?

The aswer is chunking. Dask chunks the large array, such that the data is never entirely in memory.


## 🧠 Collaborative Notes
```python
# check your python version
import sys
sys.version 
```
Python 3.10, 3.9 or 3.8 are good to use. Python 3.11 is most likely not ready yet to be used for this course.


```python
import dask
import multiprocessing
```

```python
# check how many cores you have on your computer
multiprocessing.cpu_count()
```
![Serial computation](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/fig/serial.png)

```python
i = 0
for x in list(range(5)):
    i = i + x
```

```python
work = list(range(5))
```

```python
work_bob = work[:2]
work_alice = work[2:]
```

```python
bob_result= = sum(work_bob)
alice_result = sum(work_alice)
```

```python
bob_result + alice_result
```


```python
# list comprehensions
y = [n**2 for n in work]
y
```


#### Benchmarking
```python
# Summation making use of numpy
import numpy as np
result = np.arange(10**8).sum()
```

```python
# The same summation, but using dask to parallelise the code
import dask.array as da
work = da.arange(10**8, dtype=np.uint64).sum()
result = work.compute()
```
- We use `np.unit64` to avoid integer overflow which would otherwise occur on some systems (if it happens, the result of the sum could be negative). See this [link](https://stackoverflow.com/questions/71028879/what-determines-numpy-default-integer-precision) for more information.

```python
# to check the timing if running once
%%time
np.arange(3*10**8).sum()
```

```python
# to get an avarage duration (running the task multiple times to get an avarage result _ standard deviation):
%%timeit
np.arange(3*10**8).sum()
```

The line magics `%%timeit` and `%%time` only work from jupyter notebooks. To time functions from the python interpreter or from the command line, we can use the timeit module. The documentation is [here](https://docs.python.org/3/library/timeit.html).

```python
# to store the result in the variable time
time = %timeit -o np.arange(10**7).sum()
print(f"Time taken: {time.average:.4f}s)
```

Memory profiling
Install the memory profiler package using pip. You may have to restart the kernel to use the package after installation
```python
# install the memory profiler package
pip install memory_profiler
```

```python
import numpy as np
import dask.array as da
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
```

```python
# no return statement is needed in these functions, because we are not interested in the result of the function right now, we are only going to time it
def sum_with_numpy():
    # Serial implementation
    np.arange(10**7).sum()
    
def sum_with_dask():
    # Parallel implementation
    work = da.arange(10**7).sum()
    work.compute()
    
memory_numpy = memory_usage(sum_with_numpy, interval=0.01)
memory_dask = memory_usage(sum_with_dask, interval=0.01)

# Plot results
plt.plot(memory_numpy, label='numpy')
plt.plot(memory_dask, label='dask')
plt.xlabel('Time step')
plt.ylabel('Memory / MB')
plt.legend()
plt.show()
```
There is no return statement needed in these functions, because we are not interested in the result of the function right now, we are only going to time it.


```python
# check how many cores your system has
import psutil
N_physical_cores = psutil.cpu_count(logical=False)
N_logical_cores = psutil.cpu_count(logical=True)
print(f"The number of physical/logical cores is {N_physical_cores} / {N_logical_cores}")
```

```python
# you can run the range up to the number of logical cores you have on your system
x = []
for n in range(1, 9):
    time_taken = %timeit -r 1 -o da.arange(10**8).sum().compute(num_workers=n)
    x.append(time_taken.average) 
```

```python
import pandas as pd
# make sure the range here is equal the the range you used in the previous cell
data = pd.DataFrame({"n": range(1, 9), "t": x})
data.set_index("n").plot()
```

```python
data
```

### Tomorrow
Tomorrow we will look at computing $\pi$
![](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/fig/calc_pi_3_wide.svg)


$$ A_{circle} / A_{square} = {{\pi r^2}  \over {4 r^2} } = {{\pi \over {4}}} $$




![Parallel computation](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/fig/parallel.png)


## Feedback

### :+1:
Intuitive examples, very nice first day to get people on the same page
Nice examples, ok pace, explanations clear
Pace is well-thought. Also, explained very well. Keep it up. 
I liked the carpentry style (aka trying things on your own machine during the explanations, you can really play around like this)
I also like the style of interaction/coding the examples during the explanation - makes it clearer, and brings up more questions
- Explaination is clear
Good explanation and very knowledgeable instructors. 
Informative!
Good examples - mermaid flowcharting definitely useful, but bit of a learning curve
Initial soup analagy is an excellent start
Making a point that Python is not ideal for parallel programming is a useful highlight
Nice examples, and I really like that the screen is shared in half screen when possible, works much better than screen sharing full screen
* I like the pace, with lots of time to get feedback
Nice examples, interesting tools 
I very much like this interactive tool that we use, and can type in all at the same time. Plus very nice that there are multiple 'TAs' in the call to tell the presenter that there are questions (and who can also help decoding) etc.

### To improve
Although examples are great pace is a bit slow sometimes; keeping focus is difficult
Perhaps 
Pace is definitely a bit slow for me :+6:
Pace is sometimes a bit slow, which can make it hard to stay focused
- Maybe keep the screen large all the time
Could go a bit faster
Some of the parallelization exercises seem to give strange (unintended) results on different systems

Perhaps you can also explain more detailed what happens inside the code? For example, when you use dask to generate an array, you have to use the 'compute' command afterwards. Why is that specifically? Probably because this dask module makes an object of some kind. Perhaps you could explain more about that. 
Perhaps also provide the documentations of the modules that we use, such that we can see what types of arguments we can give to some of the functions we use. 

could you include some tool like line_profiler as well

## 📚 Resources
[Calls for proposals](https://www.esciencecenter.nl/calls-for-proposals/) at the eScience center.
[Mermaid live](https://mermaid.live/edit#pako:eNpVjstqw0AMRX9FaNVC_ANeFBq7zSbQQrPzZCFsOTMk80CWCcH2v3ccb1qtxD3nCk3Yxo6xxP4W760lUTjVJkCe96ay4gb1NJyhKN7mAyv4GPgxw_7lEGGwMSUXLq-bv18lqKbjqjGodeG6bKh69r8Cz1A3R0oa0_kvOd3jDB-N-7b5_H9ihXPrs-mp7KloSaAieSq4Q8_iyXX5_WlNDKplzwbLvHYkV4MmLNmjUePPI7RYqoy8wzF1pFw7ugj5LVx-AfLqVWg) for creating dependecy diagrams.
