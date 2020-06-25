# Table of Contents

1. [Online Learning](#online_learning)
   1. [Horizontal vs Vertical Scalability](#scalability)
      1. [Vertical Scalability](#vs)
      2. [Horizontal Scalability](#hs)
   2. [Possible problems in non-online learning](#probs)
   3. [Incremental learning libraries](#ill)
2. [Maximum Likelihood estimation](#mle)
3. [Argmax](#argmax)







# Online Learning<a name="online_learning"></a>

[reference tutorial](https://medium.com/value-stream-design/online-machine-learning-515556ff72c5)

A.K.A. - incremental learning of out-of-core learning.

usually ML models are static in nature - once parameters are learned, only inferencing can take place. 

They are also ***horizontally scalable***.



## Horizontal vs Vertical Scalability<a name="scalability"></a>

### Vertical Scalability<a name="vs"></a>

1. adding more resources (CPU/RAM/DISK) to your server (database or application server is still remains one) as on demand.
2. most commonly used in applications and products of middle-range as well as small and middle-sized companies. 
3. One of the most common examples is to buy an expensive hardware and use it as a Virtual Machine hypervisor (VMWare ESX).
4. usually means upgrade of server hardware. 
5. Some of the reasons to scale vertically includes increasing IOPS (Input / Output Operations), amplifying CPU/RAM capacity, as well as disk capacity.
6. However, even after using virtualisation, whenever an improved performance is targeted, the risk for down-times with it is much higher than using horizontal scaling.

### Horizontal scalability<a name="hs"></a>

1. means that higher availability of services required, adding more processing units or physical machines to your server/database
2. growing the number of nodes in the cluster
3. reducing the responsibilities of each member node by spreading the key space wider and providing additional end-points for client connections. 
4. Horizontal Scaling has been historically much more used for high level of computing and for application and services.
5. **Although this does not alter the capacity of each individual node**, the load is decreased due to the distribution between separate server nodes.
6. why organisations prefer this largely over *<u>vertical scalability</u>* is because of increasing I/O concurrency, reducing the load on existing nodes, and increasing disk capacity can be achieved with it.



ML practitioners do the following to *update*(learn from newer data-set)

1. They manually train on newer data, and deploy the resulting model once they are happy with its performance
2. They schedule training on new data to take place, say, once a week and automatically deploy the resulting model<a name="daywise"></a>.
   1. this could be achieved using crontab.

**Ideal requirement**: learn as well as predict in *real time*.





## Possible problems<a name="probs"></a>

1. algorithm itself might not be suitable
2. the model might fail to generalise well
3. the learning rate might be wrong
4. the regularisation might be too low or too high….......... 



In the accuracy/performance vs recent-knowledge trade-off, the later is chosen many a times, to make the best possible decisions right *now,* un-affordable to have a model that only knows about things that happened yesterday. 

Consider the following case:

* news website, displayed news article are according to type of topics clicked, and by whom they are clicked.
* predict type of news that users like, serve aptly.
* suppose government issues emergency, *everyone* is interested in domestic affairs
  * When presented with a news piece about the conference, a huge percentage of the audience clicks it to learn more.
* if [day-wise learning](#daywise) was used, model would be stuck at the same position, since its update-rate is too slow.
* With **online learning** there is no such thing as *yesterday's news*.
* On exposing the model to *internet*, biased learning leading to skewed-classes.
* if learning rate too high, model might forget info learnt a second ago
* overfit/underfit
* DDoS attacks fry up the model.
* technical architecture ![equation](https://latex.codecogs.com/gif.latex?%5Crightarrow) can't be horizontally scalable.



## Incremental learning libraries<a name="ill"></a>

[Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) and [the scikit inspired Creme](https://creme-ml.github.io/)



