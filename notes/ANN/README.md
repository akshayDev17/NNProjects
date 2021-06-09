# Table of Contents

1. [Biological inspiration of activation Functions](#bio-insp-af)
2. [Activation Functions](#af)
   1. [step](#step)
   2. [sigmoid](#sigmoid)
3. [Steepest Descent](#sd)
   1. [Cauchy approach of finding learning rate](#cauchy-find-learning-rate)
   2. [Barzilai and Borwein approach of finding learning rate](#bb-approach)
4. [Back-Propagation](#backprop)
5. [References](#references)





# Biological inspiration of activation Functions<a name="bio-insp-af"></a>

- in biological neurons, electrical signals are conducted only when  the neuronal membrane potential rises above  a certain threshold potential value.
- A function that takes the input signal and generates an output signal, but takes into account some kind of threshold is called an activation function. 



# Activation functions<a name="af"></a>



## step<a name="step"></a>

- once the threshold input is reached, output jumps up.





## sigmoid<a name="sigmoid"></a>

- smoother than the step function 
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-x%7D%7D)
  - e = 2.71828
  - x = w1x1 + w2x2 ... wnxn , if this sum is smaller than some threshold, the *neuron* will *not fire*, if its larger, then 
- 





y = WX , X is n x 1 input and W is m x n weight matrix, since y can have dimensions that are different in number than that of X.

y can either be the final output or the next layer. layer l having n neurons has to be connected to layer l+1 having m neurons, hence the weight is  a matrix(unlike a vector in linear regression).

- after generating this matrix , X (l+1) , apply the activation function on each element, and the obtained vector is passed in as input to the next layer.
  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?X%5E%7B%28l&plus;1%29%7D%20%3D%20O%5El%20%3D%20g%5Cleft%28%5Cmathbf%7BW%5E%7Bl%2C%20l&plus;1%7D.X%5El%7D%20%5Cright%20%29)



- The middle layers are also known as hidden layers, and the name just stuck because the outputs of the middle layer are not necessarily made apparent as (final)outputs, so are *hidden*.





# Steepest Descent<a name="sd"></a>

- correctness of this algorithm

  - [taylor series involved](https://math.stackexchange.com/questions/4151297/different-form-of-taylor-series-in-leibniz-notation)
  - [gradient descent convergence](https://www.stat.cmu.edu/~ryantibs/convexopt-F13/scribes/lec6.pdf)
  - [gradient descent rigorous proof](https://math.stackexchange.com/questions/1746953/how-does-one-rigorously-prove-that-gradient-descent-indeed-decreases-the-functio)
  - [matrix/direction based proof](https://people.seas.harvard.edu/~yaron/AM221-S16/lecture_notes/AM221_lecture10.pdf)

- ## Cauchy approach of finding learning rate<a name="cauchy-find-learning-rate"></a>

  - One of the most obvious choices of *λ* is to choose the one that minimizes the objective function:
    *λ**k*=argmin*λ**f*(**x***k*−*λ*∇*f*(**x***k*))
  - This approach is conveniently called the steepest descent method.  
  - Although it seems to the best choice, it converges only linearly (error ∝1/*k*) and is very sensitive to ill-conditioning of problems.

- ## Barzilai and Borwein approach of finding learning rate<a name="bb-approach"></a>

  - An [approach proposed in 1988](http://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf) is to find the step size that minimizes:
    <img src="display_images/bb_gd.png" />







# Back-Propagation<a name="backprop"></a>

- <img src="display_images/backprop_1.png" />
- 







# References<a name="references"></a>

- Make Your Own Neural Network by Tariq Rashid