# Table of Contents

1. [Biological inspiration of activation Functions](#bio-insp-af)
2. [Activation Functions](#af)
   1. [step](#step)
   2. [sigmoid](#sigmoid)
3. [Steepest Descent](#sd)
   1. [Convergence Proof](#sd-proof)
      1. [convexity assumption](#convex)
      2. [Lipschitz continuous gradient assumption](#lcg)
      3. [3-point identity proof](#3pointidentityrproof)
      4. [Descent Lemma ](#descent-lemma)
      5. [Lemma-5](#lemma-5)
      6. [Final Convergence Criterion](#cc)
   2. [Cauchy approach of finding learning rate](#cauchy-find-learning-rate)
   3. [Barzilai and Borwein approach of finding learning rate](#bb-approach)
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
  - [Main source for the following proof](https://1202kbs.github.io/GD/) 

- ## Convergence Proof<a name="sd-proof"></a>

  - ### convexity assumption<a name="convex"></a>

    - the error/cost/loss function of the neural network is assumed to be convex in nature.
    - <img src="display_images/convex_function.png" width="400"/>
    - this essentially means that the line joining any 2 points on the function will always lie above its own curve in between these 2 points. Mathematically, this means that<img src="display_images/convexity_definition.png" /> ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctau%20%5Cin%20%5B0%2C1%5D)
    - in our case, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?f%3A%20%5Cmathcal%7BR%7D%5Ed%20%5Crightarrow%20R%20%5Ctextrm%7B%20and%20%7D%20%5Cmathbf%7Bx%7D%2C%5Cmathbf%7By%7D%20%5Cin%20%5Cmathcal%7BR%7D%5Ed), i.e. x and y are vectors of dimensionality d and f generates a scalar using a vector as an input.
      ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?f%20%3D%20f%28x%5E%7B%281%29%7D%2C%20x%5E%7B%282%29%7D%20%5Ccdots%20x%5E%7B%28d%29%7D%29%2C%20x%20%3D%20%5Cbegin%7Bbmatrix%7D%20x%5E%7B%281%29%7D%20%5C%5C%20x%5E%7B%282%29%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20x%5E%7B%28d%29%7D%20%5Cend%7Bbmatrix%7D)
    - this can also be explained in the following form, also known as the **first-order condition** of a convex function <img src="display_images/convex_function_first_order_condition.png" />
    - <font color="red">intuitive explanation pending</font>
      - [first order condition explained graphically](http://www.princeton.edu/~aaa/Public/Teaching/ORF523/S16/ORF523_S16_Lec7_gh.pdf)
      - <font color="red">prove this for 2 dimensional vectors, you have already done it with 1-D vectors ?</font>
      - this is called the [first-order condition](http://www.ifp.illinois.edu/~angelia/L3_convfunc.pdf) using the [convex function definition](https://www.tutorialspoint.com/convex_optimization/convex_optimization_differentiable_function.htm).
    - the following is the proof for the fact that if a function obeys this first-order condition, it is also a convex function
    - <img src="proofs/first_order_derivation.png" />
    - 

  - ### Lipschitz continuous gradient assumption<a name="lcg"></a>

    - the gradient of the cost/error/loss function is assumed to be Lipschitz continuous.
      <img src="display_images/lcg.png" /> , where L is a positive constant known as the **Lipschitz constant**.
    - a Lipschitz continuous function can be explained in the [following graphical manner](https://www.youtube.com/watch?v=aWQbFU_eXvE)
      - <img src="display_images/lipschitz.jpeg" width="450"/>
      - the slope of any line joining two points on the function will have an absolute value of its slope at most L.
      - this means that if a point is chosen at random and a line of slope greater than L or less than -L is drawn, then **if it is Lipschitz continuous**, the line **will not cut the function at any other point**.

    - <font color="red">how to test if a function holds lipschitz condition or not?</font>

  - ### 3-point identity proof<a name="3pointidentityrproof"></a>

    - Bra-ket notation in vectors<img src="display_images/braket.png" />
    - <img src="display_images/3-point-identity.png" />
    - <img src="proofs/3-point-identity-proof.png" />

  - ### Descent Lemma <a name="descent-lemma"></a>

    - <img src="display_images/descent-lemma.png" />
    - **Proof**
      <img src="proofs/descent-lemma-proof.png" />
    - <img src="proofs/g-to-f-proof.png" />

  - ### Lemma-5<a name="lemma-5"></a>

    - Couldn't think of a better name, hence lemma-5.
    - <img src="display_images/lemma-5.png" />
    - **Proof**:
      <img src="proofs/lemma-5-proof.png" />
    - <font color="red">intuitive explanation remaining!!!!</font>
    - see, you need ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?x-z) and ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?x-z%5E&plus;) since z and ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?z%5E&plus;) will be substituted with ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?z_k) and ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?z_%7Bk&plus;1%7D) (as will be seen in the [next section](#cc)) , and these will cancel out on summing across all iterations, hence while arriving at this identity, think of these 2 magnitude terms being involved in the inequality.

  - ### Final Convergence Criterion<a name="cc"></a>

    - x\* is the optimal point, i.e. the point at which the loss function is the least.
    -  <img src="proofs/convergence-proof.png" />
    - 

  - 

- 

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