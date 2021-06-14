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



# Neural Nets - Introduction<a name="nnintro"></a>

- <img src="display_images/neuralNet.png" width="600"/>
  - ![equation](https://latex.codecogs.com/gif.latex?W_%7Bi%2Cj%7D), `i` means the layer supplying the signal, and `j` means the layer receiving the signal.
- ![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20O_%7B%5Ctextrm%7Binput%7D%7D%20%26%3D%20W_%7B%5Ctextrm%7Binput%7D%2C%20%5Ctextrm%7Bhidden%7D%7D%20%5Ccdot%20X_%7B%5Ctextrm%7Binput%7D%7D%20%5C%5C%20X_%7B%5Ctextrm%7Bhidden%7D%7D%20%26%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-O_%7B%5Ctextrm%7Binput%7D%7D%7D%7D%20%5C%5C%20O_%7B%5Ctextrm%7Bhidden%7D%7D%20%26%3D%20W_%7B%5Ctextrm%7Bhidden%7D%2C%20%5Ctextrm%7Boutput%7D%7D%20%5Ccdot%20X_%7B%5Ctextrm%7Bhidden%7D%7D%20%5C%5C%20Y_%7B%5Ctextrm%7Bactual%20output%7D%7D%20%26%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20e%5E%7B-O_%7B%5Ctextrm%7Bhidden%7D%7D%7D%7D%20%5C%5C%20%5Cend%7Balign*%7D)
  - in the case above, where the dimensionality of input, hidden and output layers is known(all are 3)
    ![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20W_%7Bi%2C%20h%7D%20%26%3D%20%5Cbegin%7Bbmatrix%7D%20w_%7B1%2C1%7D%5E%7Bi%2Ch%7D%20%26%20w_%7B2%2C1%7D%5E%7Bi%2Ch%7D%20%26%20w_%7B3%2C1%7D%5E%7Bi%2Ch%7D%20%5C%5C%20%5C%5C%20w_%7B1%2C2%7D%5E%7Bi%2Ch%7D%20%26%20w_%7B2%2C2%7D%5E%7Bi%2Ch%7D%20%26%20w_%7B3%2C2%7D%5E%7Bi%2Ch%7D%20%5C%5C%20%5C%5C%20w_%7B1%2C3%7D%5E%7Bi%2Ch%7D%20%26%20w_%7B2%2C3%7D%5E%7Bi%2Ch%7D%20%26%20w_%7B3%2C3%7D%5E%7Bi%2Ch%7D%20%5Cend%7Bbmatrix%7D%20%5C%5C%20W_%7Bh%2C%20o%7D%20%26%3D%20%5Cbegin%7Bbmatrix%7D%20w_%7B1%2C1%7D%5E%7Bh%2Co%7D%20%26%20w_%7B2%2C1%7D%5E%7Bh%2Co%7D%20%26%20w_%7B3%2C1%7D%5E%7Bh%2Co%7D%20%5C%5C%20%5C%5C%20w_%7B1%2C2%7D%5E%7Bh%2Co%7D%20%26%20w_%7B2%2C2%7D%5E%7Bh%2Co%7D%20%26%20w_%7B3%2C2%7D%5E%7Bh%2Co%7D%20%5C%5C%20%5C%5C%20w_%7B1%2C3%7D%5E%7Bh%2Co%7D%20%26%20w_%7B2%2C3%7D%5E%7Bh%2Co%7D%20%26%20w_%7B3%2C3%7D%5E%7Bh%2Co%7D%20%5Cend%7Bbmatrix%7D%20%5C%5C%20%5Cend%7Balign*%7D)
- 







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





# Back-Propagation<a name="backprop"></a>

- <img src="display_images/backprop_1.png" />
- <img src="display_images/error_backprop.png" />
  - error is distributed amongst the nodes of the hidden layer, i.e. the nodes that provide the input signal to the output layer
  - ![equation](https://latex.codecogs.com/gif.latex?e_%7B%5Ctextrm%7Boutput%7D%2C%201%7D%20%3D%20Y_%7B%5Ctextrm%7Bactual%20output%7D%2C1%7D%20-%20Y_%7B%5Ctextrm%7Btarget%7D%2C1%7D)
- <img src="display_images/error_hidden.png" />
- The normalizing is removed for convenience, and the following image(borrowed from [this blogpost](http://makeyourownneuralnetwork.blogspot.com/2016/07/error-backpropagation-revisted.html)), wherein the blue and green are almost indistinguishable , where the former denotes the non-normalized error and the latter the normalized one.
  <img src="display_images/error_with_without_normalization.png" />
  - <font color="red">still search WHY this works?</font>
- ![equation](https://latex.codecogs.com/gif.latex?Err_%7B%5Ctextrm%7Bhidden%7D%7D%20%3D%20w%5ET_%7B%5Ctextrm%7Bhidden%2C%20output%7D%7D%5Ccdot%20err_%7B%5Ctextrm%7Boutput%7D%7D)
- The way in which these weights are updated is using gradient descent.





# Steepest Descent<a name="sd"></a>

- this is the **theoretical way** of using gradient descent.

- correctness of this algorithm

  - [taylor series involved](https://math.stackexchange.com/questions/4151297/different-form-of-taylor-series-in-leibniz-notation)
  - [Main source for the following proof](https://1202kbs.github.io/GD/) 

- ![equation](https://latex.codecogs.com/gif.latex?z_%7B%5Ctextrm%7Bnew%7D%7D%20%3D%20z_%7B%5Ctextrm%7Bold%7D%7D%20-%20%5Calpha.%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20W%7D)

  - we increase z in the opposite direction to the gradient, i.e. a positive gradient means we reduce z, and a negative gradient means we increase z.

- ## Convergence Proof<a name="sd-proof"></a>

  - ### convexity assumption<a name="convex"></a>

    - the error/cost/loss function of the neural network is assumed to be convex in nature.
    - <img src="display_images/convex_function.png" width="400"/>
    - this essentially means that the line joining any 2 points on the function will always lie above its own curve in between these 2 points. Mathematically, this means that<img src="display_images/convexity_definition.png" /> ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctau%20%5Cin%20%5B0%2C1%5D)
    - in our case, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?f%3A%20%5Cmathcal%7BR%7D%5Ed%20%5Crightarrow%20R%20%5Ctextrm%7B%20and%20%7D%20%5Cmathbf%7Bx%7D%2C%5Cmathbf%7By%7D%20%5Cin%20%5Cmathcal%7BR%7D%5Ed), i.e. x and y are vectors of dimensionality d and f generates a scalar using a vector as an input.
      ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?f%20%3D%20f%28x%5E%7B%281%29%7D%2C%20x%5E%7B%282%29%7D%20%5Ccdots%20x%5E%7B%28d%29%7D%29%2C%20x%20%3D%20%5Cbegin%7Bbmatrix%7D%20x%5E%7B%281%29%7D%20%5C%5C%20x%5E%7B%282%29%7D%20%5C%5C%20%5Cvdots%20%5C%5C%20x%5E%7B%28d%29%7D%20%5Cend%7Bbmatrix%7D)
    - this can also be explained in the following form, also known as the **first-order condition** of a convex function <img src="display_images/convex_function_first_order_condition.png" />
    - **Intuitive explanation**
      - [first order condition explained graphically](https://people.seas.harvard.edu/~yaron/AM221-S16/lecture_notes/AM221_lecture8.pdf)
      - The first order Taylor expansion at any point(R.H.S.) is a global under-estimator of the function(L.H.S.) .
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
    - <img src="proofs/intermediate.png">
    - Hence, proving descent lemma requires **gradient of** the error/loss/**cost function** to be **Lipschitz continuous**.

  - ### Lemma-5<a name="lemma-5"></a>

    - Couldn't think of a better name, hence lemma-5.
    - <img src="display_images/lemma-5.png" />
    - **Proof**:
      <img src="proofs/lemma-5-proof.png" />
    - <font color="red">intuitive explanation remaining!!!!</font>
    - see, you need ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?x-z) and ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?x-z%5E&plus;) since z and ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?z%5E&plus;) will be substituted with ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?z_k) and ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?z_%7Bk&plus;1%7D) (as will be seen in the [next section](#cc)) , and these will cancel out on summing across all iterations, hence while arriving at this identity, think of these 2 magnitude terms being involved in the inequality.

  - ### Final Convergence Criterion<a name="cc"></a>

    - z\* is the optimal point, i.e. the point at which the loss function is the least.
    -  <img src="proofs/convergence-proof.png" />
    - 

  - 

- therefore, a cost/error/loss function **requires** 

  - **itself to be convex** and
  - **its gradient to be Lipschitz continuous**.

- 

- ## Cauchy approach of finding learning rate<a name="cauchy-find-learning-rate"></a>

  - One of the most obvious choices of *λ* is to choose the one that minimizes the objective function:
    *λ**k*=argmin*λ**f*(**x***k*−*λ*∇*f*(**x***k*))
  - This approach is conveniently called the steepest descent method.  
  - Although it seems to the best choice, it converges only linearly (error ∝1/*k*) and is very sensitive to ill-conditioning of problems.

- ## Barzilai and Borwein approach of finding learning rate<a name="bb-approach"></a>

  - An [approach proposed in 1988](http://pages.cs.wisc.edu/~swright/726/handouts/barzilai-borwein.pdf) is to find the step size that minimizes:
    <img src="display_images/bb_gd.png" />







- 





# Gradient Descent Practical Approach<a name="gd_practical"></a>

- <img src="display_images/practical_gradient.png" />
- A similar error slope for the weights between the input and hidden layers.
- <img src="display_images/gradient_update_generalized.png" />
  - here the forward prop happens from layer `i` to layer `j` ,and the backprop from` j` to `i`.
- <img src="display_images/update_equation.png" />
- ![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5Cbegin%7Bbmatrix%7D%20%5CDelta%20w_%7B1%2C1%7D%20%26%20%5CDelta%20w_%7B2%2C1%7D%20%26%20%5Ccdots%20%26%20w_%7Bk%2C%201%7D%20%5C%5C%20%5CDelta%20w_%7B1%2C2%7D%20%26%20%5Ccdots%20%26%20%5Ccdots%20%5C%5C%20%26%20%5Cvdots%20%26%20%5Cvdots%20%26%20%5C%5C%20%5CDelta%20w_%7B1%2Cj%7D%20%26%20%5CDelta%20w_%7B2%2Cj%7D%20%26%20%5Ccdots%20%26%20w_%7Bk%2C%20j%7D%20%5Cend%7Bbmatrix%7D%20%26%3D%20%5Calpha%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20E_1*S_1%281-S_1%29%20%5C%5C%20E_1*S_1%281-S_1%29%20%5C%5C%20%5Cvdots%20%5C%5C%20E_k*S_k%281-S_k%29%20%5Cend%7Bbmatrix%7D%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20O_1%20%26%20O_2%20%26%20%5Ccdots%20%26%20O_j%5Cend%7Bbmatrix%7D%20%5C%5C%20%5Cend%7Balign*%7D)
  - this is the weight matrix update equation.
  - observe that the partial differential of error(E) w.r.t. the weights has a negative sign, and the weight update equation also has a negative sign, hence these 2 cancel out, and thus we end up with a delta-change on the LHS.
  - errors are produced from the next(succeeding) layer, whereas the output vector is produced from the input(preceding) layer.
  - <img src="display_images/shorthand_delta_weights.png" />
- 







# Problem of saturation due to large weights/inputs/outputs<a name="saturation"></a>

- If the inputs are large, the activation function gets very flat.
- A very flat activation function is problematic because we use the gradient to learn new weights.
  <img src="display_images/saturation_gradient_sigmoid.png" />
-  This is called saturating a neural network. Hence, the inputs should be kept small.
- even tan(h) suffers from saturation of weights, i.e. gradient tending to 0.<img src="display_images/tan-h-saturation.jpeg" width="700" />
- A good recommendation is to re-scale inputs into the range 0.0 to 1.0.
- Even Outputs having large values cause saturation problems.



# Initializing Weights<a name="weights-init"></a>

- <font color="red">LEFT !!!!</font>



# References<a name="references"></a>

- Make Your Own Neural Network by Tariq Rashid