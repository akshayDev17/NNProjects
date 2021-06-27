# Table of Contents

1. [Random Variables](#rv)
2. [Continuous Variables and Probability Density Functions](#pdfs)
3. [Marginal Probability](#marginal-distribution)
4. [Conditional Probability](#cp)
6. [Independence and Conditional Independence](#independence)
7. [Expectation, Variance and Covariance](#expectation)
8. [Bernoulli Distribution](#bernoulli)
9. [Multinoulli distribution](#multinoulli)
10. [Multivariate Normal Distribution](#multivariate_normal)
11. [Laplace Distribution](#laplace)
12. [Dirac Distribution](#dirac)
13. [Mixtures of Distributions](#mixture)
    - [Latent Variable](#latent)
14. [Useful Properties of Common Functions](#properties)
16. [Technical Details of Continuous Variables](#details)
17. [Information Theory](#information_theory)
16. [Kullback-Liebler Divergence](#kld)
17. [Cross-Entropy](#cross-entropy)
18. [Structured Probabilistic Models](#spm)
    1. [Directed Graphical Models](#dgm)
    2. [Un-Directed Graphical Models](#ugm)



# [Random Variables<a name="rv"></a>](https://docs.google.com/document/d/1hwV8jragvdL-sDUv8RsOHtB83iBXWJIKCcvu-DjHc38/edit#heading=h.76ktuxjm8bxo)



# [Continuous Variables and Probability Density Functions<a name="pdfs"></a>](https://docs.google.com/document/d/1hwV8jragvdL-sDUv8RsOHtB83iBXWJIKCcvu-DjHc38/edit#heading=h.pmbracoxoabf)





# Marginal Probability<a name="marginal-distribution"></a>

- we know the probability distribution over a set of variables and we want to know the probability distribution over just a subset of them.
- this distribution over the **subset** is called **marginal probability distribution**.
- we have discrete random variables x and y, and we know P (x; y). We can find P (x)
  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cforall%20x%20%5Cin%20%5Ctextrm%7Bx%7D%2C%20P%28%5Ctextrm%7Bx%7D%20%3D%20x%29%20%3D%20%5Csum%5Climits_%7By%7D%20P%28%5Ctextrm%7Bx%7D%3Dx%2C%20%5Ctextrm%7By%7D%3Dy%29)
- For continuous variables, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?p%28x%29%20%3D%20%5Cint%20p%28x%2C%20y%29%20dy)



# [Conditional Probability<a name="cp"></a>](https://docs.google.com/document/d/1hwV8jragvdL-sDUv8RsOHtB83iBXWJIKCcvu-DjHc38/edit#heading=h.n9648tjdwlzw)





# [Independence and Conditional Independence<a name="independence"></a>](https://docs.google.com/document/d/1hwV8jragvdL-sDUv8RsOHtB83iBXWJIKCcvu-DjHc38/edit#heading=h.s9rvpbuz3ej4)







# Expectation, Variance and Covariance<a name="expectation"></a>

- Covariance matrix
  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20%5C%2C%2C%5C%2C%20cov%28%5Cpmb%7Bx%7D%29_%7Bi%2C%20j%7D%20%3D%20cov%28x_i%2C%20x_j%29%20%5C%2C%2C%5C%2C%20cov%28x_i%2C%20x_i%29%20%3D%20var%28x_i%29)
  - i.e. x is a vector of dimensionality n.





# [Bernoulli Distribution<a name="bernoulli"></a>](https://docs.google.com/document/d/1hwV8jragvdL-sDUv8RsOHtB83iBXWJIKCcvu-DjHc38/edit#heading=h.8nal59hd5qah)



# Multinoulli distribution<a name="multinoulli"></a>

- this has a total of k states are possible, as opposed to only 2 states(x = 0 and x = 1) in Bernoulli distribution .
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%20state%20%7D%20%5Cpmb%7Bx%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5C%5C%20%5Cvdots%20%5C%5C%20x_k%20%5Cend%7Bbmatrix%7D%20%5Ctextrm%7B%20has%20probability%20vector%20%7D%20%5Cpmb%7Bp%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20p_1%20%5C%5C%20p_2%20%5C%5C%20%5Cvdots%20%5C%5C%20p_k%20%5Cend%7Bbmatrix%7D%20%5Ctextrm%7B%20such%20that%20%7D%20p_i%20%5Ctextrm%7B%20is%20the%20probability%20of%20state%20%7D%20x_i%20%5Cin%20%5B0%2C%201%5D)
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Csum%5Climits_%7Bi%3D1%7D%5En%20p_i%20%3D%201%20%5Ctextrm%7B%2C%20i.e.%20the%20sum%20of%20all%20components%20of%20the%20probability%20vector%20%3D%201%7D)
- The Bernoulli and Multinoulli distributions are sufficient to describe any distribution over their domain.
  - both are very generic descriptions of a p.d.f. since we don't exactly know how p,q(in Bernoulli) and p_1,p_2..p_k (in Multinoulli) are generated, they may very well be even complicated p.d.f.'s





# Multivariate Normal Distribution<a name="multivariate_normal"></a>

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BN%7D%28%5Cpmb%7Bx%7D%3B%20%5Cpmb%7B%5Cmu%7D%2C%20%5Cpmb%7B%5CSigma%7D%29%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7B%282%5Cpi%29%5En%20det.%28%5CSigma%29%7D%7D%20exp%5Cleft%28-%5Cfrac%7B1%7D%7B2%7D%20%28%5Cpmb%7Bx%7D%20-%20%5Cpmb%7B%5Cmu%7D%29%5ET%20%5Cpmb%7B%5CSigma%7D%5E%7B-1%7D%20%28%5Cpmb%7Bx%7D%20-%20%5Cpmb%7B%5Cmu%7D%29%20%5Cright%20%29%20%5Cnewline%20%5Cpmb%7Bx%7D%2C%20%5Cpmb%7B%5Cmu%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20%5C%2C%5C%2C%5C%2C%2C%20%5C%2C%5C%2C%20%5Cpmb%7B%5CSigma%7D%20%5Ctextrm%7B%20is%20the%20covariance%20matrix%7D)





# Laplace Distribution<a name="laplace"></a>

- place a sharp peak of probability mass at an arbitrary point ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmu) is the Laplace distribution
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?L%28x%3B%20%5Cmu%20%5Cgamma%29%20%3D%20%5Cfrac%7B1%7D%7B2%5Cgamma%7D%20exp%20%5Cleft%28-%5Cfrac%7B%7Cx%20-%20%5Cmu%7C%7D%7B%5Cgamma%7D%20%5Cright%20%29)
- on increasing mu:
  <img src="change_mu.gif" />
- on increasing gamma:
  <img src="change_gamma.gif" />





# Dirac Distribution<a name="dirac"></a>

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?p%28x%29%20%3D%20%5Cdelta%28x%20-%20%5Cmu%29%20%3D%20%5Cbegin%7Bcases%7D%201%20%26%20x%20%3D%20%5Cmu%20%5C%5C%200%20%26%20%5Ctextrm%7Botherwise%20%7D%20%5Cend%7Bcases%7D)





# Mixtures of Distributions<a name="mixture"></a>

- made up of several component distributions. 

- On each trial, the choice of which component distribution should generate the sample is determined by sampling a component identity from a Multinoulli distribution.

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?P%28x%29%20%3D%20%5Csum%5Climits_%7Bi%7D%20P%28c%3Di%29P%28x%7Cc%3Di%29)

  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?P%28c%3Di%29) is the Multinoulli distribution over component identities
  - this distribution boil downs to basically, on each trial, randomly sample a p.d.f. followed by randomly sample a value for this r.v. using this sampled p.d.f.

- ## Latent Variable<a name="latent"></a>

  - is a random variable that we cannot observe directly. 
  - The component identity variable c of the mixture model provides an example. 
  - Latent variables may be related to x through the joint distribution, in this case, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?P%28x%2C%20c%29%20%3D%20P%28x%20%7C%20c%29P%28c%29)





# Useful Properties of Common Functions<a name="properties"></a>

- **logistic sigmoid** ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Csigma%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1%20&plus;%20exp%28-x%29%7D)
- **softplus function** ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20log%281%20&plus;%20exp%28x%29%29)
  - can be useful for producing the ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Csigma) parameter of a normal distribution because its range is ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%280%2C%20%5Cinfty%29).
  - this is a smoothened version of the ReLU function, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20max%280%2C%20x%29)
  - <img src="softened.png" />, red = softplus, blue = ReLU.
- <img src="properties.png" />







# Technical Details of Continuous Variables<a name="details"></a>

- Suppose we have two random variables, x and y, such that y = g(x), where g is an invertible, continuous, differentiable transformation. 
- One might expect that py(y) = px(g−1(y)). 
- This is actually not the case. 
- As a simple example, suppose we have scalar random variables x and y.
- Suppose y = x2 and x ∼ U(0; 1). 
- If we use the rule py(y) = px(2y) then py will be 0 everywhere except the interval [0; 12], and it will be 1 on this interval. 
- This means, which  violates the definition of a probability distribution.
- Recall that the probability of x lying in an infinitesimally small region with volume δx is given by p(x)δx. 
- Since g can expand or contract space, the infinitesimal volume surrounding x in x space may have
  different volume in y space.
- <img src="functional_space.png" />
- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BIn%20higher%20dimensions%2C%20the%20derivative%20generalizes%7D%20%5C%5C%20%5Ctextrm%7Bto%20the%20determinant%20of%20the%20Jacobian%20matrix-the%20matrix%20with%20%7D%20%5C%5C%20J_%7Bi%2C%20j%7D%20%3D%20%5Cfrac%7B%5Cpartial%20x_i%7D%7B%5Cpartial%20x_j%7D%20%5C%2C%5C%2C%2C%5C%2C%20%5Cpmb%7Bx%7D%5C%2C%2C%5C%2C%20%5Cpmb%7By%7D%20%5CRightarrow%20p_x%28%5Cpmb%7Bx%7D%29%20%3D%20p_y%28g%28%5Cpmb%7Bx%7D%29%29%5Cleft%7C%20det.%5Cleft%28%5Cfrac%7B%5Cpartial%20g%28%5Cpmb%7Bx%7D%29%7D%7B%5Cpmb%7Bx%7D%7D%20%5Cright%20%29%20%5Cright%7C)
  - <font color="red">remaining !!!</font>









# Information Theory<a name="information_theory"></a>

- basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred
- We would like to quantify information in a way that formalizes this intuition.
  - Likely events should have low information content, and in the extreme case, events that are  guaranteed to happen should have no information content whatsoever. 
  - Less likely events should have higher information content. 
  - Independent events should have additive information. 
  - For example, finding out that a tossed coin has come up as heads twice should convey twice as much information as finding out that a tossed coin has come up as heads once.
- Accordingly, **self-information** of an event x = x is ![https://latex.codecogs.com/gif.latex?I%28x%29%20%3D%20-log%28P%28x%29%29](https://latex.codecogs.com/gif.latex?I%28x%29%20%3D%20-log%28P%28x%29%29)
  - this is written in units of *<u>nats</u>*. 
  - One nat is the amount of information gained by observing an event of probability 1/e. 
  - Other texts use base-2 logarithms and units called bits or shannons.
  - <img src="information.png" /> -log(x) for x in the range [0,1]
- The amount of uncertainty is quantified in an entire probability distribution using the Shannon entropy
  ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?H%28x%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20P%7D%20%5Cleft%5BI%28x%29%5Cright%5D%20%3D%20-%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20P%7D%20%5Cleft%5Blog%28P%28x%29%29%20%5Cright%5D%20%3D%20H%28P%29)
- this is the expected amount of information in an event drawn from that distribution.
- as we can conclude from the expression, highly probable event(or distributions which are nearly deterministic for a given value of x, the r.v.) will have low Shannon entropy, and a less likely event(or distributions much closer to the uniform, i.e. all events are equally likely) will have high entropy
- **Shannon entropy** for **continuous** R.V. is called **differential entropy**.



# Kullback-Liebler Divergence<a name="kld"></a>

- for measuring **deviation amongst probability distributions**, the KL-divergence is used
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?D%28%7B%5Ccolor%7Bred%7DP%7D%7C%7CQ%29_%7BKL%7D%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20%7B%5Ccolor%7Bred%7DP%7D%28x%29%7D%20%5Cleft%5Blog%20%5Cfrac%7B%7B%5Ccolor%7Bred%7DP%28x%29%7D%7D%7BQ%28x%29%7D%20%5Cright%20%5D%20%3D%20%5Cint%20P%28x%29%5Cleft%5B%20log%28P%28x%29%29%20-%20log%28Q%28x%29%29%20%5Cright%5D%20dx%20%5Cnewline%20D%28%7B%5Ccolor%7Bred%7DQ%7D%7C%7CP%29_%7BKL%7D%20%3D%20%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20%7B%5Ccolor%7Bred%7DQ%7D%28x%29%7D%20%5Cleft%5Blog%20%5Cfrac%7B%7B%5Ccolor%7Bred%7DQ%28x%29%7D%7D%7BP%28x%29%7D%20%5Cright%20%5D%20%3D%20%5Cint%20Q%28x%29%5Cleft%5B%20log%28Q%28x%29%29%20-%20log%28P%28x%29%29%20%5Cright%5D%20dx)
  -  the extra amount of information  needed to send a message containing symbols drawn from probability distribution P , when we use a code that was designed to minimize the length of messages drawn from probability distribution Q <font color="Red">remaining !!! </font>
- as it is obvious from the formula, the divergence is **not commutative**, hence it matters whether you optimize ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%7B%5Ccolor%7Bred%7DD_%7BKL%7D%28P%7C%7CQ%29%20%7D%20%5Ctextrm%7B%20vs.%20%7D%20%7B%5Ccolor%7Bblue%7D%20D_%7BKL%7D%28Q%7C%7CP%29%20%7D)





# Cross-Entropy<a name="cross-entropy"></a>

- ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?H%28P%2C%20Q%29%20%3D%20H%28P%29%20&plus;%20D_%7BKL%7D%28P%7C%7CQ%29%20%3D%20-%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20P%7D%20log%28Q%28x%29%29)
- Minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence, because Q does not participate in the omitted term.
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?H%28%7B%5Ccolor%7Bred%7DP%7D%2C%20Q%29%20%3D%20H%28%7B%5Ccolor%7Bred%7DP%7D%29%20&plus;%20D_%7BKL%7D%28%7B%5Ccolor%7Bred%7DP%7D%7C%7CQ%29%20%3D%20-%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20%7B%5Ccolor%7Bred%7DP%7D%7D%20log%28Q%28x%29%29%20%5Cnewline%20H%28%7B%5Ccolor%7Bred%7DQ%7D%2C%20P%29%20%3D%20H%28%7B%5Ccolor%7Bred%7DQ%7D%29%20&plus;%20D_%7BKL%7D%28%7B%5Ccolor%7Bred%7DQ%7D%7C%7CP%29%20%3D%20-%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20%7B%5Ccolor%7Bred%7DQ%7D%7D%20log%28P%28x%29%29%20%5Cnewline%20min._%7BQ%7D%5Cleft%28%20H%28%7B%5Ccolor%7Bred%7DP%7D%2C%20Q%29%20%5Cright%20%29%20%3D%20min_%7BQ%7D%20%5Cleft%28%20-%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20%7B%5Ccolor%7Bred%7DP%7D%7D%20log%28Q%28x%29%29%20%5Cright%20%29%20%5Cnewline%20min._%7BQ%7D%5Cleft%28D_%7BKL%7D%28%7B%5Ccolor%7Bred%7DP%7D%7C%7CQ%29%20%5Cright%20%29%20%3D%20min._%7BQ%7D%5Cleft%28%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20%7B%5Ccolor%7Bred%7DP%7D%7D%20log%28P%28x%29%29-%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20%7B%5Ccolor%7Bred%7DP%7D%7D%20log%28Q%28x%29%29%20%5Cright%20%29%20%5Cnewline%20%5Ctextrm%7B%20as%20the%20first%20term%20is%20independent%20of%20%7D%20Q%28x%29%20%5C%5C%20min._%7BQ%7D%5Cleft%28D_%7BKL%7D%28%7B%5Ccolor%7Bred%7DP%7D%7C%7CQ%29%20%5Cright%20%29%20%3D%20min._%7BQ%7D%5Cleft%28-%5Cmathbb%7BE%7D_%7Bx%20%5Ctextrm%7B%20from%20%7D%20%7B%5Ccolor%7Bred%7DP%7D%7D%20log%28Q%28x%29%29%20%5Cright%29)





# Structured Probabilistic Models<a name="spm"></a>

- Often, the multivariate probability distributions involve direct interactions between relatively few variables. 

  - Using a single function to describe the entire joint probability distribution can be very inefficient (both computationally and statistically).
  - Instead of using a single function to represent a probability distribution, we can split a probability distribution into many factors that we multiply together.
  - for instance, for 3 r.v.'s a,b,c, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?p%28a%2Cb%2Cc%29%20%3D%20p%28a%29.p%28b%7Ca%29.p%28c%7Cb%29)

- When we represent this factorization of a probability distribution with a graph, we call it a structured probabilistic model, or graphical model.

- there are 2 main kinds: directed and undirected.

- ## Directed Graphical Models<a name="dgm"></a>

  - directed edges define parent and child node relationship, wherein the origin of the edge is the parent node and the node to which the arrow-head points is the child node.
  - ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bthe%20conditional%20distribution%20over%20%7D%20x_i%20%5Ctextrm%7B%20given%20the%20parents%20of%20%7D%20x_i%20%5Ctextrm%7B%2C%20denoted%20by%20%7D%20Pa_%7B%5Cmathcal%7BG%7D%7D%28x_i%29%20%5Cnewline%20p%28x%29%20%3D%20%5Cprod_%7Bi%7D%20p%28x_i%20%7C%20Pa_%7B%5Cmathcal%7BG%7D%7D%28x_i%29%29)
  - <img src="dgm.png" />
    ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?p%28a%2Cb%2Cc%2Cd%2Ce%29%20%3D%20p%28a%29.p%28b%7Ca%29.p%28c%7Ca%2Cb%29.p%28d%7Cb%29.p%28e%7Cc%29)

  - 

- ## Un-Directed Graphical Models<a name="ugm"></a>

  - no notion of parent and child node.
  - **Clique**:  Any set of nodes that are all connected to each other in the graph ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BG%7D)
  - each clique ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BC%7D%5E%7B%28i%29%7D) is associated with a function, ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?%5Cphi%5E%7B%28i%29%7D%20%28%5Cmathcal%7BC%7D%5E%7B%28i%29%7D%29)
  - These factors are just functions, **not probability distributions**. 
  - The **output** of each factor **must be non-negative**, but there is no constraint that the factor must sum or integrate to 1 like a probability distribution.
  - <img src="ugm.png" />
  - as we can see, the cliques are (a,b,c), (b,d) , (c,e), hence the probability distribution function is 
    ![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?p%28a%2Cb%2Cc%2Cd%2Ce%29%20%3D%20%5Cfrac%7B1%7D%7BZ%7D%20%5Cphi%5E%7B%281%29%7D%28a%2Cb%2Cc%29.%5Cphi%5E%7B%282%29%7D%28b%2Cd%29.%5Cphi%5E%7B%283%29%7D%28c%2C%20e%29%20%5CRightarrow%20%5Cint%20%5Cphi%5E%7B%281%29%7D%28a%2Cb%2Cc%29.%5Cphi%5E%7B%282%29%7D%28b%2Cd%29.%5Cphi%5E%7B%283%29%7D%28c%2C%20e%29%20da%5C%2C%20db%5C%2C%20dc%5C%2C%20dd%5C%2C%20de%5C%2C%20%3D%20Z)
  - <font color="red">find usage !!!</font>

- 