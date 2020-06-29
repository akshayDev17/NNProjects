# Table of Contents

1. [The model](#model)
   1. [Localising feature vectors](#localising_vectors)
2. [Training the model](#training)
   1. [Derivation](#derivation)
   2. [Local-feature representation](#decompose)









# The model<a name="model"></a>

* based on conditional log–linear models.
*  Each sentence ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D) for i = 1 . . . n in our training data has a set of ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bn%7D_%7B%5Ctextrm%7Bi%7D%7D) (each sentence has ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bn%7D_%7B%5Ctextrm%7Bi%7D%7D) number of parse trees) candidate parse trees ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C1%7D%7D%2C%20.....%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cn%7D_%7B%5Ctextrm%7Bi%7D%7D%7D) , which are the output of an N–best baseline parser. 
* Each candidate parse has an associated F–measure score,indicating its similarity to the gold–standard parse.
  * ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C1%7D%7D) parse tree with highest F-score for ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D).
* for a parse tree ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D) this model assigns hidden variables to each word in this tree.
  * if ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D) spans *m* words, then the hidden-variable value domain will be for each word are the sets ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BH%7D_%7B%5Ctextrm%7B1%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D%29%2C%5C%2C%20%5Ctextrm%7BH%7D_%7B%5Ctextrm%7B2%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D%29%2C%20......%5Ctextrm%7BH%7D_%7B%5Ctextrm%7Bm%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D%29) 
* global hidden-value assignment: assigns a hidden-variable value to each word in ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D) <a name="global_assgn_word"></a>
  * ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%3D%28%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7B1%7D%7D%2C%5C%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7B2%7D%7D%2C%20.....%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bm%7D%7D%29%5C%2C%20%5Cepsilon%5C%2C%20%5Ctextrm%7B%5Ctextbf%7BH%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29) where ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%5Ctextbf%7BH%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29%20%5C%2C%20%3D%20%5Ctextrm%7BH%7D_%7B%5Ctextrm%7B1%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D%29%5Ctimes%20%5Ctextrm%7BH%7D_%7B%5Ctextrm%7B2%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D%29%5Ctimes%20......%5Ctextrm%7BH%7D_%7B%5Ctextrm%7Bm%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2Cj%7D%7D%29) or the cardinal product across domain of each word.
  * hence a hidden-variable value is attached to each word.
  * ![equation](https://latex.codecogs.com/gif.latex?%5CPhi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%7D%2C%20%5Ctextrm%7Bj%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29%20%5C%2C%20%5Cepsilon%20%5C%2C%5Cmathbb%7BR%7D%5E%7B%5Ctextrm%7Bd%7D%7D) is a d-dimensional feature vector defined 
    * ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bk%7D%5E%7B%5Ctextrm%7Bth%7D%7D) component ![equation](https://latex.codecogs.com/gif.latex?%5CPhi_%7B%5Ctextrm%7Bk%7D%7D) = count of some substructure
    * for instance, <a name="global_rep"></a>
      ![equation](https://latex.codecogs.com/gif.latex?%5CPhi_%7B%5Ctextrm%7B12%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29) might be number of times the word *the* occurs with hidden value ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bthe%7D_%7B%5Ctextrm%7B3%7D%7D) and part of speech tag DT in ![equation](https://latex.codecogs.com/gif.latex?%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29). 
      ![equation](https://latex.codecogs.com/gif.latex?%5CPhi_%7B%5Ctextrm%7B101%7D%7D%20%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29) might be number of times ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BCEO%7D_%7B%5Ctextrm%7B1%7D%7D) appears as the subject of ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bowns%7D_%7B%5Ctextrm%7B2%7D%7D) in ![equation](https://latex.codecogs.com/gif.latex?%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29). 
    * parameter vector ![equation](https://latex.codecogs.com/gif.latex?%5CTheta%5C%2C%20%5Cepsilon%5C%2C%20%5Cmathbb%7BR%7D%5E%7B%5Ctextrm%7Bd%7D%7D) used along with parse-tree and hidden-variable vectors to generate a log-likelihood distribution: ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bp%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%20%3D%20%5Cfrac%7B%5Ctextrm%7Be%7D%5E%7B%5CPhi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29.%5CTheta%7D%7D%7B%5Csum%20%5Climits_%7B%5Ctextrm%7Bj%7D%5E%7B%27%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%5C%2C%20%5Cepsilon%5C%2C%20%5Ctextrm%7B%5Ctextbf%7BH%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%27%7D%7D%29%7D%5Ctextrm%7Be%7D%5E%7B%5CPhi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%27%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29.%5CTheta%7D%7D)  
* after marginalising out the global assignment(**h**), we have ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bp%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%5C%2C%7C%5C%2C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%3D%5Csum%5Climits_%7B%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%5C%2C%20%5Cepsilon%5C%2C%20%5Ctextrm%7B%5Ctextbf%7BH%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29%7D%20%5Ctextrm%7Bp%7D%28%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%2C%20%5C%2C%20%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%5C%2C%7C%5C%2C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29) 
* NLL w.r.t. &Theta;  = ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BL%28%7D%5CTheta%29%3D%20-%5Csum%5Climits_%7Bi%7D%5Ctextrm%7Blog%28%7D%5Ctextrm%7Bp%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%201%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%29%20%3D%20-%5Csum%5Climits_%7Bi%7D%5Ctextrm%7Blog%28%7D%5Csum%20%5Climits_%7B%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%5C%2C%20%5Cepsilon%5C%2C%20%5Ctextrm%7B%5Ctextbf%7BH%7D%20%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%201%7D%7D%29%7D%20%5Ctextrm%7Bp%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%201%7D%7D%2C%20%5C%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%29) 
* the number of global assignments ![equation](https://latex.codecogs.com/gif.latex?%7C%5Ctextrm%7B%5Ctextbf%7BH%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29%7C) increases exponentially with the increase in number of words in the sentence.
  * hence need to restriction to local space.
  * this is achieved using dependency structure.





## Localising feature vectors<a name="localising_vectors"></a>

<img src="images/parseNdependency.png" />

* Its the graph formed by joining any 2 words u,v &epsilon; ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D) with an edge, if and only if a *head-modified dependency* exists between them.

  * denoted by ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%5Ctextbf%7BD%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29) , which contains the set of edges (u, v) for all word-pairs that follow the aforementioned constraint.

* w,u,v - word indices(![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bw%7D%5E%7B%5Ctextrm%7Bth%7D%7D%2C%5C%2C%5Ctextrm%7Bu%7D%5E%7B%5Ctextrm%7Bth%7D%7D%2C%5C%2C%5Ctextrm%7Bv%7D%5E%7B%5Ctextrm%7Bth%7D%7D) words of sentence ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D)),  *single-variable local feature vector* ![equation](https://latex.codecogs.com/gif.latex?%5Cphi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7Bw%7D%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bw%7D%7D%29%5C%2C%5Cepsilon%5C%2C%20%5Cmathbb%7BR%7D%5E%7B%5Ctextrm%7Bd%7D%7D) and pair-wise local feature vector ![equation](https://latex.codecogs.com/gif.latex?%5Cphi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7Bu%7D%2C%20%5Ctextrm%7Bv%7D%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bu%7D%7D%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bv%7D%7D%29%5C%2C%5Cepsilon%5C%2C%20%5Cmathbb%7BR%7D%5E%7B%5Ctextrm%7Bd%7D%7D) ([we already know what h-u, h-v are](#global_assgn_word)).

* on decomposing the global feature vector as a sum of these local feature vectors, we get ![equation](https://latex.codecogs.com/gif.latex?%5CPhi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29%20%3D%20%5Csum%5Climits_%7B%5Ctextrm%7B1%7D%20%5Cle%20%5Ctextrm%7Bw%7D%20%5Cle%20%5Ctextrm%7Bm%7D%7D%20%5Cphi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7Bw%7D%2C%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bw%7D%7D%29%20&plus;%20%5Csum%5Climits_%7B%28%5Ctextrm%7Bu%2C%20v%7D%29%20%5C%2C%20%5Cepsilon%20%5C%2C%20%5Ctextrm%7B%5Ctextbf%7BD%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29%20%7D%20%5Cphi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7Bu%7D%2C%20%5Ctextrm%7Bv%7D%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bu%7D%7D%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bv%7D%7D%29) 

* <img src="images/predicate.png" width=500/> 

  this tells us that each component of the local feature vectors is indicative of a count of occurrences of a particular feature .

  * after summing up these local feature vectors over the parse tree, we get ![equation](https://latex.codecogs.com/gif.latex?%5CPhi_%7B%5Ctextrm%7B12%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29) or ![equation](https://latex.codecogs.com/gif.latex?%5CPhi_%7B%5Ctextrm%7B101%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29) , as defined [here](#global_rep).







# Training the model<a name="training"></a>

as define [here]() the loss function was ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7BL%28%7D%5CTheta%29%3D%20-%5Csum%5Climits_%7Bi%7D%5Ctextrm%7Blog%28%7D%5Ctextrm%7Bp%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%201%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%29%20%3D%20-%5Csum%5Climits_%7Bi%7D%5Ctextrm%7Blog%28%7D%5Csum%20%5Climits_%7B%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%5C%2C%20%5Cepsilon%5C%2C%20%5Ctextrm%7B%5Ctextbf%7BH%7D%20%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%201%7D%7D%29%7D%20%5Ctextrm%7Bp%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%201%7D%7D%2C%20%5C%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%29) 

gradient:
![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5CTheta%7D%20%3D%20-%5Csum%20%5Climits_%7Bi%7D%5Ctextrm%7B%5Ctextbf%7BF%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%201%7D%7D%2C%20%5CTheta%29%20&plus;%20%5Csum%5Climits_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%5Ctextrm%7Bp%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%7C%20%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%5Ctextrm%7B%5Ctextbf%7BF%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5CTheta%29) , where ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7B%5Ctextbf%7BF%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5CTheta%29%20%3D%20%5Csum%5Climits_%7B%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%5C%2C%5Cepsilon%5C%2C%20%5Ctextrm%7B%5Ctextbf%7BH%7D%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29%7D%20%5Cfrac%7B%5Ctextrm%7Bp%28%7D%20%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%20%29%7D%20%7B%5Ctextrm%7Bp%28%7D%20%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%20%29%7D%20%5CPhi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29) .



## Derivation<a name="derivation"></a>







## Local-feature representation<a name="decompose"></a>

since  ![equation](https://latex.codecogs.com/gif.latex?%7C%5Ctextrm%7B%5Ctextbf%7BH%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29%7C) increases exponentially, direct calculation of ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bp%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%20%5C%2C%20%5Ctextrm%7Bor%7D%20%5C%2C%20%5Ctextrm%7BF%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5CTheta%29) is impractical. Instead, on decomposing the global features into local feature-vectors, we get 

* ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bp%28%7D%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%7C%5Ctextrm%7Bs%7D_%7B%5Ctextrm%7Bi%7D%7D%2C%20%5CTheta%29%20%3D%20%5Cfrac%7B%5Ctextrm%7BZ%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%7D%7B%5Csum%5Climits_%7B%5Ctextrm%7Bj%27%3D1%7D%7D%5E%7B%5Ctextrm%7Bn%7D_%7B%5Ctextrm%7Bi%7D%7D%7D%5Ctextrm%7BZ%7D_%7B%5Ctextrm%7Bi%2C%20j%27%7D%7D%7D) where ![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B120%7D%20%5Ctextrm%7BZ%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%20%3D%20%5Csum%5Climits_%7B%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%5C%2C%20%5Cepsilon%20%5C%2C%20%5Ctextrm%7B%5Ctextbf%7BH%7D%7D%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%29%7D%20%5Ctextrm%7Be%7D%5E%7B%5CPhi%28%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%5C%2C%20%5Ctextrm%7B%5Ctextbf%7Bh%7D%7D%29%20.%20%5CTheta%7D) is the normalisation constant.
* <img src="images/F_defn.png" width=400/> <img src="images/prob_defn.png" width=350 float=right/>
  these are the marginalised probabilities ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bp%28%7D%20%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7Bw%7D%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bw%7D%7D%29) and ![equation](https://latex.codecogs.com/gif.latex?%5Ctextrm%7Bp%28%7D%20%5Ctextrm%7Bt%7D_%7B%5Ctextrm%7Bi%2C%20j%7D%7D%2C%20%5Ctextrm%7Bu%7D%2C%20%5Ctextrm%7Bv%7D%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bu%7D%7D%2C%20%5Ctextrm%7Bh%7D_%7B%5Ctextrm%7Bv%7D%7D%29) .
* the normalisation constant and marginalised probabilities can be computed using [belief propagation](https://dl.acm.org/doi/10.5555/779343.779352), a DP technique.
* using this technique of gradient calculation, minimise the loss using SGD(stochastic)