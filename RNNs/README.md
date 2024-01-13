# Table of Contents
1. [Introduction](#intro)

# Introduction<a name="intro"></a>

- sequential data, not necessarily words.
    - could be video, i.e. a sequence of images.
    - CNNs could be made recurrent - RNNs.
- ANNs cannot be used when input data is of variable size.
    - this is where RNNs come into picture.
    - <font color="red">Verify this, because padding(even for an RNN) is done for input sample, and the input layer can usually have a fixed number of units/weights.</font>
- <img src="simpleRNN.png" />
    - This is a simple recurrent layer.
    - along with the input vector being connected(weight matrix shaped (5,3)), the neurons within the layer are connected themselves to each other (weight matrix shaped (3,3)).
    - this *recurrent* connection is what sends the output(activated) from 1st word when the 2nd input, i.e. 2nd word (sample = sentence) is being processed by the layer.
        - **for the first word, this input happens to be random/null vector**.
- the layer also possesses activation function.

# Forward Propagation
1. <img src="RNNForwardProp.png" />
    1. sigmoid because sentiment analysis is binary classification.
2. $O_1$ : output from 1st word of a given sample(sentence).
3.  <img src="RNNForwardProp_2.png" />
