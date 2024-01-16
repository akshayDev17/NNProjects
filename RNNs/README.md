# Table of Contents
1. [Introduction](#intro)
2. [Forward Propagation](#forward_prop)
3. [Many to One RNN Arch](#many_to_one_arch)
4. [One to Many RNN Arch](#one_to_many_arch)
5. [Many to many RNN Arch](#many_to_many_arch)

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

# Forward Propagation<a name="forward_prop"></a>
1. <img src="RNNForwardProp.png" />
    1. sigmoid because sentiment analysis is binary classification.
2. $O_1$ : output from 1st word of a given sample(sentence).
3.  <img src="RNNForwardProp_2.png" />

# Many to One RNN Arch<a name="many_to_one_arch"></a>
1. Sequence of words(many) is fed, and output is a single integer/scalar value(one)(as opposed to a vector).
2. Sentiment Analysis
    1. based on text, predict sentiment of text
3. Rating
    1. based on movie review, star-rating prediction

# One to Many RNN Arch<a name="one_to_many_arch"></a>
1. Non-sequential data as input(it doesn't have  a sense of timesteps)
2. Output is a time-step based value, i.e. a vector.
    1. an example of `return_sequences=True`
3. Image captioning
    1. given an image(time-less, stationery data), output a caption, i.e. a sequence of words.

# Many to many RNN Arch<a name="many_to_many_arch"></a>
1. Seq2Seq model
2. same and variable length many to many
    1. output may or may not have the same number of timesteps as the input sequence. 
    2. Machine translation = variable length many to many task.
        1. read the entire input sequence, then start outputing the output sequence.
        2. solved using encoder-decoder architecture.
