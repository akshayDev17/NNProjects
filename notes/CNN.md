[Reference link for these notes](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721)

# Andrew-NG lectures

## Lec1- Computer Vision

### Computer Vision Problems

1. Image classification: cat-identification, mnist number identification.
2. Object detection: self-driving cars have to not only detect other cars but also their position as well.
3. Neural Style Transfer: apps that *cartoonify* your face, apply some sort of neural transfer.

<span style="color:red;">1 of the few problems with CV </span>: inputs can get very big(suppose 64$\times$ 64$\times$ 3 (RGB channel) = 12288 features, but for a high resolution image, like 1000$\times$ 1000$\times$ 3(RGB channel) = 3M features) now if  a fully-connected layer is used, with the first hidden layer contains 1000 hidden units, then total weights = 1000$\times$ 3M matrix = 3B parameters. computational, memory requirements $\uparrow$ , difficult to get enough data to prevent NN from over-fitting.



To handle such large-res images, convolutions, and in turn CNNs are used.