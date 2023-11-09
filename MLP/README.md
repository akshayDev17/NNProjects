# Perceptron
1. single layer NN
2. activation = Step function S; S = (`sum >= 0 ? 1 : 0`)
    1. sum = $\sum w_ix_i$
3. basically a binary classifier.
4. **Drawback** : can only work in problems where the samples are distributed in a cluster format, i.e. all positive samples are on one side of the d-dimensional space, and the negative on the other.
    1. this can be verified as a 2D feature set problem, such that the perceptron becomes z = (Ax + By) >= 0 , x and y are the features.
    2. would fail/perform poorly, say when positive samples are inside a d-dimensional closed region(such as a sphere) and all negative samples are outside of it.
5. for binary classification, hinge-loss (seen in SVM) is used, with y = (-1, 1) and step function as (-1, 1) \[instead of (0,1)\].
    1. $\mathcal{L} = \sum\limits_{i=1}^n max(0, -y_i f(x_i)) $ , $f(x_i) = \sum\limits_{j=1}^d w_jx_{i, j} $
    2. when $y_i f(x_i) \ge 0$ then $\mathcal{L} = 0$ (meaning both are of the same sign, i.e. perfect binary classification).
    3. when $y_i f(x_i) \lt 0$ then $\mathcal{L} = -y_i f(x_i) $ (meaning both are of the opposite sign, i.e. perfect mislabelling).
    4. as for gradient update step, 
        1. $\frac{\partial \mathcal{L}}{\partial w_j} = 0 $ if $y_i f(x_i) \ge 0$, since loss is constant at 0.
        2. If $y_i f(x_i) \lt 0$, $\frac{\partial \mathcal{L}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial f(x_i)}\times \frac{\partial f(x_i)}{\partial w_j}  = -y_i x_{i, j} $
        3. for this perceptron, f(x) is assumed to be Linear function.
6. Perceptron can morph into classic ML models as follows:
    1. by itself, if the step function is used in the -1,1 format, then its a linear SVM.
    2. if instead of step function, sigmoid is used for activation, with hinge-loss replaced by binary cross entropy, this becomes a logistic regressor.
    3. if softmax is used for activation, with categorical cross-entropy as loss, this becomes a multi-class classifier.
    4. if MSE if used as loss, this becomes a linear regressor (assuming linear activation).

# Multi-Layer Perceptron