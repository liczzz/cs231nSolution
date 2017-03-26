# -*- coding: utf8 -*- 
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    
    ##############################################################################
    # numerical stability:                                                       #
    # Calculations that can be proven not to magnify approximation errors are    #
    # called numerically stable. An opposite phenomenon is instability.          #
    # Typically, an algorithm involves an approximate method, and in some cases  #
    # one could prove that the algorithm would approach the right solution in    #
    # some limit. Even in this case, there is no guarantee that it would         #
    # converge to the correct solution, because the floating-point round-off or  #
    # truncation errors can be magnified, instead of damped, causing the         #
    # deviation from the exact solution to grow exponentially.                   #     
    ##############################################################################
    # float 的精度范围大约是1e-38~1e38,因此在使用浮点数的时候便会出现数值稳定性的问题，
    # 对于本问题，涉及到exp，因此如果scores太大，则exp(scores)太大，出现inf，数值不稳定，
    # 但当scores非常小时，exp(scores)为0，比较符合事实，
    # 因此为了保证数值稳定性，对scores统一减去了max(scores)，
    # 对于此问题可以证明，理论上的loss值的计算等价。
    #Shift scores by subtracting maximum value to keep all scores <= 0 to handle numeric instability
    scores -= np.max(scores)
    
    prob = np.exp(scores) / np.sum(np.exp(scores))
    loss += -np.log(prob[y[i]])
    
    for j in range(num_classes):
        if j == y[i]:
            dW[:, j] += X[i].T * (prob[j] - 1)
        else:
            dW[:, j] += X[i].T * prob[j]
    
    
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)    
  prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  loss = -np.sum(np.log(prob[np.arange(num_train), y])) / num_train + 0.5 * reg * np.sum(W * W)
  prob[np.arange(num_train), y] -= 1.0
  dW += np.dot(X.T, prob) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

