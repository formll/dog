# DoG Optimizer

This repository contains the implementation of the algorithms in the paper 
[DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule](https://arxiv.org/abs/2302.12022)
by Maor Ivgi, Oliver Hinder and Yair Carmon.

**IMPORTANT:** For best performance (and for fair comparison to other methods) **DoG/L-DoG must be combined with iterate averaging!** This package includes an easy-to-use [averager class](<#iterate-averaging>) - its default configuration should work well out of the box.

## Algorithm
DoG ("Distance over Gradients") is a parameter-free stochastic optimizer. 
DoG updates parameters $x_t$ with stochastic gradients $g_t$ according to:
```math
\begin{aligned}
   \eta_t & = \frac{ \bar{r}_t }{ \sqrt{\sum_{i \le t }{\lVert g_i\rVert ^2 + \epsilon}} } \\   
   x_{t+1} & = x_{t} - \eta_t \cdot g_t
  \end{aligned}
```
where
```math
\begin{equation*}
\bar{r}_t = \begin{cases}
\text{max}_{i \le t}{\lVert x_i - x_0 \rVert} & t \ge 1 \\
r_{\epsilon} & t=0.
\end{cases}
\end{equation*}
```
The initial movement parameter $r_{\epsilon}$ should be chosen small relative to the distance between $x_0$ and the nearest optimum $x^\star$ (see additional discussion below).

LDoG (layerwise DoG) is a variant of DoG that applies the above update rule separately to every element in the list of parameters provided to the optimizer object.

## Installation
To install the package, simply run `pip install dog-optimizer`.

## Usage
DoG is implemented using both the standard PyTorch optimizer interface, and using the standard JAX optax interface.

### PyTorch usage
DoG and LDoG are implemented using the standard pytorch optimizer interface. After installing the pacakge with `pip install dog-optimizer`,
all you need to do is replace the line that creates your optimizer with 
```python
from dog import DoG
optimizer = DoG(optimizer args)
```
for DoG, or
```python
from dog import LDoG
optimizer = LDoG(optimizer args)
```
for LDoG, 
where `optimizer args` follows the standard pytorch optimizer syntex. 
To see the list of all available parameters, run `help(DoG)` or `help(LDoG)`.

### JAX usage
DoG and LDoG are implemented using the standard optax interface. After installing the pacakge with `pip install dog-optimizer`,
all you need to do is replace the line that creates your optimizer with 
```python
    from dog import DoGJAX as DoG, LDoGJAX as LDoG, polynomial_decay_averaging
    import optax
    ldog = True  # wther to use LDoG or DoG
    opt_class = LDoG if ldog else DoG
    optimizer = opt_class(learning_rate=1, reps_rel=1e-6, eps=1e-8, init_eta=None, weight_decay=0)

    averager = polynomial_decay_averaging(gamma=8)
    optimizer = optax.chain(optimizer, averager)
```

When you finish trianing (or evaluating a checkpoint) and want to get the averaged model, simply do
```python
    from dog import get_av_model
    logits = state.apply_fn({'params': get_av_model(state.opt_state)}, batch['image'])
    loss = compute_loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)
```

### Iterate averaging
We provide an implementation of the polynomial decay averaging used throughout our experimenters. To use it simply creates a `PolynomialDecayAverager` with 
```python
from dog import PolynomialDecayAverager
averager = PolynomialDecayAverager(model)
```
then, after each `optimizer.step()`, call `averager.step()` as well.
You can then get both the current model and the averaged model with `averager.base_model` and `averager.averaged_model` respectively.

### Example script
An example of how to use the above to train a simple CNN on MNIST can be found in `examples.py` 
(based on this [pytorch example](https://github.com/pytorch/examples/blob/main/mnist/main.py)).

An additional example of using the JAX implementation can be found in `jax_example.py`.

### Choosing `reps_rel`
DoG is parameter-free by design, so there is no need to tune a learning rate parameter. 
However, as discussed in the paper, DoG has an initial step movement parameter 
$r_{\epsilon}$ that must be small enough to avoid destructively updates that cause divergence, 
but an extremely small value of $r_{\epsilon}$ would slow down training. 
We recommend choosing $r_{\epsilon}$ relative to the norm of the initial weights $x_0$. In particular, we set 
$r_{\epsilon}$ to be `reps_rel` $\times (1+\rVert x_0 \lVert)$, where `reps_rel` is a configurable parameter of the optimizer. The default value 
of `reps_rel` is 1e-6, and we have found it to work well most of the time. However, in our experiments we did encounter 
some situations that required different values of `reps_rel`:
- If optimization diverges early, it is likely that `reps_rel` (and hence $r_{\epsilon}$) is too large: 
try decreasing it by factors 100 until divergence no longer occurs. This happened when applying LDoG to fine-tune T5, 
which had large pre-trained weights; setting `reps_rel` to 1e-8 eliminated the divergence.
- If the DoG step size (`eta`) does not substantially increase from its initial value for a few hundred steps, it could be that `reps_rel` is too small: 
try increasing it by factors of 100 until you see `eta` starting to increase in the first few steps. 
This happened when training models with batch normalization; setting `reps_rel` to 1e-4 eliminated the problem.


## Citation
```
@inproceedings{ivgi2023dog,
  title={{D}o{G} is {SGD}'s best friend: A parameter-free dynamic step size schedule},
  author={Maor Ivgi and Oliver Hinder and Yair Carmon},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023},
}
```
