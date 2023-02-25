# DoG Optimizer

This repository contains the implementation of the algorithms in the paper 
[DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule](https://arxiv.org/abs/2302.12022)
by Maor Ivgi, Oliver Hinder and Yair Carmon.

## Installation
To install the package, simply run `pip install dog-optimizer`.

## Usage
DoG (or LDoG) are implemented using the standard pytorch optimizer interface. After installing the pacakge with `pip install dog-optimizer`,
All you need to do is replace the line that creates your optimizer with 
```python
from dog_optimizer import DoG
optimizer = DoG(optimizer args)
```
for DoG, or
```python
from dog_optimizer import LDoG
optimizer = LDoG(optimizer args)
```
for LDoG, 
where `optimizer args` follows the standard pytorch optimizer syntex. 
To see the list of all available parameters, run `help(DoG)` or `help(LDoG)`.


Using the polynomial decay averager is also easy. Simply create it with 
```python
from dog_optimizer import PolynomialDecayAverager
averager = PolynomialDecayAverager(model)
```
then, after each `optimizer.step()`, call `averager.step()` as well.
You can then get both the current model and the averaged model with `averager.base_model` and `averager.averaged_model` respectively.

An example of how to use the above to train a simple CNN on MNIST can be found in `examples/mnist.py` 
(based on this [pytorch example](https://github.com/pytorch/examples/blob/main/mnist/main.py)).

### Choosing `reps_rel`
DoG is parameter-free by design, so there is no need to tune a learning rate parameter. 
However, as discussed in the paper, DoG has an initial step movement parameter 
$r_{\epsilon}$ that must be small enough to avoid destructively updates that cause divergence, 
but an extremely small value of $r_{\epsilon}$ would slow down training. 
We recommend choosing $r_{\epsilon}$ relative to the norm of the initial weights $x_0$. In particular, we set 
$r_{\epsilon}$ to be `reps_rel` time $1+\|x_0\|$, where `reps_rel` is a configurable parameter of the optimizer. The default value 
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
@article{ivgi2023dog,
  title={{D}o{G} is {SGD}'s Best Friend: A Parameter-Free Dynamic Step Size Schedule}, 
  author={Maor Ivgi and Oliver Hinder and Yair Carmon}, 
  journal={arXiv:2302.12022}, 
  year={2023},
}  
```