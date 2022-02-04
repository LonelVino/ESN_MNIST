# ESN classifier with Power-efficient Photonic Systems

![](https://img.shields.io/badge/Python-v3.8-orange)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## 1. Introduction

[Echo State Network](http://www.scholarpedia.org/article/Echo_state_network) is easy-to-train recurrent neural networks, a variant of [Reservoir Computing](https://en.wikipedia.org/wiki/Reservoir_computing). In some sense, which aims to simplify training process, reduce computation and overcome fading memory problem of RNN. In this project, the system model is designed as follow:

### (1) System Equation

##### **(1.1) Hidden Layer**

$$
\mathbf{x}(k+1)=f(\mathbf{W}^{res}\mathbf{x}(k)+\mathbf{W}^{in}\mathbf{u}(k+1)+\mathbf{W}^{fb}\mathbf{y}(k))
$$
where **x**(*k*) is the *N*-dimensional reservoir state, *f* is a sigmoid function (usually the logistic sigmoid or the $\tanh$ function), **W**$^{res}$ is the *N*×*N* reservoir weight matrix, **W**$^{in}$ is the *N*×*P* input weight matrix, **u**(*k*) is the *P* dimensional input signal, **W**$^{fb}$ is the *N*×*M* output feedback matrix, and **y**(*k*) is the *M*-dimensional output signal.

> - **W**$^{res}$, **W**$^{in}$, **W**$^{fb}$ are generated randomly and fixed. 
> - In this task, the output feedback isn’t required, thus **W**$^{fb}$ is nulled. 

##### **(1.2) Output (Readout) Layer**

The extended system state <u>**z**(*k*)=[**x**(*k*);**u**(*k*)]</u> at time *k* is the concatenation of the reservoir and input states. The output is obtained from the extended system state by:
$$
y(n)=g_{out}(\mathbf{W}^{out}z(n))
$$
where $g_{out}=[g_{out}^1,...,g_{out}^M]$ is the output activation functions (typically linear or a sigmoid) and **W**$^{out}$ is a *M*×(*P*+*N*)-dimensional matrix of output weights.

### (2) Learning Equation

The desired output weights **W**$^{out}$ are the linear regression weights of the desired output **d**$(k)$ on the harvested extended states **z**$(k)$, which is called as readout weights, the only learnable parameters of the reservoir computer architecture in this project: 
$$
\mathbf{W^{out}} = \norm{\mathbf{Y}_t-\mathbf{W}^{out}\mathbf{X}}^2+\lambda\norm{\mathbf{W^{out}}}^2
$$
which is an offline algorithm. Here, $\mathbf{Y}_t\in\mathbb{R}^{m\times k_{tr}}$ is the target matrix containing the $k_{tr]}$ targets values for the $m$ outputs of the reservoir computer, when inputs vbectors are fed to the reservoir during $k_{tr}$ time steps. 

![ESN_overview](./assets/ESN_overview.png)

## 2. Prerequisites

#### 2.1 Dependencies

 * [Numpy](http://www.numpy.org/)
 * [Matlplotlib](http://matplotlib.org/) (for graphing)
 * [tqdm](https://github.com/tqdm/tqdm)
 * [pytorch](https://pytorch.org/)

```bash
git clone git@github.com:LonelVino/ESN_MNIST.git
cd ESN_MNIST
pip install -r requirements.txt
```

#### 2.2 Dataset



## 3.Reference

[1] “(PDF) An overview of reservoir computing: Theory, applications and implementations.” https://www.researchgate.net/publication/221166209_An_overview_of_reservoir_computing_Theory_applications_and_implementations (accessed Feb. 02, 2022).

[2] R. J. Williams and D. Zipser, “A Learning Algorithm for Continually Running Fully Recurrent Neural Networks,” *Neural Computation*, vol. 1, no. 2, pp. 270–280, Jun. 1989, doi: [10.1162/neco.1989.1.2.270](https://doi.org/10.1162/neco.1989.1.2.270).

[3] A. Lamb, A. Goyal, Y. Zhang, S. Zhang, A. Courville, and Y. Bengio, “Professor Forcing: A New Algorithm for Training Recurrent Networks,” *arXiv:1610.09038 [cs, stat]*, Oct. 2016, Accessed: Feb. 04, 2022. [Online]. Available: http://arxiv.org/abs/1610.09038

[4] H. Jaeger, “The" echo state" approach to analysing and training recurrent neural networks-with an erratum note’,” *Bonn, Germany: German National Research Center for Information Technology GMD Technical Report*, vol. 148, Jan. 2001.