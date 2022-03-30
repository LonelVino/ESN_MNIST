# ESN classifier with Power-efficient Photonic Systems

![](https://img.shields.io/badge/Python-v3.8-orange)![Numpy](https://camo.githubusercontent.com/a1c5e9056e3be1e1058d8517b025af60f61f75395a78245776db71a7703aff9c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6e756d70792d2532333031333234332e7376673f7374796c653d666f722d7468652d6261646765266c6f676f3d6e756d7079266c6f676f436f6c6f723d7768697465) ![Kaggle](https://camo.githubusercontent.com/9ec86d990dbc1eeab84728664a57e0fbac84211d2c6cda5a2cf10288e37e7306/68747470733a2f2f726f61642d746f2d6b6167676c652d6772616e646d61737465722e76657263656c2e6170702f6170692f73696d706c652f737562696e69756d)



## 1. Prerequisites

#### 1.1 Dependencies

 * [Numpy](http://www.numpy.org/)
 * [Matlplotlib](http://matplotlib.org/) (for graphing)
 * [tqdm](https://github.com/tqdm/tqdm)
 * [sklearn](https://scikit-learn.org/stable/)
 * [bayes_opt](https://github.com/fmfn/BayesianOptimization)

```bash
git clone git@github.com:LonelVino/ESN_MNIST.git
cd ESN_MNIST
pip install -r requirements.txt
```

#### 1.2 Dataset

##### MNIST

- https://www.kaggle.com/oddrationale/mnist-in-csv
- http://yann.lecun.com/exdb/mnist/

#### 1.3 Test Case

[ESN MNIST (Jupyter notebook)](https://github.com/LonelVino/ESN_MNIST/blob/main/%5BESN%5DNiyamas_MNIST.ipynb)

If there are errors while running this notebook locally,  you can also refer to the notebook mounted on kaggle: [[ESN] MNIST Classification](https://www.kaggle.com/lonelvino/esn-mnist-classification-niyamas).

## 2. Introduction

[Echo State Network](http://www.scholarpedia.org/article/Echo_state_network) is easy-to-train recurrent neural networks, a variant of [Reservoir Computing](https://en.wikipedia.org/wiki/Reservoir_computing). In some sense, which aims to simplify training process, reduce computation and overcome fading memory problem of RNN. In this project, the system model is designed as follow:

### (1) System Equation

##### **(1.1) Hidden Layer**

<img src="https://latex.codecogs.com/svg.image?\mathbf{x}(k&plus;1)=f(\mathbf{W}^{res}\mathbf{x}(k)&plus;\mathbf{W}^{in}\mathbf{u}(k&plus;1)&plus;\mathbf{W}^{fb}\mathbf{y}(k))" title="\mathbf{x}(k+1)=f(\mathbf{W}^{res}\mathbf{x}(k)+\mathbf{W}^{in}\mathbf{u}(k+1)+\mathbf{W}^{fb}\mathbf{y}(k))" />

where **x**(*k*) is the *N*-dimensional reservoir state, *f* is a sigmoid function (usually the logistic sigmoid or the tanh function), <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{res}" title="\mathbf{W}^{res}" /> is the *N*×*N* reservoir weight matrix, <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{in}" title="\mathbf{W}^{in}" /> is the *N*×*P* input weight matrix, **u**(*k*) is the *P* dimensional input signal, <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{fb}" title="\mathbf{W}^{fb}" /> is the *N*×*M* output feedback matrix, and **y**(*k*) is the *M*-dimensional output signal.

> - <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{res}" title="\mathbf{W}^{res}" />, <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{in}" title="\mathbf{W}^{in}" />, <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{fb}" title="\mathbf{W}^{fb}" /> are generated randomly and fixed. 
> - In this task, the output feedback isn’t required, thus <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{fb}" title="\mathbf{W}^{fb}" /> is nulled. 

##### **(1.2) Output (Readout) Layer**

The extended system state <u>**z**(*k*)=[**x**(*k*);**u**(*k*)]</u> at time *k* is the concatenation of the reservoir and input states. The output is obtained from the extended system state by:

<img src="https://latex.codecogs.com/svg.image?y(n)=g_{out}(\mathbf{W}^{out}z(n))" title="y(n)=g_{out}(\mathbf{W}^{out}z(n))" />

where <img src="https://latex.codecogs.com/svg.image?g_{out}=[g_{out}^1,...,g_{out}^M]" title="g_{out}=[g_{out}^1,...,g_{out}^M]" /> is the output activation functions (typically linear or a sigmoid) and <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{out}" title="\mathbf{W}^{out}" /> is a *M*×(*P*+*N*)-dimensional matrix of output weights.

### (2) Learning Equation

The desired output weights <img src="https://latex.codecogs.com/svg.image?\mathbf{W}^{out}" title="\mathbf{W}^{out}" /> are the linear regression weights of the desired output **d** *(k)* on the harvested extended states **z** *(k)*, which is called as readout weights, the only learnable parameters of the reservoir computer architecture in this project: 

<img src="https://latex.codecogs.com/svg.image?\mathbf{W^{out}}&space;=&space;||\mathbf{Y}_t-\mathbf{W}^{out}\mathbf{X}||^2&plus;\lambda&space;||\mathbf{W^{out}}||^2" title="\mathbf{W^{out}} = ||\mathbf{Y}_t-\mathbf{W}^{out}\mathbf{X}||^2+\lambda ||\mathbf{W^{out}}||^2" />

which is an offline algorithm. Here, <img src="https://latex.codecogs.com/svg.image?\mathbf{Y}_t\in\mathbb{R}^{m\times&space;k_{tr}}" title="\mathbf{Y}_t\in\mathbb{R}^{m\times k_{tr}}" /> is the target matrix containing the <img src="https://latex.codecogs.com/svg.image?k_{tr}" title="k_{tr}" /> targets values for the *m* outputs of the reservoir computer, when inputs vectors are fed to the reservoir during <img src="https://latex.codecogs.com/svg.image?k_{tr}" title="k_{tr}" /> time steps. 

![ESN_overview](./assets/ESN_overview.png)



## 3.Reference

[1] “(PDF) An overview of reservoir computing: Theory, applications and implementations.” https://www.researchgate.net/publication/221166209_An_overview_of_reservoir_computing_Theory_applications_and_implementations (accessed Feb. 02, 2022).

[2] R. J. Williams and D. Zipser, “A Learning Algorithm for Continually Running Fully Recurrent Neural Networks,” *Neural Computation*, vol. 1, no. 2, pp. 270–280, Jun. 1989, doi: [10.1162/neco.1989.1.2.270](https://doi.org/10.1162/neco.1989.1.2.270).

[3] A. Lamb, A. Goyal, Y. Zhang, S. Zhang, A. Courville, and Y. Bengio, “Professor Forcing: A New Algorithm for Training Recurrent Networks,” *arXiv:1610.09038 [cs, stat]*, Oct. 2016, Accessed: Feb. 04, 2022. [Online]. Available: http://arxiv.org/abs/1610.09038

[4] H. Jaeger, “The" echo state" approach to analysing and training recurrent neural networks-with an erratum note’,” *Bonn, Germany: German National Research Center for Information Technology GMD Technical Report*, vol. 148, Jan. 2001.