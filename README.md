# Hand Written Number Recognition with ANN

_Will Koenig_


## Description:
In this project I will attempt to train an Artificial Neural Network (ANN) on a data set containing handwritten digits 0-9 and test it's ability to accurately predict what digit it is given.  Since this is one of the first times that I am working with ANNs I am going to do a lot of work 'the long way'; I am not really going to take advantage of some very powerfuls for setting up and training neural networks becaues I think it is important.


## Set Up and Run Instructions:
In order to run and edit this project you are going to need to have a few languages and libraries installed.  You are also going to need to get the data sets that we are using:

### Environment Requirements:
Languages: `python`, Libraries: `jupyter, numpy, struct, os`

I am going to assume that you have python3 and pip3 configured on your machine, if not I would recommend following the guide [here]().  Once you have them installed you can go ahead and open a terminal to install your dependencies with the following command:

```python
pip3 install jupyter numpy
```

Once you have jupyter installed, from the directory that is housing your project run the following command to open the project in the jupyter notebook editor:

```pythonn
jupyter notebook
```

### Data:
We will use the following 4 data files to train our ANN, they are available [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).  After following this you are going to want to download the dataset, unzip it, and extract the following 4 files:
* t10k-images-idx3-ubyte
* t10k-labels-idx1-ubyte
* train-images-idx3-ubyte
* train-labels-idx1-ubyte

Move these data files into the same directory as your fork of this project.  Then we can use the following function to harvest the data
```python
def read_gzip_file(filename):
    with open(filename, 'rb') as f:
      zero, data_type, dims = struct.unpack('>HBB', f.read(4))
      shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
      return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
```

## A Hard-Coded Neural Network?
You might have noticed above that I did not important anything to construct the neural network with (tensorflow, keras, pytorch, etc...).  For this project I have hard-coded the structure of the neural network, the activation functions, as well as the feed-forward and back-propagation methods.  Let's take a look at what each of these process looked like and how 

### Network Architecture
The architecture of a neural network is fascinating and you don't really have the chance to see it when you are using libraries that handle most of the construction for you.  We define our architecture with the following:


```python
input_size, output_size = 784, 10
hidden1_size, hidden2_size = 128, 64
```

This defines a neural networking with the following structure:

<img width="572" alt="NetworkStructure" src="https://github.com/willk0814/numberRecognition/assets/36479286/20c6692b-a4ee-47d9-861c-789a0bb07651">

<!-- 
### Activation Function
Each node of the neural network needs its an activation function and all nodes in each layer will have the same activation function.

### Feed-Forward

### Back-Propagation

### Training with Mini-Batch Stochastic Gradient Descent -->
