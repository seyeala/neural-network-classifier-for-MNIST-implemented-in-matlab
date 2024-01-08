MNIST Neural Network Classifier using BASIC functions in MATLAB

This repo contains a NN implementation to classify handwritten digits from the MNIST dataset. It achieves approximately 98% accuracy on the test dataset. It employs basic features such as a self-implemented feedforward architecture and back-propagation.

Other Features

Four layers of NN: 1 input reshaped layer, 2 fully connected dense layers with biases, and 1 Output Softmax layer. The hidden layers use the tanh activation function.

Both tanh and softmax are stabilized to prevent issues such as vanishing gradients.

Cross-Entropy is used as the loss function, and stochastic gradient descent is employed.

Ten random samples from the test dataset is used for visualization. Also the accuracy on test holdout is printed in the end.

Validation loss is utilized for tracking and model selection based on an independent validation loss sourced from the training dataset.
In the end of the training loop the weights with the lowest validation loss are chosen for testing.

Ability to include or exclude biases for the hidden layer is implemented.

Based on eq 9 in Learning representations by back-propagating errors by Rumelheart, SGD algorithm with an additional velocity term is implemented for training. The extra term can be deactivated and the training reduces to SGD with the batch size of 1.



How to Run

1-First, download the MNIST dataset and unzip it. You will get several idx files to be used. They are also provided in this package.
2-Clone this repository.
3-Ensure that you have MATLAB installed.
4-Set the paths in the .m file (mnistclasifier.m) to the MNIST dataset files (IDX files) as follows:


        % Paths to MNIST .idx files here:
        trainimagespath = 'your path to train-images.idx3-ubyte';
        trainlabpath = 'your path to train-labels.idx1-ubyte';
        testimagpath = 'your path to t10k-images.idx3-ubyte';
        testlabpath = 'your path to t10k-labels.idx1-ubyte';



5-Choose the number of epochs for training and the learning rate in the code. If desired, you can specify a different number of hidden layers:


        outputClassSize = 10;
        inputSize = 784;
        hiddenDim_for_Layer2 = 256;
        hiddenDim_for_Layer3 = 128;

        % training hyperparameters
        learningRate = 0.005;
        numEpochs_train = 10;

6- If the biases are needed you can activate them by setting bias_activation as zero.
If set 1 all hidden layers will have biases that will be trained
	%activate biases 0 no biases 1 biases for hidden layers
	bias_activation=0



7-In the Rumelheart paper "Learning representations by back-propagating errors."  equation 9 there is an additional term for the regular SGD that is called the velocity.
If not desired set expodec as zero and the training becomes SGD with batch size of 1

	%exponential decay constant for Eq 9 in the reference paper
	expodec=0.5*learningR;

8-Run the .m file in MATLAB. As it progresses, you'll see the epoch number, validation, and training losses.
At the end, it will display ten randomly chosen images with their predicted and true labels.

