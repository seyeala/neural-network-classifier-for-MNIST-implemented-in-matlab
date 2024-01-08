


% read/normalize the training and test data
[trainingimas, traininglabls] = rMNIST(trainimagespath,trainlabpath);
trainingimas = trainingimas / 255;
[testmages, testabels] = rMNIST(testimagpath,  testalbblPath);
testmages = testmages / 255;




% Split validation data ( not asked in the HW did for experimenting the termination criteria)
numvalSamples = 10000;
validationInd = randperm(size(trainingimas,   3),numvalSamples);
validationImages = trainingimas(:,  :,validationInd);
validationLabels = traininglabls(validationInd);
trainingimas(:,  :,validationInd) = []  ;
traininglabls(validationInd) = []  ;

% Paths to MNISTMNIST iodx files here:
trainimagespath = 'change to path to train-images.idx3-ubyte';
trainlabpath = 'change to path to train-labels.idx1-ubyte';
testimagpath = 'change to path to t10k-images.idx3-ubyte';
testalbblPath = 'change to path to t10k-labels.idx1-ubyte';



% Neura Nets Architecture (1 reshape, 2 Dense with tanh activation function
%3 Dense with activation function, 4 Softmax
outpuClastSize = 10;
inputSizes = 784;
hiddenDim_for_Layer2 =  256;
hiddenDim_for_Layer3 = 128;

% training hyperparameters
learningR = 0.005;
numEpochs_train = 2;

%exponential decay constant for Eq 9 in the reference paper
expodec=0.5*learningR;
%aactivate biases 0 no biases 1 biases for hidden layers
bias_activation=1

% init weights and bias for Layer 2,3
w2 = (rand(hiddenDim_for_Layer2, inputSizes)-0.5)*0.1;
w3 = (rand(hiddenDim_for_Layer3,   hiddenDim_for_Layer2) -0.5)*0.1 ;
w4 = (rand (outpuClastSize,  hiddenDim_for_Layer3)-0.5)*0.1;

% it seems that the problam asks for no biases therefore here they are commented oout
b2 = (rand (hiddenDim_for_Layer2, 1)-0.5)*0.1*bias_activation;
b3 =  (rand(hiddenDim_for_Layer3,1)-0.5)*0.1*bias_activation;
b4 =  (rand(outpuClastSize, 1)-0.5)*0.1*bias_activation;

% We set the biases zero and do not update them as the question did not ask for biases however
%that can be reactivated by replace 0s by 0.01 updating the training loop




% tryraining Loop,
bestValinLoss =inf;
besWeigh = struct('w2',  [], 'b2'  , [],'w3',  [], 'b3',  [], 'w4',   [],'b4',  []);
validationlosshs = zeros(1,numEpochs_train);


delta_w2_old=0;
delta_w3_old=0;
delta_w4_old=0;
delta_b2_old=0;
delta_b3_old=0;
delta_b4_old=0;





for epoch = 1: numEpochs_train
    totlLoss =0;
    expdecayfact=expodec*exp(-epoch/5)
    for i = 1 :size(trainingimas, 3) %  third dimension is number of samples
        x = trainingimas(: ,:,i);
        y = oneHotEnc(traininglabls(i), outpuClastSize); % Convert labels to to 1x10 vector

% forward pass
% activation functions to be used for Backpropagation
        [nnoutput, otherweightsforbackpro] = NeuralNet(x,  w2,  w3,w4, b2,  b3, b4);

% loss is calculated
        loss =  crssEntropyLoss(y, nnoutput);
        totlLoss = totlLoss+ loss;

% gradients of the weights are calculated
        [gradw2,gradw3, gradw4,gradb2,gradb3, gradb4] = backprop(y, nnoutput, otherweightsforbackpro  , w2, w3,w4);

% Update the weights

        w2 = w2 - learningR * gradw2 + expdecayfact*delta_w2_old;
        w3 = w3 - learningR * gradw3 + expdecayfact*delta_w3_old;
        w4 = w4 - learningR * gradw4 + expdecayfact*delta_w4_old;



        b2 = b2 - learningR * gradb2 + expdecayfact*delta_b2_old*bias_activation;
        b3 = b3 - learningR * gradb3 + expdecayfact*delta_b3_old*bias_activation;
        b4 = b4 - learningR * gradb4 + expdecayfact*delta_b4_old*bias_activation;

        delta_w2_old= - learningR * gradw2 + expdecayfact*delta_w2_old;
        delta_w3_old= -  learningR * gradw3 + expdecayfact*delta_w3_old;
        delta_w4_old= - learningR * gradw4 + expdecayfact*delta_w4_old;
        delta_b2_old= - learningR * gradb2 + expdecayfact*delta_b2_old*bias_activation;
        delta_b3_old= - learningR * gradb3 + expdecayfact*delta_b3_old*bias_activation;
        delta_b4_old= - learningR * gradb4 + expdecayfact*delta_b4_old*bias_activation;
    end
% calculate average loss
    avgLoss = totlLoss / size(trainingimas, 3);

% calculaye validation loss
    totalValidaLoss = 0;
    for i = 1: size( validationImages, 3)
        x = validationImages (:, :,i);
        y = oneHotEnc(validationLabels(i),  outpuClastSize) ;

        nnoutput = NeuralNet(x,w2,  w3, w4, b2,  b3,  b4) ;
        loss = crssEntropyLoss(y, nnoutput);
        totalValidaLoss = totalValidaLoss + loss ;
    end

    avgValid = totalValidaLoss / size (validationImages,3) ;
    validationlosshs(epoch) = avgValid;
    epochnumber=epoch
    TrainLoss=avgLoss
    Validloss=avgValid

% if this epoch h

    if avgValid < bestValinLoss
        besValinLoss = avgValid;
        besWeigh.w2 = w2;
        besWeigh.w3 = w3;
        besWeigh.w4 = w4;

        besWeigh.b2 = b2;
        besWeigh.b3 = b3;
        besWeigh.b4 = b4;
    end
end

%  set the weights to the best ones
w2 = besWeigh.w2;
w3 = besWeigh.w3;
w4 = besWeigh.w4;

b2 = besWeigh.b2;
b3 = besWeigh.b3;
b4 = besWeigh.b4;


%  10 random  pictures and test them
numTestImages = size(testImages, 3);
randominc = randperm(numTestImages, 10);

figure;

for i = 1:10
    index = randominc( i);
    pic = testmages(: ,:, index);

    trueLabel = testabels( index);

    nnoutput = NeuralNet(pic,  w2,  w3,  w4, b2,  b3, b4);
    [ll, predictedLabel] = max( nnoutput);
    predictedLabel =  predictedLabel - 1;

    % show the image
    subplot(2,5, i);
    imshow( pic, []);
    title(sprintf( 'predict: %d\ntruth: %d', predictedLabel , trueLabel) );
end


correctppedictions= 0;

for i= 1:size(testmages, 3)

    image= testmages(:,:,i);
    truelabel = testabels(i);

% Get NN predictiopn
    nnoutput= NeuralNet(image, w2, w3,  w4, b2,  b3, b4);

% Get the max possinlity
    [ll, predictedlabel]= max( nnoutput);
    predictedlabel =  predictedlabel - 1; % Adjust for 0-based indexing

% Is predict
    if predictedlabel == truelabel
        correctppedictions = correctppedictions +1;
    end
end


accuracy = (correctppedictions / size( testmages, 3))* 100;
fprintf('Acc_test_data: %.2f%%\n', accuracy);


%loss function

function loss = crssEntropyLoss(y1, y2)
    epsi = 0.0000000000000001; % prevent log(0)
    y2=max(y2,epsi);
    loss = -1*sum(y1 .* log(y2));
end

%one Hot shot encoder

function y_oneh =oneHotEnc(label, numClasses)
    y_oneh = zeros(numClasses, 1);
    y_oneh(label+ 1) = 1; % Assuming label is 0-based. If it's 1-based, remove the + 1
end

%Neural Nets
function [nn, otherweightsforbackpro] = NeuralNet(x,   w2, w3,  w4, b2,  b3, b4)
    [x, z1] = Layer1(x);
    [x, z2] = Layer2(x, w2, b2);
    [x, z3] = Layer3(x, w3, b3);
    [x, z4] = Layer4(x, w4, b4);

    nn = x;
    otherweightsforbackpro = struct('z1', z1,'z2', z2, 'z3',z3,  'z4', z4, 'act2', my_tanh(z2),   'act3'  , my_tanh(z3));
end

%Layer 1-reshape
function [x_ou, z] = Layer1(x)
    x_ou = reshape(x, [], 1);
    z = x_ou;
end


%Layer 2-Dense with tanhH activation function
function [x_ou, z] = Layer2(x,w2, b2)
    z = w2 * x + b2;
    x_ou = my_tanh(z);

end


%Layer 3-Dense with tanhH activation function
function [x_ou, z] = Layer3(x, w3,b3)
    z = w3 * x + b3;
    x_ou = my_tanh(z);
end

%Layer 4-Softmax

function [x_ou, z] = Layer4(x, w4, b4)
    z = w4 * x + b4;
    x_ou = my_softmax(z);
end

%activ. function Tanh
function y = my_tanh(x)
% clip the vlues  to stablize the function
    MAXi = 10;
    MINi = -10;
    x(x > MAXi) = MAXi;
    x(x < MINi) = MINi;
    y = tanh(x);
end

%Back propagation function
function [gradw2, gradw3, gradw4, gradb2, gradb3, gradb4] = backprop(y, nnoutput, otherweightsforbackpro, w2, w3, w4)

    dlossbydy = nnoutput - y;

% layer 4: Softmax followed by a dense
    gradz4 = dlossbydy ;
    gradw4 = gradz4 * otherweightsforbackpro.act3' ;
    gradb4 = gradz4;

% layer 3: tanh activation  and a fully connected layer
    dthbydz3 = 1 - otherweightsforbackpro.act3.^2;
    gradz3 = (w4' * gradz4) .* dthbydz3;
    gradw3 = gradz3 * otherweightsforbackpro.act2';
    gradb3 = gradz3;

% layer2: tanh activation  and a fully connected layer
    dthbydz2 = 1 - otherweightsforbackpro.act2.^2;
    gradz2 = (w3' * gradz3) .* dthbydz2;
    gradw2 = gradz2 * otherweightsforbackpro.z1';
    gradb2 = gradz2;
end

%read mnist file, refrence on how to read the file is found here:
%http://yann.lecun.com/exdb/mnist/
function [pics, label] = rMNIST(imgFile, lblFile)
    % read pictures
    file = fopen(imgFile, 'r', 'b');

    magNum = fread(file,1, 'int32');

    nuImgs = fread(file,1, 'int32');

    nuRows = fread(file, 1, 'int32');

    nuCols = fread(file, 1,'int32');

    pics = fread(file,inf, 'unsigned char');

    pics = reshape(pics, nuCols, nuRows, nuImgs);

    pics = permute(pics,[2 1 3]);
    fclose(file);

% read all labels
    file = fopen(lblFile,  'r', 'b');

    magicNum = fread(file,1, 'int32');

    numLabels = fread(file,  1,'int32');

    label= fread(file, inf,'unsigned char');

    fclose(file);
end

%softmax activation function that is stablized

function p = my_softmax(x)

    % the maximum value is deducted for stabilization
    x = x- max(x);
    expo_x =exp(x);
    p = expo_x / sum(expo_x);
end
