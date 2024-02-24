clc;
clear;
close all;


numClasses = 2;

net = resnet50();
lgraph = layerGraph(net);


layersToRemove = {
    'fc1000'
    'fc1000_softmax'
    'ClassificationLayer_fc1000'
    };
lgraph = removeLayers(lgraph, layersToRemove);


newLayers = [
    fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
    softmaxLayer('Name', 'rcnnSoftmax')
    classificationLayer('Name', 'rcnnClassification')
    ];
lgraph = addLayers(lgraph, newLayers);


lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');


load gTruth;
layers = resnet50('Weights', 'none')


dataFolder = '/Users/giov17/Desktop/png_resized';
exts = {'.png'}

imds = imageDatastore(dataFolder, 'FileExtensions', exts, 'IncludeSubfolders', true, 'LabelSource','foldernames');

%imageAugmenter = imageDataAugmenter('RandRotation',[1,2]);
%augimds = augmentedImageDatastore([28 28],imds, 'DataAugmentation',imageAugmenter);

%augimds = shuffle(augimds);



summary(trainingData)
options = trainingOptions('sgdm', ...
  'MiniBatchSize', 128, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 5);
[detector,info] = trainRCNNObjectDetector(trainingData, layers, options, 'NegativeOverlapRange', [0 0.3]);



% https://it.mathworks.com/help/deeplearning/ug/transfer-learning-with-deep-network-designer.html




% PyTorch models

%model_yolov5s = "./models/yolov5s.pt"

%net = importNetworkFromPyTorch(model_yolov5s)

