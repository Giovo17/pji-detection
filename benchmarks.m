clc;
clear;
close all;


% Resnet50 finetuning

dataFolder = '~/Desktop/rescale_224x224/';
exts = {'.png'}
imds = imageDatastore(dataFolder, 'FileExtensions', exts, 'IncludeSubfolders', true, 'LabelSource','foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds,0.7,'randomized');


numClasses = numel(categories(imdsTrain.Labels));


net = resnet50;
%deepNetworkDesigner(net)
inputSize = net.Layers(1).InputSize
lgraph = layerGraph(net); 


% Specify new layers for classification based on number of unique classes
newfc = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc1000',newfc);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);


% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);


augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);



% Training hyperparameters
opts = trainingOptions('adam', ...
                       'MiniBatchSize', 32, ...
                       'MaxEpochs', 5, ...
                       'InitialLearnRate',1e-4, ...
                       'Shuffle','every-epoch', ...
                       'ValidationData', augimdsValidation, ...
                       'ValidationFrequency',3, ...
                       'Plots', 'training-progress', ...
                       'Verbose', false);


% Model finetuning
[net, info] = trainNetwork(augimdsTrain, lgraph, opts);




