clc;
clear;
close all;




function [net, info] = resnet50_finetuning(train_data, opts)


    numClasses = numel(categories(train_data.Labels));


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


    % Model finetuning
    [net, info] = trainNetwork(train_data, lgraph, opts);

end


function [net, info] = squeeznet_finetuning(train_data, opts)


    numClasses = numel(categories(train_data.Labels));


    net = squeeznet;
    %deepNetworkDesigner(net)
    inputSize = net.Layers(1).InputSize
    lgraph = layerGraph(net); 


    % Specify new layers for classification based on number of unique classes
    new_convolution2dLayer =  convolution2dLayer([1 1],2,"new_conv10")
    lgraph = replaceLayer(lgraph,"conv10",new_convolution2dLayer);

    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);


    % Model finetuning
    [net, info] = trainNetwork(train_data, lgraph, opts);

end


% Data import
dataFolder = '~/Desktop/rescale_224x224/';
exts = {'.png'}
imds = imageDatastore(dataFolder, 'FileExtensions', exts, 'IncludeSubfolders', true, 'LabelSource','foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds,0.7,'randomized');


% Data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation',[0 360], ...
    'RandXTranslation',[-30 30], ...
    'RandYTranslation',[-30 30]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);


% Training hyperparameters
opts = trainingOptions('adam', ...
                       'MiniBatchSize', 32, ...
                       'MaxEpochs', 1, ...
                       'InitialLearnRate', 1e-4, ...
                       'L2Regularization', 0.0005, ...
                       'Shuffle','every-epoch', ...
                       'ValidationData', augimdsValidation, ...
                       'ValidationFrequency', 150, ...
                       'Plots', 'training-progress', ...
                       'Verbose', false);


%resnet50_finetuning(augimdsTrain, opts)
squeeznet_finetuning(augimdsTrain, opts)


