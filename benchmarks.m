clc;
clear;
close all;


% Data import
dataFolder = '~/Desktop/rescale_224x224/'; % googlenet, resnet18, resnet50
%dataFolder = '~/Desktop/rescale_227x227/'; % squeezenet
%dataFolder = '~/Desktop/rescale_256x256/'; % darknet19
exts = {'.png'}
imds = imageDatastore(dataFolder, 'FileExtensions', exts, 'IncludeSubfolders', true, 'LabelSource','foldernames');
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds,0.7,0.15,'randomized');


% Data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation',[0 360], ...
    'RandXTranslation',[-30 30], ...
    'RandYTranslation',[-30 30]);


%[info, YPred, scores] = squeezenet_finetuning(imdsTrain, imdsValidation, imdsTest, imageAugmenter)
%[info, YPred, scores] = googlenet_finetuning(imdsTrain, imdsValidation, imdsTest, imageAugmenter)
[info, YPred, scores] = resnet18_finetuning(imdsTrain, imdsValidation, imdsTest, imageAugmenter)
%[info, YPred, scores] = resnet50_finetuning(imdsTrain, imdsValidation, imdsTest, imageAugmenter)
%[info, YPred, scores] = darknet19_finetuning(imdsTrain, imdsValidation, imdsTest, imageAugmenter)




function [info, YPred, scores] = squeezenet_finetuning(train_data, val_data, test_data, imageAugmenter)

    numClasses = numel(categories(train_data.Labels));

    net = squeezenet;
    inputSize = net.Layers(1).InputSize
    lgraph = layerGraph(net); 


    % Data Augmentation
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_data,'DataAugmentation',imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),val_data);


    % Specify new layers for classification based on number of unique classes
    new_convolution2dLayer =  convolution2dLayer(1,2,'Name',"new_conv10",'Stride',1,'Padding',0);
    lgraph = replaceLayer(lgraph,"conv10",new_convolution2dLayer);

    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);


    % Training hyperparameters
    opts = trainingOptions('adam', ...
                           'MiniBatchSize', 32, ...
                           'MaxEpochs', 2, ...
                           'InitialLearnRate', 1e-4, ...
                           'L2Regularization', 5e-4, ...
                           'Shuffle','every-epoch', ...
                           'ValidationData', augimdsValidation, ...
                           'ValidationFrequency', 100, ...
                           'Plots', 'training-progress', ...
                           'Verbose', false);


    % Model finetuning
    [net, info] = trainNetwork(train_data, lgraph, opts);
    exportONNXNetwork(net, "squeezenet.onnx") % Export network


    % Model test and metrics extraction
    [YPred,scores] = classify(net,test_data);


    % Export results
    writematrix(YPred,'squeezenet_ypred.csv')
    writematrix(scores,'squeezenet_scores.csv')
    writematrix(test_data.Labels,'squeezenet_ytrue.csv')

end



function [info, YPred, scores] = googlenet_finetuning(train_data, val_data, test_data, imageAugmenter)

    numClasses = numel(categories(train_data.Labels));

    net = googlenet;
    inputSize = net.Layers(1).InputSize
    lgraph = layerGraph(net); 


    % Data Augmentation
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_data,'DataAugmentation',imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),val_data);


    % Specify new layers for classification based on number of unique classes
    newfc = fullyConnectedLayer(2,"Name","loss3-classifier","BiasLearnRateFactor",2);
    lgraph = replaceLayer(lgraph,"loss3-classifier",newfc);

    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'output',newClassLayer);
    

    % Training hyperparameters
    opts = trainingOptions('adam', ...
                           'MiniBatchSize', 32, ...
                           'MaxEpochs', 2, ...
                           'InitialLearnRate', 1e-4, ...
                           'L2Regularization', 5e-4, ...
                           'Shuffle','every-epoch', ...
                           'ValidationData', augimdsValidation, ...
                           'ValidationFrequency', 100, ...
                           'Plots', 'training-progress', ...
                           'Verbose', false);


    % Model finetuning
    [net, info] = trainNetwork(train_data, lgraph, opts);
    exportONNXNetwork(net, "googlenet.onnx")


    % Model test and metrics extraction
    [YPred,scores] = classify(net,test_data);


    % Export results
    writematrix(YPred,'googlenet_ypred.csv')
    writematrix(scores,'googlenet_scores.csv')
    writematrix(test_data.Labels,'googlenet_ytrue.csv')

end



function [info, YPred, scores] = resnet18_finetuning(train_data, val_data, test_data, imageAugmenter)

    numClasses = numel(categories(train_data.Labels));

    net = resnet18;
    inputSize = net.Layers(1).InputSize
    lgraph = layerGraph(net); 


    % Data Augmentation
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_data,'DataAugmentation',imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),val_data);


    % Specify new layers for classification based on number of unique classes
    newfc = fullyConnectedLayer(numClasses, 'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'fc1000',newfc);

    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);
    

    % Training hyperparameters
    opts = trainingOptions('adam', ...
                           'MiniBatchSize', 32, ...
                           'MaxEpochs', 1, ...
                           'InitialLearnRate', 1e-4, ...
                           'L2Regularization', 5e-4, ...
                           'Shuffle','every-epoch', ...
                           'ValidationData', augimdsValidation, ...
                           'ValidationFrequency', 100, ...
                           'Plots', 'training-progress', ...
                           'Verbose', false);


    % Model finetuning
    [net, info] = trainNetwork(train_data, lgraph, opts);
    exportONNXNetwork(net, "resnet18.onnx")


    % Model test and metrics extraction
    [YPred,scores] = classify(net,test_data);


    % Export results
    writematrix(YPred,'resnet18_ypred.csv')
    writematrix(scores,'resnet18_scores.csv')
    writematrix(test_data.Labels,'resnet18_ytrue.csv')

end



function [info, YPred, scores] = resnet50_finetuning(train_data, val_data, test_data, imageAugmenter)

    numClasses = numel(categories(train_data.Labels));

    net = resnet50;
    inputSize = net.Layers(1).InputSize
    lgraph = layerGraph(net); 


    % Data Augmentation
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_data,'DataAugmentation',imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),val_data);


    % Specify new layers for classification based on number of unique classes
    newfc = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'fc1000',newfc);

    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);
    

    % Training hyperparameters
    opts = trainingOptions('adam', ...
                           'MiniBatchSize', 32, ...
                           'MaxEpochs', 1, ...
                           'InitialLearnRate', 1e-4, ...
                           'L2Regularization', 5e-4, ...
                           'Shuffle','every-epoch', ...
                           'ValidationData', augimdsValidation, ...
                           'ValidationFrequency', 100, ...
                           'Plots', 'training-progress', ...
                           'Verbose', false);


    % Model finetuning
    [net, info] = trainNetwork(train_data, lgraph, opts);
    exportONNXNetwork(net, "resnet50.onnx")


    % Model test and metrics extraction
    [YPred,scores] = classify(net,test_data);


    % Export results
    writematrix(YPred,'resnet50_ypred.csv')
    writematrix(scores,'resnet50_scores.csv')
    writematrix(test_data.Labels,'resnet50_ytrue.csv')

end



function [info, YPred, scores] = darknet19_finetuning(train_data, val_data, test_data, imageAugmenter)

    numClasses = numel(categories(train_data.Labels));

    net = darknet19;
    inputSize = net.Layers(1).InputSize
    lgraph = layerGraph(net); 


    % Data Augmentation
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_data,'DataAugmentation',imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),val_data);


    % Specify new layers for classification based on number of unique classes
    new_convolution2dLayer = convolution2dLayer(1,2,'Name',"new_conv19",'Stride',1,'Padding',0);
    lgraph = replaceLayer(lgraph,"conv19",new_convolution2dLayer);

    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'output',newClassLayer);
    

    % Training hyperparameters
    opts = trainingOptions('adam', ...
                           'MiniBatchSize', 32, ...
                           'MaxEpochs', 1, ...
                           'InitialLearnRate', 1e-4, ...
                           'L2Regularization', 5e-4, ...
                           'Shuffle','every-epoch', ...
                           'ValidationData', augimdsValidation, ...
                           'ValidationFrequency', 100, ...
                           'Plots', 'training-progress', ...
                           'Verbose', false);


    % Model finetuning
    [net, info] = trainNetwork(train_data, lgraph, opts);
    exportONNXNetwork(net, "darknet19.onnx")


    % Model test and metrics extraction
    [YPred,scores] = classify(net,test_data);


    % Export results
    writematrix(YPred,'darknet19_ypred.csv')
    writematrix(scores,'darknet19_scores.csv')
    writematrix(test_data.Labels,'darknet19_ytrue.csv')

end


