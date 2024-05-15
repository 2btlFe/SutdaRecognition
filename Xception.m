%% Get training images
sutda_ds = imageDatastore('OD_dataset_revised/', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainImgs, tempImgs] = splitEachLabel(sutda_ds, 0.6);
[valImgs, testImgs] = splitEachLabel(tempImgs, 0.5);
numClasses = numel(categories(sutda_ds.Labels));

%% Image augmentation
outputSize = [299, 299];
pixelRange = [-30 30];
rotationRange = [-90 90];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandRotation',rotationRange);

augimdsTrain = augmentedImageDatastore(outputSize, trainImgs, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(outputSize, valImgs);
augimdsTest =  augmentedImageDatastore(outputSize, testImgs);

%% Visualize sample images
numTrainImages = numel(trainImgs.Labels);
idx = randperm(numTrainImages, 16);
figure
for i = 1:16
    subplot(4, 4, i)
    I = readimage(trainImgs, idx(i));
    imshow(I)
end

%% Create a network by modifying AlexNet
% net = alexnet; 
% analyzeNetwork(net);
% layers = net.Layers;
% layers(end-2) = fullyConnectedLayer(numClasses);
% layers(end) = classificationLayer;
% 
% analyzeNetwork(layers);

%% ResNet50 Transfer Learning 
% net = resnet50;
% 
% analyzeNetwork(net)
% 
% lgraph = layerGraph(net);
% clear net;
% 
% % New Learnable Layer
% newLearnableLayer = fullyConnectedLayer(numClasses, ... 
%     'Name', 'new_fc', ...
%     'WeightLearnRateFactor', 10, ...
%     'BiasLearnRateFactor', 10);
% 
% % Replacing the last layers with new layers
% lgraph = replaceLayer(lgraph, 'fc1000', newLearnableLayer);
% newsoftmaxLayer = softmaxLayer('Name', 'new_softmax');
% lgraph = replaceLayer(lgraph, 'fc1000_softmax', newsoftmaxLayer);
% newClassLayer = classificationLayer('Name', 'new_classoutput');
% lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);
% 
% analyzeNetwork(lgraph)

%% EfficientNet Transfer Learning 
% net = efficientnetb0;
% 
% analyzeNetwork(net)
% 
% lgraph = layerGraph(net);
% clear net;
% 
% % New Learnable Layer
% newLearnableLayer = fullyConnectedLayer(numClasses, ... 
%     'Name', 'new_fc', ...
%     'WeightLearnRateFactor', 10, ...
%     'BiasLearnRateFactor', 10);
% 
% % Replacing the last layers with new layers
% lgraph = replaceLayer(lgraph, 'efficientnet-b0|model|head|dense|MatMul', newLearnableLayer);
% newsoftmaxLayer = softmaxLayer('Name', 'new_softmax');
% lgraph = replaceLayer(lgraph, 'Softmax', newsoftmaxLayer);
% newClassLayer = classificationLayer('Name', 'new_classoutput');
% lgraph = replaceLayer(lgraph, 'classification', newClassLayer);
% 
% analyzeNetwork(lgraph)

%% ResNet 101 Transfer learning
% net = resnet101;
% 
% analyzeNetwork(net)
% 
% lgraph = layerGraph(net);
% clear net;
% 
% % New Learnable Layer
% newLearnableLayer = fullyConnectedLayer(numClasses, ... 
%     'Name', 'new_fc', ...
%     'WeightLearnRateFactor', 10, ...
%     'BiasLearnRateFactor', 10);
% 
% % Replacing the last layers with new layers
% lgraph = replaceLayer(lgraph, 'fc1000', newLearnableLayer);
% newsoftmaxLayer = softmaxLayer('Name', 'new_softmax');
% lgraph = replaceLayer(lgraph, 'prob', newsoftmaxLayer);
% newClassLayer = classificationLayer('Name', 'new_classoutput');
% lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);
% 
% analyzeNetwork(lgraph)

%% Exception
net = xception;

analyzeNetwork(net)

lgraph = layerGraph(net);
clear net;

% New Learnable Layer
newLearnableLayer = fullyConnectedLayer(numClasses, ... 
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

% Replacing the last layers with new layers
lgraph = replaceLayer(lgraph, 'predictions', newLearnableLayer);
newsoftmaxLayer = softmaxLayer('Name', 'new_softmax');
lgraph = replaceLayer(lgraph, 'predictions_softmax', newsoftmaxLayer);
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

analyzeNetwork(lgraph)

%% Hyperparameter
learnRates = [0.001, 0.0005, 0.0001];
batchSizes = [16, 32];
epochs = [20, 40, 60];


%% Set training algorithm options
bestAccuracy = 0;
bestParams = struct('Rate', 0, 'BatchSize', 0, 'Epochs', 0);
bestModel = 0;

for lr = learnRates
    for bs = batchSizes
        for ep = epochs
            options = trainingOptions('sgdm', ...
                'InitialLearnRate', lr, ...
                'MaxEpochs', ep, ...
                'MiniBatchSize', bs, ...
                'ExecutionEnvironment', 'gpu', ...
                'Plots', 'training-progress', ...
                'Verbose', false);  % Turn off verbose to keep the output clean
            
            
            % [trainedNet, trainInfo] = trainNetwork(augimdsTrain, layers, options);
            [trainedNet, trainInfo] = trainNetwork(augimdsTrain, lgraph, options);
            valpreds = classify(trainedNet, augimdsValidation);
            Accuracy = mean(valpreds == valImgs.Labels);

            fprintf('Testing LR=%f, BS=%d, Epochs=%d, Accuracy=%.2f\n', lr, bs, ep, Accuracy);

            if Accuracy > bestAccuracy
                bestAccuracy = Accuracy;
                bestParams.Rate = lr;
                bestParams.BatchSize = bs;
                bestParams.Epochs = ep;
                bestModel = trainedNet;
            end
        end
    end
end

%% Save model
save('bestModel.mat', 'bestModel', '-v7.3');

fprintf('Best Params -> LR: %f, BatchSize: %d, Epochs: %d, Accuracy: %.2f\n', ...
    bestParams.Rate, bestParams.BatchSize, bestParams.Epochs, bestAccuracy);

%% Test

sutdanet = load('bestModel.mat').bestModel;

%Use trained network to classify test images
testpreds = classify(sutdanet, augimdsTest);

% Evaluation
Accuracy = mean(testpreds == testImgs.Labels);
fprintf('Test Accuracy: %.2f\n', Accuracy);

%% Display the first N test images with their predicted labels
figure;
numImagesToShow = 20; % Number of images you want to display
for i = 1:numImagesToShow
    % Read the next batch of images from augmentedImageDatastore
    [img, info] = read(testImgs);
    
    % Display the image
    subplot(4, 5, i); % Adjust the grid size based on how many images you want to display
    imshow(img);
    
    title_pred = strcat(string(testpreds(i)), '/', string(testImgs.Labels(i)));
    title(sprintf('Predict: %s', title_pred));
    
    fprintf("Predict: %s\n", title_pred);

    % Reset augImdsTest after the last image is displayed
    if i == numImagesToShow
        reset(testImgs);
    end
end

sgtitle('Test Image Predictions');