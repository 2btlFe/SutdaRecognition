%% validateInputData 
function validateInputData(ds)
% Validates the input images, bounding boxes and labels and displays the 
% paths of invalid samples. 

% Copyright 2021 The MathWorks, Inc.

% Path to images
info = ds.UnderlyingDatastores{1}.Files;

ds = transform(ds, @isValidDetectorData);
data = readall(ds);

validImgs = [data.validImgs];
validBoxes = [data.validBoxes];
validLabels = [data.validLabels];

msg = "";

if(any(~validImgs))
    imPaths = info(~validImgs);
    str = strjoin(imPaths, '\n');
    imErrMsg = sprintf("Input images must be non-empty and have 2 or 3 dimensions. The following images are invalid:\n") + str;
    msg = (imErrMsg + newline + newline);
end

if(any(~validBoxes))
    imPaths = info(~validBoxes);
    str = strjoin(imPaths, '\n');
    boxErrMsg = sprintf("Bounding box data must be M-by-4 matrices of positive integer values. The following images have invalid bounding box data:\n") ...
        + str;
    
    msg = (msg + boxErrMsg + newline + newline);
end

if(any(~validLabels))
    imPaths = info(~validLabels);
    str = strjoin(imPaths, '\n');
    labelErrMsg = sprintf("Labels must be non-empty and categorical. The following images have invalid labels:\n") + str;
    
    msg = (msg + labelErrMsg + newline);
end

if(~isempty(msg))
    error(msg);
end

end

function out = isValidDetectorData(data)
% Checks validity of images, bounding boxes and labels
for i = 1:size(data,1)
    I = data{i,1};
    boxes = data{i,2};
    labels = data{i,3};

    imageSize = size(I);
    mSize = size(boxes, 1);

    out.validImgs(i) = iCheckImages(I);
    out.validBoxes(i) = iCheckBoxes(boxes, imageSize);
    out.validLabels(i) = iCheckLabels(labels, mSize);
end

end

function valid = iCheckImages(I)
% Validates the input images.

valid = true;
if ndims(I) == 2
    nDims = 2;
else
    nDims = 3;
end
% Define image validation parameters.
classes        = {'numeric'};
attrs          = {'nonempty', 'nonsparse', 'nonnan', 'finite', 'ndims', nDims};
try
    validateattributes(I, classes, attrs);
catch
    valid = false;
end
end

function valid = iCheckBoxes(boxes, imageSize)
% Validates the ground-truth bounding boxes to be non-empty and finite.

valid = true;
% Define bounding box validation parameters.
classes = {'numeric'};
attrs   = {'nonempty', 'integer', 'nonnan', 'finite', 'positive', 'nonzero', 'nonsparse', '2d', 'ncols', 4};
try
    validateattributes(boxes, classes, attrs);
    % Validate if bounding box in within image boundary.
    validateattributes(boxes(:,1)+boxes(:,3)-1, classes, {'<=', imageSize(2)});
    validateattributes(boxes(:,2)+boxes(:,4)-1, classes, {'<=', imageSize(1)}); 
catch
    valid = false;
end
end

function valid = iCheckLabels(labels, mSize)
% Validates the labels.

valid = true;
% Define label validation parameters.
classes = {'categorical'};
attrs   = {'nonempty', 'nonsparse', '2d', 'ncols', 1, 'nrows', mSize};
try
    validateattributes(labels, classes, attrs);
catch
    valid = false;
end
end

%% Auxiliary Function
function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.1]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    
    data(ii,1:2) = {I,bboxes};
end
end








%% Load Dataset 
data = load('Sutda_GT.mat');
Dataset = data.gTruth;
labelData = Dataset.LabelData;

% LabelData의 각 셀을 확인하고 비어 있으면 기본 값으로 채우기
for i = 1:size(Dataset.LabelData, 1)
    for j = 1:size(Dataset.LabelData, 2)
        if isempty(Dataset.LabelData{i, j})
            Dataset.LabelData{i, j} = [0 0 0 0]; % 또는 적절한 기본값
        end
    end
end


%% Cross Validation
rng("default")
totalData = height(Dataset.DataSource.Source);
indices = randperm(totalData);   % Total length of imges
sprintf("%d is the total length", totalData);

% 각 세트의 크기 계산
numTrain = floor(0.6 * totalData);
numValidation = floor(0.2 * totalData);
numTest = totalData - numTrain - numValidation; % 나머지를 테스트 데이터로 사용

% 레이블 데이터 세트로 분할
trainData = Dataset.LabelData(indices(1:numTrain), :);
validationData = Dataset.LabelData(indices(numTrain+1:numTrain+numValidation), :);
testData = Dataset.LabelData(indices(numTrain+numValidation+1:end), :);

% 이미지 경로도 함께 매칭하여 분할 (DataSource 사용)
imagePaths = Dataset.DataSource.Source; % 여기서 DataSource.Source는 각 이미지 파일의 경로를 포함해야 함
trainImages = imagePaths(indices(1:numTrain));
validationImages = imagePaths(indices(numTrain+1:numTrain+numValidation));
testImages = imagePaths(indices(numTrain+numValidation+1:end));

%% Generate Dataset
imdsTrain = imageDatastore(trainImages);
bldsTrain = boxLabelDatastore(trainData);

imdsValidation = imageDatastore(validationImages);
bldsValidation = boxLabelDatastore(validationData);

imdsTest = imageDatastore(testImages);
bldsTest = boxLabelDatastore(testData);

%% Data Aggregation
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

%% ValidateInputData
% 샘플이 유효하지 않은 영상 형식이거나 NaN을 포함하고 있음
% 경계 상자가 0, NaN, Inf를 포함하고 있거나 비어 있음
% 레이블이 누락되었거나 범주형이 아님
% 
% function data = customReadData(imds, blds)
%     img = readimage(imds); % 이미지 읽기
%     bboxData = read(blds); % 바운딩 박스 데이터 읽기
%     data.image = img;
%     data.bbox = bboxData(:, 1); % 바운딩 박스 정보
%     data.labels = bboxData(:, 2); % 레이블 정보
% end

% trainingData.ReadFcn = @(x) customReadData(x{1}, x{2});
% validationData.ReadFcn = @(x) customReadData(x{1}, x{2});
% testData.ReadFcn = @(x) customReadData(x{1}, x{2});

validateInputData(trainingData);
validateInputData(validationData);
validateInputData(testData);

%% Show trainindData
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

reset(trainingData);

%% Network construction
inputSize = [224 224 3];

className = {
    'Kwang_0', 'GGeun_1', 'Pi_2', 'YG_3', 'GGeun_4', 'Pi_5', 'Kwang_6', 'GGeun_7', ...
    'Pi_8', 'YG_9', 'GGeun_10', 'Pi_11', 'YG_12', 'GGeun_13', 'Pi_14', 'YG_15', ...
    'GGeun_16', 'Pi_17', 'YG_18', 'GGeun_19', 'Pi_20', 'Kwang_21', 'YG_22', 'Pi_23', ... 
    'YG_24', 'GGeun_25', 'Pi_26', 'YG_27', 'YG_34', 'Sp_36', 'Sp_31', 'Pi_32', ...
    'Pi_29', 'Kwang_33', 'Kwang_30', 'GGeun_35', 'GGeun_28'
};

rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));

% Anchor Box Generation
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

%% Anchor Configuration
area = anchors(:, 1).*anchors(:,2); % Area
[~,idx] = sort(area,"descend"); % Resorting by descending order of area

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };

detector = yolov4ObjectDetector("csp-darknet53-coco",className,anchorBoxes,InputSize=inputSize);

%% Data Augmentation
augmentedTrainingData = transform(trainingData,@augmentData);


% Adjust DTD background change 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)

%% Define Training Option - Update to recent version
options = trainingOptions("adam", ...
    'GradientDecayFactor', 0.9, ...
    'SquaredGradientDecayFactor', 0.999, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', "none", ...
    'MiniBatchSize', 4, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 70, ...
    'BatchNormalizationStatistics', "moving", ...
    'DispatchInBackground', true, ...
    'ResetInputNormalization', false, ...
    'Shuffle', "every-epoch", ...
    'VerboseFrequency', 20, ...
    'ValidationFrequency', 1000, ...
    'CheckpointPath', tempdir, ...
    'ValidationData', validationData, ...
    'Plots', 'training-progress', ...
    'Verbose', false);



%% Training
doTraining = true;
if doTraining       
    % Train the YOLO v4 detector.
    [detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);
else
    % Load pretrained detector for the example.
    detector = downloadPretrainedYOLOv4Detector();
end

%% Execute detector
data = read(trainingData);
I = data{1};

% I = imread("highway.png");
[bboxes,scores,labels] = detect(detector,I);

%% Display result
I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)

%% Save the Model
% 'detector'는 학습된 YOLO 검출기 객체입니다.
save('detectorModel_0_36.mat', 'detector');


