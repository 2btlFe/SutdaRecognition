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
    I = A{ii,1};        % image
    bboxes = A{ii,2};   % bboxes
    labels = A{ii,3};   % labels 
    sz = size(I);       % shape 

    if numel(sz) == 3 && sz(3) == 3 % size [H, W, C] & C == 3
        I = jitterColorHSV(I,...   % Contrast / Hue(-0.5 ~ 0.5) / Saturation(
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.1]);         % XReflect - leftright flip / Scaling[1~1.1]
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes. - box Warping
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
data = load('./label_data/GT_0524_2442.mat');
Dataset = data.gTruth;
labelData = Dataset.LabelData;

for i = 1:size(labelData, 1)
    for j = 1:size(labelData, 2)
        if isempty(labelData{i, j})
            labelData{i, j} = [0 0 0 0]; % 또는 적절한 기본값
        end
    end
end

% LabelData의 값을 int32로 변환
for i = 1:size(labelData, 1)
    for j = 1:size(labelData, 2)
        % 셀 요소를 추출
        boundingBox = labelData{i, j};
        % 요소가 셀 배열인 경우 변환
        if iscell(boundingBox)
            boundingBox = cell2mat(boundingBox);
        end
        % 요소가 숫자 배열인 경우 int32로 변환하고 다시 셀 배열로 저장
        if isnumeric(boundingBox)
            labelData{i, j} = {int32(boundingBox)};
        end
    end
end

% 기존 DataSource와 새로 변환된 LabelData를 사용하여 새로운 groundTruth 객체 생성
newDataset = groundTruth(Dataset.DataSource, Dataset.LabelDefinitions,  labelData);

%% Cross Validation
rng("default")  % 'default' seed
totalData = height(newDataset.DataSource.Source);
indices = randperm(totalData);   % Total length of images
sprintf("%d is the total length", totalData);

% 각 세트의 크기 계산
numTrain = floor(0.6 * totalData);
numValidation = floor(0.2 * totalData);
numTest = totalData - numTrain - numValidation; % 나머지를 테스트 데이터로 사용

% 레이블 데이터 세트로 분할
trainData = labelData(indices(1:numTrain), :);
validationData = labelData(indices(numTrain+1:numTrain+numValidation), :);
testData = labelData(indices(numTrain+numValidation+1:end), :);

% 이미지 경로도 함께 매칭하여 분할 (DataSource 사용)
imagePaths = newDataset.DataSource.Source; % 여기서 DataSource.Source는 각 이미지 파일의 경로를 포함해야 함
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
trainingData = combine(imdsTrain, bldsTrain);
validationData = combine(imdsValidation, bldsValidation);
testData = combine(imdsTest, bldsTest);


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
annotatedImage = insertShape(I,"Rectangle",bbox);   % put bounding box 
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

% className = {
%     'Ggeun_3', 'Ggeun_1', 'Gwang_1', 'YG_4', 'Ggeun_4', 'YG_2', 'Ggeun_2', 'Gwang_3','YG_5', ...
%     'Ggeun_5', 'YG_6', 'Ggeun_6', 'YG_7', 'Ggeun_7', 'Gwang_8', 'YG_8', 'YG_9', ...
%     'Ggeun_9', 'YG_10', 'Ggeun_10'
% };

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
    'MiniBatchSize', 2, ...
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



% data = read(trainingData);
% inputImage = data{1};
% 
% % 탐지 수행
% [bboxes, scores, labels] = detect(detector, inputImage, 'Threshold', 0.5);
% 
% % 디버깅: 탐지된 바운딩 박스, 점수, 라벨 출력
% disp('Detected bboxes:');
% disp(bboxes);
% disp('Detected scores:');
% disp(scores);
% disp('Detected labels:');
% 
% % Non-Maximum Suppression (NMS) 적용
% if ~isempty(bboxes)
%     indices = selectStrongestBbox(bboxes, scores, 'RatioType', 'Min', 'OverlapThreshold', 0.4);
% 
%     if ~isempty(indices)
%         % categorical indices를 정수형 배열로 변환
%         intIndices = double(indices);
% 
%         % 최종 결과 적용
%         finalBboxes = bboxes(indices, :);
%         finalScores = scores(indices);
%         finalLabels = labels(indices);
% 
%         % 디버깅: 최종 선택된 바운딩 박스, 점수, 라벨 출력
%         disp('Final selected bboxes:');
%         disp(finalBboxes);
%         disp('Final selected scores:');
%         disp(finalScores);
%         disp('Final selected labels:');
% 
%         % 라벨과 점수를 결합하여 텍스트 주석 생성
%         numAnnotations = length(finalLabels);
%         annotations = strings(numAnnotations, 1);
%         for i = 1:numAnnotations
%             annotations(i) = sprintf('%s: %.2f', finalLabels(i), finalScores(i));
%         end
%     else
%         disp('No bounding boxes selected after NMS.');
%         finalBboxes = [];
%         finalScores = [];
%         finalLabels = [];
%         annotations = [];
%     end
% else
%     disp('No bboxes detected.');
%     finalBboxes = [];
%     finalScores = [];
%     finalLabels = [];
%     annotations = [];
% end
% 
% % 결과 시각화
% if ~isempty(finalBboxes)
%     outputImage = insertObjectAnnotation(inputImage, 'rectangle', finalBboxes, annotations);
%     figure;
%     imshow(outputImage);
% else
%     disp('No final bboxes to display.');
%     figure;
%     imshow(inputImage);
% end

%% NMS
% 데이터 읽기 및 탐지 실행
data = read(trainingData);
inputImage = data{1};



% 탐지 수행
[bboxes, scores, labels] = detect(detector, inputImage, 'Threshold', 0.5);

% 디버깅: 탐지된 바운딩 박스, 점수, 라벨 출력
disp('Detected bboxes:');
disp(bboxes);
disp('Detected scores:');
disp(scores);
disp('Detected labels:');
disp(labels);

% 각 레이블에 대해 NMS 적용
uniqueLabels = unique(labels);
filteredBboxes = [];
filteredScores = [];
filteredLabels = [];

for i = 1:length(uniqueLabels)
    currentLabel = uniqueLabels(i);
    labelMask = labels == currentLabel;
    labelBboxes = bboxes(labelMask, :);
    labelScores = scores(labelMask);
    labelLabels = labels(labelMask);
    
    if ~isempty(labelBboxes)
        [selectedBboxes, selectedScores, selectedLabels] = selectStrongestBboxMulticlass(labelBboxes, labelScores, labelLabels, 'RatioType', 'Min', 'OverlapThreshold', 0.4);
        filteredBboxes = [filteredBboxes; selectedBboxes];
        filteredScores = [filteredScores; selectedScores];
        filteredLabels = [filteredLabels; selectedLabels];
    end
end

% 디버깅: 레이블별 NMS 적용 후 바운딩 박스, 점수, 라벨 출력
disp('Filtered bboxes after label-wise NMS:');
disp(filteredBboxes);
disp('Filtered scores after label-wise NMS:');
disp(filteredScores);
disp('Filtered labels after label-wise NMS:');

% 전체 NMS 적용
if ~isempty(filteredBboxes)
    [finalBboxes, finalScores, indices] = selectStrongestBbox(filteredBboxes, filteredScores, 'RatioType', 'Min', 'OverlapThreshold', 0.4);
    finalLabels = filteredLabels(indices);

    % 디버깅: 최종 선택된 바운딩 박스, 점수, 라벨 출력
    disp('Final selected bboxes:');
    disp(finalBboxes);
    disp('Final selected scores:');
    disp(finalScores);
    disp('Final selected labels:');
    
    % 라벨과 점수를 결합하여 텍스트 주석 생성
    numAnnotations = length(finalLabels);
    annotations = strings(numAnnotations, 1);
    for i = 1:numAnnotations
        annotations(i) = sprintf('%s: %.2f', finalLabels(i), finalScores(i));
    end
else
    finalBboxes = [];
    finalScores = [];
    finalLabels = [];
    annotations = [];
end

% 결과 시각화
if ~isempty(finalBboxes)
    outputImage = insertObjectAnnotation(inputImage, 'rectangle', finalBboxes, annotations);
    figure;
    imshow(outputImage);
else
    disp('No final bboxes to display.');
    figure;
    imshow(inputImage)
end

%% Testing the model
% % testData에 대한 mAP 계산 코드
% 
% % testData 초기화
% reset(testData);
% 
% % 검출 결과 저장용 변수 초기화
% allBboxes = {};
% allScores = {};
% allLabels = {};
% groundTruthBoxes = {};
% groundTruthLabels = {};
% 
% % 고유 라벨을 저장할 맵 초기화
% labelMap = containers.Map();
% labelCounter = 1;
% 
% % 데이터셋 반복
% while hasdata(testData)
%     % 데이터 읽기
%     data = read(testData);
%     inputImage = data{1};
%     gtBoxes = data{2}; % 실제 바운딩 박스 데이터
%     gtLabels = data{3}; % 실제 라벨 데이터
% 
%     % 고유 라벨을 숫자로 매핑
%     gtLabelNumbers = zeros(size(gtLabels));
%     for k = 1:length(gtLabels)
%         label = char(gtLabels(k));
%         if ~isKey(labelMap, label)
%             labelMap(label) = labelCounter;
%             labelCounter = labelCounter + 1;
%         end
%         gtLabelNumbers(k) = labelMap(label);
%     end
% 
%     % 실제 라벨 정보 저장
%     groundTruthBoxes{end+1} = gtBoxes; % 바운딩 박스 데이터를 double 형식으로 변환
%     groundTruthLabels{end+1} = gtLabelNumbers; % 라벨을 숫자 형식으로 저장
% 
%     % 탐지 수행
%     [bboxes, scores, labels] = detect(detector, inputImage, 'Threshold', 0.5);
% 
%     % 각 레이블에 대해 NMS 적용
%     uniqueLabels = unique(labels);
%     filteredBboxes = [];
%     filteredScores = [];
%     filteredLabels = [];
% 
%     for j = 1:length(uniqueLabels)
%         currentLabel = uniqueLabels(j);
%         labelMask = labels == currentLabel;
%         labelBboxes = bboxes(labelMask, :);
%         labelScores = scores(labelMask);
%         labelLabels = labels(labelMask);
% 
%         if ~isempty(labelBboxes)
%             [selectedBboxes, selectedScores, selectedLabels] = selectStrongestBboxMulticlass(labelBboxes, labelScores, labelLabels, 'RatioType', 'Min', 'OverlapThreshold', 0.4);
%             filteredBboxes = [filteredBboxes; selectedBboxes];
%             filteredScores = [filteredScores; selectedScores];
%             filteredLabels = [filteredLabels; selectedLabels]; % 숫자 형식으로 변환
%         end
%     end
% 
%     % 전체 NMS 적용
%     if ~isempty(filteredBboxes)
%         [finalBboxes, finalScores, indices] = selectStrongestBbox(filteredBboxes, filteredScores, 'RatioType', 'Min', 'OverlapThreshold', 0.4);
%         finalLabels = filteredLabels(indices);
%     else
%         finalBboxes = [];
%         finalScores = [];
%         finalLabels = [];
%     end
% 
%     % 검출 결과 저장
%     allBboxes{end+1} = finalBboxes; % 바운딩 박스 데이터를 double 형식으로 변환
%     allScores{end+1} = finalScores;
%     allLabels{end+1} = finalLabels; % 숫자 형식으로 변환
% end
% 
% % 바운딩 박스와 라벨 데이터가 적절한 형식인지 확인
% allBboxes = cellfun(@(x) double(x), allBboxes, 'UniformOutput', false);
% allScores = cellfun(@(x) double(x), allScores, 'UniformOutput', false);
% allLabels = cellfun(@(x) double(x), allLabels, 'UniformOutput', false);
% 
% groundTruthBoxes = cellfun(@(x) double(x), groundTruthBoxes, 'UniformOutput', false);
% groundTruthLabels = cellfun(@(x) double(x), groundTruthLabels, 'UniformOutput', false);
% 
% % 평가용 데이터 준비
% detectionResults = table(allBboxes', allScores', allLabels', 'VariableNames', {'Boxes', 'Scores', 'Labels'});
% groundTruth = table(groundTruthBoxes', groundTruthLabels', 'VariableNames', {'Boxes', 'Labels'});
% 
% % mAP 계산
% [ap, recall, precision] = evaluateDetectionPrecision(detectionResults, groundTruth);
% 
% % mAP 출력
% disp('mAP:');
% disp(ap);
% 
% % 평가 결과 시각화
% figure;
% plot(recall{1}, precision{1});
% xlabel('Recall');
% ylabel('Precision');
% title('Precision-Recall Curve');
% grid on;



%% Save the Model
% 'detector'는 학습된 YOLO 검출기 객체입니다.
save('detectorModel_0524_2424.mat', 'detector');


%% TestDataset

load('detectorModel_0524_2424.mat', 'detector');
testDataDir = 'test4'; % 테스트 데이터가 저장된 디렉토리 경로
testImages = imageDatastore(testDataDir);

% 예측 수행 및 결과 시각화
% 'result' 디렉토리가 존재하지 않으면 생성
dirname = 'Result/result_0525_test4';
if ~exist(dirname, 'dir')
    mkdir(dirname);
end

for i = 1:numel(testImages.Files)
    img = readimage(testImages, i);
    
    % 탐지 수행
    [bboxes, scores, labels] = detect(detector, img, 'Threshold', 0.5);
    
    % 디버깅: 탐지된 바운딩 박스, 점수, 라벨 출력
    disp('Detected bboxes:');
    disp(bboxes);
    disp('Detected scores:');
    disp(scores);
    disp('Detected labels:');
    
    % 각 레이블에 대해 NMS 적용
    uniqueLabels = unique(labels);
    filteredBboxes = [];
    filteredScores = [];
    filteredLabels = [];
    
    for j = 1:length(uniqueLabels)
        currentLabel = uniqueLabels(j);
        labelMask = labels == currentLabel;
        labelBboxes = bboxes(labelMask, :);
        labelScores = scores(labelMask);
        labelLabels = labels(labelMask);
        
        if ~isempty(labelBboxes)
            [selectedBboxes, selectedScores, selectedLabels] = selectStrongestBboxMulticlass(labelBboxes, labelScores, labelLabels, 'RatioType', 'Min', 'OverlapThreshold', 0.4);
            filteredBboxes = [filteredBboxes; selectedBboxes];
            filteredScores = [filteredScores; selectedScores];
            filteredLabels = [filteredLabels; selectedLabels];
        end
    end
    
    % 디버깅: 레이블별 NMS 적용 후 바운딩 박스, 점수, 라벨 출력
    disp('Filtered bboxes after label-wise NMS:');
    disp(filteredBboxes);
    disp('Filtered scores after label-wise NMS:');
    disp('Filtered labels after label-wise NMS:');
    
    % 전체 NMS 적용
    if ~isempty(filteredBboxes)
        [finalBboxes, finalScores, indices] = selectStrongestBbox(filteredBboxes, filteredScores, 'RatioType', 'Min', 'OverlapThreshold', 0.4);
        finalLabels = filteredLabels(indices);
    
        % 디버깅: 최종 선택된 바운딩 박스, 점수, 라벨 출력
        disp('Final selected bboxes:');
        disp(finalBboxes);
        disp('Final selected scores:');
        disp('Final selected labels:');
        
        % 라벨과 점수를 결합하여 텍스트 주석 생성
        numAnnotations = length(finalLabels);
        annotations = strings(numAnnotations, 1);
        for k = 1:numAnnotations
            annotations(k) = sprintf('%s: %.2f', finalLabels(k), finalScores(k));
        end
    else
        finalBboxes = [];
        finalScores = [];
        finalLabels = [];
        annotations = [];
    end
    
    % 결과 시각화 및 저장
    [~, name, ext] = fileparts(testImages.Files{i});
    outputFile = fullfile(dirname, [name, ext]);
    
    if ~isempty(finalBboxes)
        outputImage = insertObjectAnnotation(img, 'rectangle', finalBboxes, annotations);
        imwrite(outputImage, outputFile);
        disp(['Annotated image saved as "', outputFile, '".']);
    else
        imwrite(img, outputFile);
        disp(['Original image saved as "', outputFile, '".']);
    end
end



