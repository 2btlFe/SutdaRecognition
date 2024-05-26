data1 = load("label_0_7.mat");
data2 = load("label_8_17.mat");
data3 = load("label_18_27.mat");
data4 = load("label_28_37.mat");

disp(data1.gTruth);
disp(data2.gTruth);
disp(data3.gTruth);
disp(data4.gTruth);


%% 예제로 두 데이터 세트의 DataSource가 cell array인 경우
dataSource1 = data1.gTruth.DataSource.Source;
dataSource2 = data2.gTruth.DataSource;

% 기존 DataSource에서 전체 경로를 가져옵니다.
oldFilePaths = data2.gTruth.DataSource;


% 새 기본 경로를 정의합니다.
newBasePath = '/home/lbc/Desktop/ComplexDesignProject/fulldata/';

% 새 파일 경로 목록을 생성합니다.
newFilePaths = cell(size(oldFilePaths));  % 새 경로를 저장할 셀 배열 초기화

for i = 1:length(oldFilePaths)
    % 각 파일 경로에서 중요한 하위 경로 부분만 추출합니다. (예: 'GGeun_10\YHS_9110.jpg')
    tokens = regexp(oldFilePaths{i}, '[\\/]([^\\/]+[\\/][^\\/]+)$', 'tokens');  % 정규 표현식으로 필요한 부분 추출
    if ~isempty(tokens)
        % 새로운 전체 경로를 조합합니다.
        newFilePaths{i} = fullfile(newBasePath, tokens{1}{1});
    else
        % 만약 예상치 못한 형식이면, 원래 경로를 유지합니다.
        newFilePaths{i} = oldFilePaths{i};
    end
end

newFilePaths2 = cell(size(oldFilePaths));
for i = 1:length(newFilePaths)
    % 백슬래시를 슬래시로 변환
    newFilePaths2{i} = strrep(newFilePaths{i}, '\', '/');
end




%% 변환할 디렉토리 이름 매핑
folderMapping = {
    '8', 'Pi_8';
    '9', 'YG_9';
    '10', 'GGeun_10';
    '11', 'Pi_11';
    '12', 'YG_12';
    '13', 'GGeun_13';
    '14', 'Pi_14';
    '15', 'YG_15';
    '16', 'GGeun_16';
    '17', 'Pi_17';
};

% 새 파일 경로 배열을 생성
newFilePaths3 = cell(size(newFilePaths2));

for i = 1:length(newFilePaths2)
    newPath = newFilePaths2{i};
    % 각 매핑에 대하여 경로를 수정
    for j = 1:size(folderMapping, 1)
        oldFolderName = ['/' folderMapping{j, 1} '/'];
        newFolderName = ['/' folderMapping{j, 2} '/'];
        % 경로에서 폴더 이름 변경
        newPath = strrep(newPath, oldFolderName, newFolderName);
    end
    newFilePaths3{i} = newPath;
end


% DataSource 업데이트
% 새로운 DataSource 객체 생성
newDataSource = groundTruthDataSource(newFilePaths3);

% 새로운 groundTruth 객체 생성
newGTruth = groundTruth(newDataSource, data2.gTruth.LabelDefinitions, data2.gTruth.LabelData);

% 필요한 경우, 기존 변수에 새 객체 할당
data2.gTruth = newGTruth;

%% 


% LabelDefinitions 병합
allLabelDefs = [data1.gTruth.LabelDefinitions; data2.gTruth.LabelDefinitions];
[~, uniqueIdx] = unique(allLabelDefs.Name, 'stable');
combinedLabelDefs = allLabelDefs(uniqueIdx, :);


% 새 LabelData의 모든 레이블에 대해 빈 셀 배열 초기화
combinedLabelData = array2table(cell(825, 18), 'VariableNames', combinedLabelDefs.Name);

% data1의 LabelData 채우기
for i = 1:width(data1.gTruth.LabelData)
    labelName = data1.gTruth.LabelData.Properties.VariableNames{i};
    combinedLabelData(1:363, labelName) = data1.gTruth.LabelData(:, i);
end

% data2의 LabelData 채우기
for i = 1:width(data2.gTruth.LabelData)
    labelName = data2.gTruth.LabelData.Properties.VariableNames{i};
    combinedLabelData(364:825, labelName) = data2.gTruth.LabelData(:, i);
end


% 가정: 새 DataSource가 이미 생성되었거나 적절한 DataSource 설정 필요
newDataSource = [data1.gTruth.DataSource.Source; data2.gTruth.DataSource.Source];


% 이미지 파일 경로가 포함된 배열이라고 가정
imageFilePaths = newDataSource;  % newDataSource는 이미지 파일 경로의 배열이어야 함

% groundTruthDataSource 객체 생성
dataSourceObj = groundTruthDataSource(imageFilePaths);

% 새로운 groundTruth 객체 생성
GT_0_18 = groundTruth(dataSourceObj, combinedLabelDefs, combinedLabelData);
