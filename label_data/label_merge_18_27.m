
data3 = load("label_18_27.mat");
disp(data3.gTruth);


dataSource2 = data3.gTruth.DataSource;

% 기존 DataSource에서 전체 경로를 가져옵니다.
oldFilePaths = data3.gTruth.DataSource;


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
% folderMapping = {
%     '8', 'Pi_8';
%     '9', 'YG_9';
%     '10', 'GGeun_10';
%     '11', 'Pi_11';
%     '12', 'YG_12';
%     '13', 'GGeun_13';
%     '14', 'Pi_14';
%     '15', 'YG_15';
%     '16', 'GGeun_16';
%     '17', 'Pi_17';
% };
% 
% % 새 파일 경로 배열을 생성
% newFilePaths3 = cell(size(newFilePaths2));
% 
% for i = 1:length(newFilePaths2)
%     newPath = newFilePaths2{i};
%     % 각 매핑에 대하여 경로를 수정
%     for j = 1:size(folderMapping, 1)
%         oldFolderName = ['/' folderMapping{j, 1} '/'];
%         newFolderName = ['/' folderMapping{j, 2} '/'];
%         % 경로에서 폴더 이름 변경
%         newPath = strrep(newPath, oldFolderName, newFolderName);
%     end
%     newFilePaths3{i} = newPath;
% end


% DataSource 업데이트
% 새로운 DataSource 객체 생성
newDataSource = groundTruthDataSource(newFilePaths2);

% 새로운 groundTruth 객체 생성
newGTruth = groundTruth(newDataSource, data3.gTruth.LabelDefinitions, data3.gTruth.LabelData);

% 필요한 경우, 기존 변수에 새 객체 할당
data3.gTruth = newGTruth;


%% 
% newGTruth.LabelData의 모든 레이블을 처리
columnNames = newGTruth.LabelData.Properties.VariableNames;
newLabelData = table();

for i = 1:length(columnNames)
    labelName = columnNames{i};
    if isfield(newGTruth.LabelData.(labelName){1}, 'Position')
        newLabelData.(labelName) = cellfun(@(s) s.Position, newGTruth.LabelData.(labelName), 'UniformOutput', false, 'ErrorHandler', @(x,y) []);
    else
        newLabelData.(labelName) = repmat({[]}, size(newGTruth.LabelData, 1), 1);
    end
end

%% 
% 기존 LabelDefinitions 복사
newLabelDefs = newGTruth.LabelDefinitions;

% Hierarchy 열 제거
newLabelDefs.Hierarchy = [];




%% 새 groundTruth 객체 생성
updatedGTruth = groundTruth(newGTruth.DataSource, newLabelDefs, newLabelData);

%% Label Merge

% Datasource Aggregation
% updatedGTruth와 GT_0_18의 이미지 경로를 추출하고 합치기
allImagePaths = [GT_0_18.DataSource.Source; updatedGTruth.DataSource.Source];
% 새 DataSource 객체 생성
newDataSource = groundTruthDataSource(allImagePaths);


% LabelDefinitions 합치기
combinedLabelDefs = [GT_0_18.LabelDefinitions; updatedGTruth.LabelDefinitions];

% 중복 제거 (레이블 이름 기준)
[~, uniqueIdx] = unique(combinedLabelDefs.Name, 'stable');
combinedLabelDefs = combinedLabelDefs(uniqueIdx, :);

% 새 LabelData 테이블 생성 (빈 셀로 초기화)
totalImages = size(GT_0_18.LabelData, 1) + size(updatedGTruth.LabelData, 1);
newLabelData = array2table(cell(totalImages, length(combinedLabelDefs.Name)), 'VariableNames', combinedLabelDefs.Name);

% GT_0_18의 LabelData 복사
for i = 1:size(GT_0_18.LabelData, 2)
    labelName = GT_0_18.LabelData.Properties.VariableNames{i};
    newLabelData{1: size(GT_0_18.LabelData, 1) , labelName} = GT_0_18.LabelData{:, i};
end

% updatedGTruth의 LabelData 복사
for i = 1:size(updatedGTruth.LabelData, 2)
    labelName = updatedGTruth.LabelData.Properties.VariableNames{i};
    newLabelData{size(GT_0_18.LabelData, 1) + 1 : end, labelName} = updatedGTruth.LabelData{:, i};
end


% 새로운 groundTruth 객체 생성
GT_0_27 = groundTruth(newDataSource, combinedLabelDefs, newLabelData);

