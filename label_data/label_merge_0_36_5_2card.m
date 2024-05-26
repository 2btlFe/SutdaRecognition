%%
data2 = load("GT_0_35_5_2card.mat");
data2_gTruth = data2.gTruth;

%%
data = load("2cardlabel.mat");
disp(data.gTruth);

dataSource = data.gTruth.DataSource;

% 기존 DataSource에서 전체 경로를 가져옵니다.
oldFilePaths = data.gTruth.DataSource;


%% 새 기본 경로를 정의합니다.
newBasePath = '/home/lbc/Desktop/ComplexDesignProject/2cardlabel_data/';

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

newDataSource = groundTruthDataSource(newFilePaths2);


%% 변환할 디렉토리 이름 매핑
% folderMapping = {
%     'GGeun_10', 'GGeun_28';
%     'Pi_10', 'Pi_29';
%     'Gwang_11', 'Kwang_30';
%     'Sp_11', 'Sp_31';
%     'Pi_11', 'Pi_32';
%     'Gwang_12', 'Kwang_33';
%     'YG_12', 'YG_34';
%     'GGeun_12', 'GGeun_35';
%     'Sp_12', 'Sp_36';
% };
% % 
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
% 
% 
% % DataSource 업데이트
% % 새로운 DataSource 객체 생성
% newDataSource = groundTruthDataSource(newFilePaths3);
% 
% % 새로운 groundTruth 객체 생성
% newGTruth = groundTruth(newDataSource, data3.gTruth.LabelDefinitions, data3.gTruth.LabelData);
% 
% % 필요한 경우, 기존 변수에 새 객체 할당
% data3.gTruth = newGTruth;


%% 
% LabelDefinitions 복사
% newLabelDefs = data.gTruth.LabelDefinitions;
% 
% % 레이블 이름 변경
% idx = strcmp(newLabelDefs.Name, 'Gwang_30');
% newLabelDefs.Name(idx) = {'Kwang_30'};
% 
% idx = strcmp(newLabelDefs.Name, 'Gwang_33');
% newLabelDefs.Name(idx) = {'Kwang_33'};
% 
% 
% % data3.gTruth.LabelData에서 열 이름 변경
% newLabelData = data.gTruth.LabelData;
% newLabelData = renamevars(newLabelData, {'Gwang_30', 'Gwang_33'}, {'Kwang_30', 'Kwang_33'});
% 
% % 새로운 groundTruth 객체 생성
% updatedGTruth = groundTruth(data3.gTruth.DataSource, newLabelDefs, newLabelData);

%% Label Merge

% 새로운 groundTruth 객체 생성
gTruth = groundTruth(newDataSource, data.gTruth.LabelDefinitions, data.gTruth.LabelData);

%% Save the mat file
save("GT_0_35_5_2card.mat", 'gTruth');


%% Load the mat file

