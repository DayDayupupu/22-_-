clear;
clc;

path = 'E:\pycharm project\MGNet_pytorch\MG_dataset\RGBD_for_train\';
GT_path = [path,'depth\'];

GTOutput = dir(fullfile(GT_path));
fileNames_GT = {GTOutput.name}';

num_data = size(fileNames_GT, 1);
GT_edge_folder = [path,'depth_edge'];
mkdir(GT_edge_folder);

for i = 4    % 遍历所有图片

    GT_file=[GT_path, fileNames_GT{i,1}];
    GT_image = imread(GT_file);
    
    str = fileNames_GT{i,1};
    [begin, str_size] = size(str);
    str = str(str_size-3:str_size);

    max(max(GT_image))
    min(min(GT_image))
    
    W = fspecial('gaussian',[5,5],1); 
    GT_image = imfilter(GT_image, W, 'replicate');
    figure,imshow(GT_image)
    figure,imhist(GT_image)
    G_edge = edge(GT_image,'canny',0.1);

    
    imwrite(G_edge,[GT_edge_folder,'\',fileNames_GT{i,1}]);%将图片保存在程序所在文件夹中
    disp(['处理完第' num2str(i-2) '张图片'])
end
disp('处理完毕')


