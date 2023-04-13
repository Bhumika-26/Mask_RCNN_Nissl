direcT = '/nfs/data/main/M32/Process_Detection/ProcessDetPass1_unreg/Likelihood/MD848';
direcOut = '/nfs/data/main/M32/Process_Detection/ProcessDetPass1_unreg/MD848';
direc = dir(fullfile(direcT, '*.jp2'));

thresh = 100;
ch = 1;
parpool(6);
parfor i = 1 : length(direc)
    disp(direc(i).name);
    try
        img = imread(fullfile(direcT, direc(i).name));
        imBW  = img(:,:,ch) > thresh;
        imgOut = zeros([size(img,1) size(img,2) 3]); 
        imgOut(:,:,1) = 0;
        imgOut(:,:,2) = imBW;
        imgOut(:,:,3) = imBW;
        imwrite(imgOut, fullfile(direcOut, direc(i).name));
    catch ME
        disp("FAIL!!!! ");
        fprintf('Error: %s\n', ME.message);
        continue;
        
    end
end
   