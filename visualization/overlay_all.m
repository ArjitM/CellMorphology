load directories.mat
k = 1
for loc=folders
    F(k) = overlay(loc);
    k = k + 1;
end
% create the video writer with 1 fps
writerObj = VideoWriter('cellsID.avi');
writerObj.FrameRate = 100;
open(writerObj);
% set the seconds per image
writeVideo(writerObj, F);
    

function[fr] = overlay(loc)

char(loc)
files = dir(char(loc))
[h, ~]=size(files);
for i=3:h
    
    if ~contains(files(i).name,'csv') && ~contains(files(i).name, 'largest') && ~contains(files(i).name, 'overlaid') && ~contains(files(i).name, '_') && contains(files(i).name, '.tif')
        try
            fillCells(strcat(files(i).folder, '/', files(i).name))
            fr = getframe(gcf);
        catch ME
        end
        
    end
end
end

