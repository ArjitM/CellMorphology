function [all] = moore_neighbor(img)
img
pic = imread(img);
[B,L,N,A] = bwboundaries(pic);
% figure;
% imshow(img);
% hold on;

all = {};
for k = 1:N 
        % Boundary k is the parent of a hole if the k-th column 
        % of the adjacency matrix A contains a non-zero element 
        
        boundary = {B{k}};
        if (nnz(A(:,k)) > 0) 
%             plot(boundary(:,2),... 
%                 boundary(:,1),'r','LineWidth',2); 
            % Loop through the children of boundary k 
            for l = find(A(:,k))' 
                boundary{end+1} = B{l}; 
%                 plot(boundary(:,2),... 
%                     boundary(:,1),'g','LineWidth',2); 
            end
        end
        all{end+1} = boundary; 
end
save(replace(img, '_BinaryPivots.tif', '_bounds.mat'), 'all');
end