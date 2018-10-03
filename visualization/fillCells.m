function[]=fillCells(fileName)
pic = imread(fileName);

fileName
sliceNum = str2num(extractBefore(extractAfter(fileName, '00'), '.'))
id = imread(strcat(extractBefore(fileName, '00'),strcat('largest', num2str(sliceNum), '.tif')));

temp = size(id);

imshow(pic);

alpha=zeros(temp(1),temp(2));
for x=1:temp(1)
    for y=1:temp(2)
        if id(x,y,1)>50 || id(x,y,2)>50 || id(x,y,3)>50
             %{
            id(x,y,1)=256;
            id(x,y,2)=0;
            id(x,y,3)=0;
            %}
            alpha(x,y)=0.90;
        end
    end
end

hold on
p=imshow(id);
hold off

%alpha = zeros(512,512)+0.5;
set(p, 'AlphaData', alpha);
f=getframe;
produced=f.cdata(2:513,2:513,1:3);
imwrite(produced,replace(fileName,'.','overlaid.'));
end