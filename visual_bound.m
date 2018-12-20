
all = moore_neighbor(img);
figure
imshow(img)
hold on
for d=all
    bound = d{1}
    p = bound{1}
    try
        c = bound{2:end}
        for cb=c
            cbb=cb{1};
            plot(cbb(:,2),cbb(:,1),'g','LineWidth',2);
        end
    end
    plot(p(:,2),p(:,1),'r','LineWidth',2);
end
