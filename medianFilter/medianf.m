name=dir('./img');
[m,n]=size(name);
hold on
ref=imread(['./img/' name(m).name]);
h=fspecial('average', 3);
for i=3:m
    figure;
    img=imread(['./img/' name(i).name]);
    subplot(1,2,1)
    imshow(img,[]);
    p=PSNR(img,ref);
    
    title(['Original PSNR=' num2str(p)]);
%     img(:,:,1)=imfilter(img(:,:,1),h,'symmetric');
%     img(:,:,2)=imfilter(img(:,:,2),h,'symmetric');
%     img(:,:,3)=imfilter(img(:,:,3),h,'symmetric');
    img(:,:,1)=medfilt2(img(:,:,1),[5,5],'symmetric');
    img(:,:,2)=medfilt2(img(:,:,2),[5,5],'symmetric');
    img(:,:,3)=medfilt2(img(:,:,3),[5,5],'symmetric');
    subplot(1,2,2)
    imshow(img,[]);
    p=PSNR(img,ref);
    
    title(['Midian Filter PSNR=' num2str(p)]);
    imwrite(img,['./out/filtered' name(i).name]);
end