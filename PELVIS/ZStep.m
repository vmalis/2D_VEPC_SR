% Step 1.5
% ev plots and streamlines save in the study folder

%==========================================================================
% FRAME crop

start_frame=1;
end_frame=5;
%==========================================================================


%% ======================================================================
% !!!!!  choose ev_calculated_data.mat not mri_data.mat     !!!!!!!!!!!
% =======================================================================
[FileName,PathName] = uigetfile('*.mat','choose a mat file','~/Desktop');

cd(PathName)
load (FileName);



%Select roi
figure
imshow(mat2gray(im_m(:,:,1)),'InitialMagnification', 200)
image_crop=imrect(gca);
image_crop_pos=getPosition(image_crop);

CROP_X=int16([image_crop_pos(1),image_crop_pos(1)+image_crop_pos(3)]);
CROP_Y=int16([image_crop_pos(2),image_crop_pos(2)+image_crop_pos(4)]);


close


%% Ploting EigenVectors overlayed on eigen values and 
%  streamlines overlayed on magnitude image


multiWaitbar('creating plots and movies', 0, 'Color', 'g');



Y_1=permute(Y_1,[2,1,3]);
Y_2=permute(Y_2,[2,1,3]);

% Open video files to write in
writerObj1 = VideoWriter('NEV_SL.avi');
writerObj2 = VideoWriter('PEV_SL.avi');
open(writerObj1);
open(writerObj2);


for i=start_frame:end_frame;
    for x=1:256
        for y=1:256
             Vector_quiv_u_red (x,y) = VectorF_red (x,y,i,1);
             Vector_quiv_v_red (x,y) = VectorF_red (x,y,i,2);
             Vector_quiv_u_blue(x,y) = VectorF_blue(x,y,i,1);
             Vector_quiv_v_blue(x,y) = VectorF_blue(x,y,i,2);
        end
    end
h_fig1=figure('Visible','off');
imshow(Y_1(:,:,i),'DisplayRange',[-400 100],'InitialMagnification',250);
colormap(winter)
colorbar
hold on
[X,Y] = meshgrid(1:256,1:256);
quiver(X,Y,Vector_quiv_u_red,Vector_quiv_v_red,1,'r');
axis tight
axis([CROP_X(1) CROP_X(2) CROP_Y(1) CROP_Y(2)])
hold off
fullname=sprintf('NEV#%d.png',i);
print(h_fig1, '-dpng', fullname);

h_fig2=figure('Visible','off');
imshow(Y_2(:,:,i),'DisplayRange',[-100 400],'InitialMagnification',250);
colormap(autumn)
colorbar
hold on
quiver(X,Y,Vector_quiv_u_blue,Vector_quiv_v_blue,1,'b');
axis tight
axis([CROP_X(1) CROP_X(2) CROP_Y(1) CROP_Y(2)])
hold off
fullname=sprintf('PEV#%d.png',i);
print(h_fig2, '-dpng', fullname);


[x,y] =meshgrid(CROP_X(1):4:CROP_X(2),CROP_Y(1):4:CROP_Y(2));

x=double(x);
y=double(y);

h_fig3=figure('Visible','off');
imshow(mat2gray(im_m(:,:,i)),'InitialMagnification', 200)
hold on
hlines=streamline(X,Y,Vector_quiv_u_red,Vector_quiv_v_red,x,y);
axis tight
axis([CROP_X(1) CROP_X(2) CROP_Y(1) CROP_Y(2)])
set(hlines,'LineWidth',1,'Color','r')
hold off
writeVideo(writerObj1,getframe(gca));

h_fig4=figure('Visible','off');
imshow(mat2gray(im_m(:,:,i)),'InitialMagnification', 200)
hold on
hlines=streamline(X,Y,Vector_quiv_u_blue,Vector_quiv_v_blue,x,y);
set(hlines,'LineWidth',1,'Color','b')
axis tight
axis([CROP_X(1) CROP_X(2) CROP_Y(1) CROP_Y(2)])
hold off
writeVideo(writerObj2,getframe(gca));

multiWaitbar('creating plots and movies', i/(end_frame));

end

multiWaitbar('creating plots and movies', 'Close');  

% Close video files
close(writerObj1);
close(writerObj2);