%=========================================================================
%   Step3 for 2d Strain Analysis for Pelvis
%
%       part of 2d Strain Rate Toolkit
%=========================================================================
%
% INput: (output of Step1)
%
%           2) ev_calculated_data.mat
%
% OUTput:   1) plots, excel spreadsheets
%_____________________________________________________
% required subroutines:
%           1) track2dv4
%           2) subplottight
%           3) inoverlay
%           4) xlwrite
%_____________________________________________________
% 
% written by Vadim Malis
% 02/15 at UCSD RIL
%==========================================================================

%%
javaaddpath('poi_library/poi-3.8-20120326.jar');
javaaddpath('poi_library/poi-ooxml-3.8-20120326.jar');
javaaddpath('poi_library/poi-ooxml-schemas-3.8-20120326.jar');
javaaddpath('poi_library/xmlbeans-2.3.0.jar');
javaaddpath('poi_library/dom4j-1.6.1.jar');
javaaddpath('poi_library/stax-api-1.0.1.jar');
 
PathName = uigetdir('choose data folder','~/Desktop');

cd(PathName)
load('ev_calculated_data.mat','ANGLE','VectorF_blue','VectorF_red',...
    'Y_1','Y_2','Series_name','SliceLocation','dt','force_mean','im_m',...
    'numphases','resolution','v_rl','v_ap','v_si','v_rl_sm','v_ap_sm',...
    'v_si_sm','PatientID','force_ref')

% Number of ROIS
p = 7;
% Number of frames
q=numphases;


%% calculate vector magnitudes
NEV=permute(Y_1,[2,1,3]);NEV(~NEV)=NaN;     %negative EV
PEV=permute(Y_2,[2,1,3]);PEV(~PEV)=NaN;     %positive EV
ANGLE=permute(ANGLE,[2,1,3]);ANGLE(~ANGLE)=NaN;

%% get reference frame: force and min velocity aproaches (force is default)

%force
force_samplerate=size(force_mean,2)/numphases;
[~,point_idx]=min(force_mean(:));
force_ref=round(point_idx/force_samplerate);

% %min velocity
% V_R_mean=squeeze(mean(mean(V_R,1),2));
% V_R_std=squeeze(std(std(V_R,1,1),1,2));
% [~,velocity_ref]=min(V_R_mean+V_R_std);



%% Select main ROI
figure
imshow(mat2gray(im_m(:,:,force_ref)),'InitialMagnification', 200)
roi_size=75;
image_crop=imrect(gca,[100,100,roi_size,roi_size]);
setResizable(image_crop,0)
wait(image_crop);

image_crop_pos=getPosition(image_crop);

CROP_X=int16([image_crop_pos(1),image_crop_pos(1)+image_crop_pos(3)-1]);
CROP_Y=int16([image_crop_pos(2),image_crop_pos(2)+image_crop_pos(4)-1]);

close

%% crop all volumes
ANGLE       =   ANGLE(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
NEV         =   NEV(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
PEV         =   PEV(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
SEV         =   PEV+NEV;
IMAGE       =   mat2gray(im_m(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:));
v_rl        =   v_rl(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
v_si        =   v_si(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
v_ap        =   v_ap(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
FULL_IMAGE  =   mat2gray(im_m);


%% Selecting ROIs

%roi confirmation variable
check=true;
while check
    
    
%% Masking GUI


%image stack
im_seq      =   zeros(roi_size,roi_size,q,3);
    
%converting to RGB
im_seq(:,:,:,1) = IMAGE;
im_seq(:,:,:,2) = IMAGE;
im_seq(:,:,:,3) = IMAGE;

I=squeeze(im_seq(:,:,force_ref,:));

figure
imshow(I,'InitialMagnification', 500);
title('magnitude image')


%YOU CAN CHANGE ROI SIZE HERE----------------
roi_x=3;
roi_y=3;

%%%------------ROI1--------------------------
h = imrect(gca,[roi_size/2,roi_size/2,roi_x,roi_y]);
setResizable(h,0)
wait(h);
mask   = createMask(h);
border = uint8(bwperim(mask,8));
[row1,col1] = find(mask);
I=imoverlay(I,border,[1 1 0]);

cla(gca)

imshow(I,'InitialMagnification', 500);

%%%------------ROI2--------------------------
h = imrect(gca,[col1(1),row1(1),roi_x,roi_y]);
setResizable(h,0)
wait(h);
mask   = createMask(h);
border = uint8(bwperim(mask,8));
[row2,col2] = find(mask);
I=imoverlay(I,border,[1 1 0]);

cla(gca)

imshow(I,'InitialMagnification', 500);


%%%------------ROI3--------------------------
h = imrect(gca,[col2(1),row2(1),roi_x,roi_y]);
setResizable(h,0)
wait(h);
mask   = createMask(h);
border = uint8(bwperim(mask,8));
[row3,col3] = find(mask);
I=imoverlay(I,border,[1 1 0]);

cla(gca)

imshow(I,'InitialMagnification', 500);

%%%------------ROI4--------------------------
h = imrect(gca,[col3(1),row3(1),roi_x,roi_y]);
setResizable(h,0)
wait(h);
mask   = createMask(h);
border = uint8(bwperim(mask,8));
[row4,col4] = find(mask);
I=imoverlay(I,border,[1 1 0]);

cla(gca)

imshow(I,'InitialMagnification', 500);

%%%------------ROI5--------------------------
h = imrect(gca,[col4(1),row4(1),roi_x,roi_y]);
setResizable(h,0)
wait(h);
mask   = createMask(h);
border = uint8(bwperim(mask,8));
[row5,col5] = find(mask);
I=imoverlay(I,border,[1 1 0]);

cla(gca)

imshow(I,'InitialMagnification', 500);

%%%------------ROI6--------------------------
h = imrect(gca,[col5(1),row5(1),roi_x,roi_y]);
setResizable(h,0)
wait(h);
mask   = createMask(h);
border = uint8(bwperim(mask,8));
[row6,col6] = find(mask);
I=imoverlay(I,border,[1 1 0]);

cla(gca)

imshow(I,'InitialMagnification', 500);

%%%------------ROI7--------------------------
h = imrect(gca,[col6(1),row6(1),roi_x,roi_y]);
setResizable(h,0)
wait(h);
mask   = createMask(h);
border = uint8(bwperim(mask,8));
[row7,col7] = find(mask);
I=imoverlay(I,border,[1 1 0]);

close

msg=msgbox('Please wait while tracking is in progress');

%tracking
[xs1,ys1] = track2dv4(col1,row1,v_rl*-1,v_si,dt,resolution,force_ref);
[xs2,ys2] = track2dv4(col2,row2,v_rl*-1,v_si,dt,resolution,force_ref);
[xs3,ys3] = track2dv4(col3,row3,v_rl*-1,v_si,dt,resolution,force_ref);
[xs4,ys4] = track2dv4(col4,row4,v_rl*-1,v_si,dt,resolution,force_ref);
[xs5,ys5] = track2dv4(col5,row5,v_rl*-1,v_si,dt,resolution,force_ref);
[xs6,ys6] = track2dv4(col6,row6,v_rl*-1,v_si,dt,resolution,force_ref);
[xs7,ys7] = track2dv4(col7,row7,v_rl*-1,v_si,dt,resolution,force_ref);

XS=zeros(roi_x*roi_y,p,numphases);
YS=zeros(roi_x*roi_y,p,numphases);
XS(:,1,:)=xs1; XS(:,2,:)=xs2; XS(:,3,:)=xs3; XS(:,4,:)=xs4;
XS(:,5,:)=xs5; XS(:,6,:)=xs6; XS(:,7,:)=xs7;

YS(:,1,:)=ys1; YS(:,2,:)=ys2; YS(:,3,:)=ys3; YS(:,4,:)=ys4;
YS(:,5,:)=ys5; YS(:,6,:)=ys6; YS(:,7,:)=ys7;

%Creating MASK
MASK=zeros(roi_size,roi_size,q,7);

for i=1:q
   
    for jj=1:size(row1,1)    
    MASK(int16(ys1(jj,i)),int16(xs1(jj,i)),i,1)=1; 
    MASK(int16(ys2(jj,i)),int16(xs2(jj,i)),i,2)=1;
    MASK(int16(ys3(jj,i)),int16(xs3(jj,i)),i,3)=1;
    MASK(int16(ys4(jj,i)),int16(xs4(jj,i)),i,4)=1;  
    MASK(int16(ys5(jj,i)),int16(xs5(jj,i)),i,5)=1;
    MASK(int16(ys6(jj,i)),int16(xs6(jj,i)),i,6)=1;
    MASK(int16(ys7(jj,i)),int16(xs7(jj,i)),i,7)=1;
    end

end





clear xs1 xs2 xs3 xs4 xs5 xs6 xs7 ys1 ys2 ys3 ys4 ys5 ys6 ys7


%% video

vid = VideoWriter('video_roi.avi');
vid.Quality=100;
open(vid);


figure('Position',[200,100,1024,550])
    for j=1:q
        
        subplottight(1,2,1)
        imshow(squeeze(im_seq(:,:,j,:)))
        title('Partial FOV')
        hold on
        plot(XS(:,:,j),YS(:,:,j),'y.','MarkerSize',10)
        for i=1:p
            text(mean(XS(:,i,j),1),mean(YS(:,i,j),1)+4,num2str(i),'FontSize',22,'Color','y')
        end
            
            
        subplottight(1,2,2)
        imshow(squeeze(FULL_IMAGE(:,:,j,:)))
        title('Full FOV')
        hold on
        plot(XS(:,:,j)+double(CROP_X(1)),YS(:,:,j)+double(CROP_Y(1)),'y.','MarkerSize',3)
        
        frame = getframe(gcf);     
        
        if j==force_ref
            I=frame2im(frame);        
        end
              
        writeVideo(vid,frame);

    end

close(vid);

    %questdlg
    choice = questdlg('Looks good?','?', 'No','Yes','Yes');

    switch choice
        case 'No'
            check=true;
        case 'Yes'
            check=false;
        name_image = sprintf('image.png'); 
        imwrite(I,name_image,'png');
    end

close

end

close(msg)
close

%% Masking

%allocating for masked
n_ev = zeros(p,roi_size,roi_size,q);
p_ev = zeros(p,roi_size,roi_size,q);
s_ev = zeros(p,roi_size,roi_size,q);
ang  = zeros(p,roi_size,roi_size,q);


for i=1:p       %rois

    for j=1:q   %frames
    
        n_ev(i,:,:,j) =   NEV(:,:,j).*MASK(:,:,j,i);
        p_ev(i,:,:,j) =   PEV(:,:,j).*MASK(:,:,j,i);
        s_ev(i,:,:,j) =   SEV(:,:,j).*MASK(:,:,j,i);
        ang(i,:,:,j) =    ANGLE(:,:,j).*MASK(:,:,j,i);
        
    end
end


save('Mask.mat','MASK');

n_ev(n_ev==0) = NaN;
p_ev(p_ev==0) = NaN;
s_ev(s_ev==0) = NaN;
ang(ang==0) = NaN;





%% Data for plot


% Displacement
d_X=bsxfun(@minus,XS,XS(:,:,force_ref));
d_Y=bsxfun(@minus,YS,YS(:,:,force_ref));

d_R=sqrt(d_X.^2+d_Y.^2)*10/resolution;

%Velocities

v_x=zeros(roi_x*roi_y,p,numphases);
v_y=zeros(roi_x*roi_y,p,numphases);
v_z=zeros(roi_x*roi_y,p,numphases);

for i=1:roi_x*roi_y
    for j=1:p
        for k=1:numphases
    
            v_x(i,j,k)=v_rl(int16(YS(i,j,k)),int16(XS(i,j,k)),k);
            v_y(i,j,k)=(-1)*v_si(int16(YS(i,j,k)),int16(XS(i,j,k)),k);
            v_z(i,j,k)=v_ap(int16(YS(i,j,k)),int16(XS(i,j,k)),k);

        end
    end
end


% per ROI
N_EV = squeeze(nanmean(nanmean(n_ev,2),3));
P_EV = squeeze(nanmean(nanmean(p_ev,2),3));
S_EV = squeeze(nanmean(nanmean(s_ev,2),3));
ANGL = squeeze(nanmean(nanmean(ang,2),3));
D_R  = squeeze(mean(d_R,1));
V_X  = squeeze(mean(v_x,1));
V_Y  = squeeze(mean(v_y,1));
V_Z  = squeeze(mean(v_z,1));


% average
N_EV_A = squeeze(nanmean(nanmean(nanmean(n_ev,2),3),1));
P_EV_A = squeeze(nanmean(nanmean(nanmean(p_ev,2),3),1));
S_EV_A = squeeze(nanmean(nanmean(nanmean(s_ev,2),3),1));
ANGL_A = squeeze(nanmean(nanmean(nanmean(ang,2),3),1));
D_R_A  = squeeze(mean(D_R,1));
V_X_A  = squeeze(mean(V_X,1));
V_Y_A  = squeeze(mean(V_Y,1));
V_Z_A  = squeeze(mean(V_Z,1));



% % SD
N_EV_SD_A = std(N_EV(:,:),0,1);
P_EV_SD_A = std(P_EV(:,:),0,1);
S_EV_SD_A = std(S_EV(:,:),0,1);
ANGL_SD_A = std(ANGL(:,:),0,1);
D_R_SD_A  = std(D_R(:,:),0,1);
V_X_SD_A  = std(V_X(:,:),0,1);
V_Y_SD_A  = std(V_Y(:,:),0,1);
V_Z_SD_A  = std(V_Z(:,:),0,1);



%% ========================================================================
%% Plots

mkdir('A_plots')
cd('A_plots')

plot_colormap=lines(p);

legend_info=cell(p+1,1);
for i=1:p
    legend_info{i}=sprintf('ROI %d',i);
end

legend_info{end}=sprintf('MEAN');

%-----------------------------------------------------------------------NEV
hnv=figure;
set(gcf,'Visible','off');

haxis1 = axes('Position',[0 0 1 1],'Visible','off');
haxis2 = axes('Position',[0.07 0.2 0.9 .75]);

    for i=1:p
        intData = interp1(1:q,squeeze(N_EV(i,:)),1:0.1:q,'spline');
        h=plot(squeeze(N_EV(i,:)),'.','Color',plot_colormap(i,:));
        set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
        hold on
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))
    end
 intData = interp1(1:q,N_EV_A,1:0.1:q,'spline');
 plot(1:0.1:q,intData,'k','LineWidth',3)
    
ylim([-1500 300])
xlim([0 q+1])
legend(legend_info);
xlabel('frame number')
title('Negative Eigen Value');

str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
set(gcf,'CurrentAxes',haxis1)
text(.3,0.05,str,'FontSize',12)

filename=sprintf('NEV.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);


hnv=figure;
set(gcf,'Visible','off');

    for i=1:p
        intData = interp1(1:q,squeeze(N_EV(i,:)),1:0.1:q,'spline');
        subplot(3,3,i)
        h=plot(squeeze(N_EV(i,:)),'.','Color',plot_colormap(i,:));
        tit=sprintf('NEV: ROI-%1d',i);
        title(tit)
        hold on
        subplot(3,3,i)
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))        
    end

subplot(3,3,[8,9])
axis off
str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
text(.1,0.5,str,'FontSize',12)

filename=sprintf('NEV-sub.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);


%-----------------------------------------------------------------------PEV
hnv=figure;
set(gcf,'Visible','off');

haxis1 = axes('Position',[0 0 1 1],'Visible','off');
haxis2 = axes('Position',[0.07 0.2 0.9 .75]);

    for i=1:p
        intData = interp1(1:q,squeeze(P_EV(i,:)),1:0.1:q,'spline');
        h=plot(squeeze(P_EV(i,:)),'.','Color',plot_colormap(i,:));
        set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
        hold on
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))
    end
    
intData = interp1(1:q,P_EV_A,1:0.1:q,'spline');
plot(1:0.1:q,intData,'k','LineWidth',3)
    
ylim([-300 1500])
xlim([0 q+1])
legend(legend_info);
xlabel('frame number')
title('Positive Eigen Value');

str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
set(gcf,'CurrentAxes',haxis1)
text(.3,0.05,str,'FontSize',12)

filename=sprintf('PEV.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);


hnv=figure;
set(gcf,'Visible','off');

    for i=1:p
        intData = interp1(1:q,squeeze(P_EV(i,:)),1:0.1:q,'spline');
        subplot(3,3,i)
        h=plot(squeeze(P_EV(i,:)),'.','Color',plot_colormap(i,:));
        tit=sprintf('PEV: ROI-%1d',i);
        title(tit)
        hold on
        subplot(3,3,i)
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))        
    end

subplot(3,3,[8,9])
axis off
str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
text(.1,0.5,str,'FontSize',12)

filename=sprintf('PEV-sub.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);


%-----------------------------------------------------------------------SEV
hnv=figure;
set(gcf,'Visible','off');

haxis1 = axes('Position',[0 0 1 1],'Visible','off');
haxis2 = axes('Position',[0.07 0.2 0.9 .75]);

    for i=1:p
        intData = interp1(1:q,squeeze(S_EV(i,:)),1:0.1:q,'spline');
        h=plot(squeeze(S_EV(i,:)),'.','Color',plot_colormap(i,:));
        set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
        hold on
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))
    end
    
intData = interp1(1:q,S_EV_A,1:0.1:q,'spline');
plot(1:0.1:q,intData,'k','LineWidth',3)
    
ylim([-1500 1500])
xlim([0 q+1])
legend(legend_info);
xlabel('frame number')
title('Sum Eigen Values');

str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
set(gcf,'CurrentAxes',haxis1)
text(.3,0.05,str,'FontSize',12)

filename=sprintf('SEV.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);



hnv=figure;
set(gcf,'Visible','off');

    for i=1:p
        intData = interp1(1:q,squeeze(S_EV(i,:)),1:0.1:q,'spline');
        subplot(3,3,i)
        h=plot(squeeze(S_EV(i,:)),'.','Color',plot_colormap(i,:));
        tit=sprintf('SEV: ROI-%1d',i);
        title(tit)
        hold on
        subplot(3,3,i)
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))        
    end

subplot(3,3,[8,9])
axis off
str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
text(.1,0.5,str,'FontSize',12)

filename=sprintf('SEV-sub.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);


%----------------------------------------------------------------------ANGL
hnv=figure;
set(gcf,'Visible','off');

haxis1 = axes('Position',[0 0 1 1],'Visible','off');
haxis2 = axes('Position',[0.07 0.2 0.9 .75]);

    for i=1:p
        intData = interp1(1:q,squeeze(ANGL(i,:)),1:0.1:q,'spline');
        h=plot(squeeze(ANGL(i,:)),'.','Color',plot_colormap(i,:));
        set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
        hold on
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))
    end
    
intData = interp1(1:q,ANGL_A,1:0.1:q,'spline');
plot(1:0.1:q,intData,'k','LineWidth',3)
    
ylim([-100 100])
xlim([0 q+1])
legend(legend_info);
xlabel('frame number')
title('Strain Rate Angle: Negative EV with +''X''');

str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
set(gcf,'CurrentAxes',haxis1)
text(.3,0.05,str,'FontSize',12)

filename=sprintf('ANGL.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);


hnv=figure;
set(gcf,'Visible','off');

    for i=1:p
        intData = interp1(1:q,squeeze(ANGL(i,:)),1:0.1:q,'spline');
        subplot(3,3,i)
        h=plot(squeeze(ANGL(i,:)),'.','Color',plot_colormap(i,:));
        tit=sprintf('SR Angle: ROI-%1d',i);
        title(tit)
        hold on
        subplot(3,3,i)
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))        
    end

subplot(3,3,[8,9])
axis off
str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
text(.1,0.5,str,'FontSize',12)

filename=sprintf('ANGL-sub.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);


%------------------------------------------------------------------------DR
hnv=figure;
set(gcf,'Visible','off');

haxis1 = axes('Position',[0 0 1 1],'Visible','off');
haxis2 = axes('Position',[0.07 0.2 0.9 .75]);

    for i=1:p
        intData = interp1(1:q,squeeze(D_R(i,:)),1:0.1:q,'spline');
        h=plot(squeeze(D_R(i,:)),'.','Color',plot_colormap(i,:));
        set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
        hold on
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))
    end
    
intData = interp1(1:q,D_R_A,1:0.1:q,'spline');
plot(1:0.1:q,intData,'k','LineWidth',3)
    
ylim([0 10])
xlim([0 q+1])
legend(legend_info);
xlabel('frame number')
title('Displacement');

str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
set(gcf,'CurrentAxes',haxis1)
text(.3,0.05,str,'FontSize',12)

filename=sprintf('DR.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);





hnv=figure;
set(gcf,'Visible','off');

    for i=1:p
        intData = interp1(1:q,squeeze(D_R(i,:)),1:0.1:q,'spline');
        subplot(3,3,i)
        h=plot(squeeze(D_R(i,:)),'.','Color',plot_colormap(i,:));
        tit=sprintf('DR: ROI-%1d',i);
        title(tit)
        hold on
        subplot(3,3,i)
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))        
    end

subplot(3,3,[8,9])
axis off
str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
text(.1,0.5,str,'FontSize',12)

filename=sprintf('DR-sub.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);



%------------------------------------------------------------------------VX
hnv=figure;
set(gcf,'Visible','off');

haxis1 = axes('Position',[0 0 1 1],'Visible','off');
haxis2 = axes('Position',[0.07 0.2 0.9 .75]);

    for i=1:p
        intData = interp1(1:q,squeeze(V_X(i,:)),1:0.1:q,'spline');
        h=plot(squeeze(V_X(i,:)),'.','Color',plot_colormap(i,:));
        set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
        hold on
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))
    end
    
intData = interp1(1:q,V_X_A,1:0.1:q,'spline');
plot(1:0.1:q,intData,'k','LineWidth',3)
    
ylim([-3 3])
xlim([0 q+1])
legend(legend_info);
xlabel('frame number')
title('Velocity V_x');

str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
set(gcf,'CurrentAxes',haxis1)
text(.3,0.05,str,'FontSize',12)

filename=sprintf('VX.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);



hnv=figure;
set(gcf,'Visible','off');

    for i=1:p
        intData = interp1(1:q,squeeze(V_X(i,:)),1:0.1:q,'spline');
        subplot(3,3,i)
        h=plot(squeeze(V_X(i,:)),'.','Color',plot_colormap(i,:));
        tit=sprintf('V_X: ROI-%1d',i);
        title(tit)
        hold on
        subplot(3,3,i)
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))        
    end

subplot(3,3,[8,9])
axis off
str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
text(.1,0.5,str,'FontSize',12)

filename=sprintf('VX-sub.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);









%------------------------------------------------------------------------VY
hnv=figure;
set(gcf,'Visible','off');

haxis1 = axes('Position',[0 0 1 1],'Visible','off');
haxis2 = axes('Position',[0.07 0.2 0.9 .75]);

    for i=1:p
        intData = interp1(1:q,squeeze(V_Y(i,:)),1:0.1:q,'spline');
        h=plot(squeeze(V_Y(i,:)),'.','Color',plot_colormap(i,:));
        set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
        hold on
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))
    end
    
intData = interp1(1:q,V_Y_A,1:0.1:q,'spline');
plot(1:0.1:q,intData,'k','LineWidth',3)
    
ylim([-3 3])
xlim([0 q+1])
legend(legend_info);
xlabel('frame number')
title('Velocity V_y');

str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
set(gcf,'CurrentAxes',haxis1)
text(.3,0.05,str,'FontSize',12)

filename=sprintf('VY.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);



hnv=figure;
set(gcf,'Visible','off');

    for i=1:p
        intData = interp1(1:q,squeeze(V_Y(i,:)),1:0.1:q,'spline');
        subplot(3,3,i)
        h=plot(squeeze(V_Y(i,:)),'.','Color',plot_colormap(i,:));
        tit=sprintf('VY: ROI-%1d',i);
        title(tit)
        hold on
        subplot(3,3,i)
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))        
    end

subplot(3,3,[8,9])
axis off
str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
text(.1,0.5,str,'FontSize',12)

filename=sprintf('VY-sub.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);



%------------------------------------------------------------------------VZ
hnv=figure;
set(gcf,'Visible','off');

haxis1 = axes('Position',[0 0 1 1],'Visible','off');
haxis2 = axes('Position',[0.07 0.2 0.9 .75]);

    for i=1:p
        intData = interp1(1:q,squeeze(V_Z(i,:)),1:0.1:q,'spline');
        h=plot(squeeze(V_Z(i,:)),'.','Color',plot_colormap(i,:));
        set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
        hold on
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))
    end
    
intData = interp1(1:q,V_Z_A,1:0.1:q,'spline');
plot(1:0.1:q,intData,'k','LineWidth',3)
    
ylim([-3 3])
xlim([0 q+1])
legend(legend_info);
xlabel('frame number')
title('Velocity V_z');

str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
set(gcf,'CurrentAxes',haxis1)
text(.3,0.05,str,'FontSize',12)

filename=sprintf('VZ.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);


hnv=figure;
set(gcf,'Visible','off');

    for i=1:p
        intData = interp1(1:q,squeeze(V_Z(i,:)),1:0.1:q,'spline');
        subplot(3,3,i)
        h=plot(squeeze(V_Z(i,:)),'.','Color',plot_colormap(i,:));
        tit=sprintf('V_Z: ROI-%1d',i);
        title(tit)
        hold on
        subplot(3,3,i)
        plot(1:0.1:q,intData,'Color',plot_colormap(i,:))        
    end

subplot(3,3,[8,9])
axis off
str(1) = {['Subject ID: ' PatientID]};
str(2) = {['Series name: ' Series_name]};
str(3) = {['Slice Location: ' num2str(SliceLocation)]};
text(.1,0.5,str,'FontSize',12)

filename=sprintf('VZ-sub.eps');
print(hnv,'-depsc2', '-r300', '-tiff', '-loose',filename);




%% ========================================================================

  
% %% Save data to spreadsheet  
    
cd ../
mkdir('A_results')
cd('A_results')


    xlsdata = 'results.xlsx'; % file name

    % Negative_EV
    sht = 'Negative_EV';
    xlwrite(xlsdata,{'ROI 1'},sht,'A1');
    xlwrite(xlsdata,{'ROI 2'},sht,'B1');
    xlwrite(xlsdata,{'ROI 3'},sht,'C1');
    xlwrite(xlsdata,{'ROI 4'},sht,'D1');
    xlwrite(xlsdata,{'ROI 5'},sht,'E1');
    xlwrite(xlsdata,{'ROI 6'},sht,'F1');
    xlwrite(xlsdata,{'ROI 7'},sht,'G1');
    
    xlwrite(xlsdata,{'Mean'},sht,'H1');
    xlwrite(xlsdata,{'Mean SD'},sht,'I1');
    
    xlwrite(xlsdata,transpose(squeeze(N_EV(1,:))),sht,'A2');
    xlwrite(xlsdata,transpose(squeeze(N_EV(2,:))),sht,'B2');
    xlwrite(xlsdata,transpose(squeeze(N_EV(3,:))),sht,'C2');
    xlwrite(xlsdata,transpose(squeeze(N_EV(4,:))),sht,'D2');
    xlwrite(xlsdata,transpose(squeeze(N_EV(5,:))),sht,'E2');
    xlwrite(xlsdata,transpose(squeeze(N_EV(6,:))),sht,'F2');
    xlwrite(xlsdata,transpose(squeeze(N_EV(7,:))),sht,'G2');
    xlwrite(xlsdata,transpose((squeeze(N_EV_A(:)))'),sht,'H2');
    xlwrite(xlsdata,transpose((squeeze(N_EV_SD_A(:)))'),sht,'I2');
    %----------------------------------------------------
    
    % Positive_EV
    sht = 'Positive_EV';
    xlwrite(xlsdata,{'ROI 1'},sht,'A1');
    xlwrite(xlsdata,{'ROI 2'},sht,'B1');
    xlwrite(xlsdata,{'ROI 3'},sht,'C1');
    xlwrite(xlsdata,{'ROI 4'},sht,'D1');
    xlwrite(xlsdata,{'ROI 5'},sht,'E1');
    xlwrite(xlsdata,{'ROI 6'},sht,'F1');
    xlwrite(xlsdata,{'ROI 7'},sht,'G1');
    
    xlwrite(xlsdata,{'Mean'},sht,'H1');
    xlwrite(xlsdata,{'Mean SD'},sht,'I1');
    
    xlwrite(xlsdata,transpose(squeeze(P_EV(1,:))),sht,'A2');
    xlwrite(xlsdata,transpose(squeeze(P_EV(2,:))),sht,'B2');
    xlwrite(xlsdata,transpose(squeeze(P_EV(3,:))),sht,'C2');
    xlwrite(xlsdata,transpose(squeeze(P_EV(4,:))),sht,'D2');
    xlwrite(xlsdata,transpose(squeeze(P_EV(5,:))),sht,'E2');
    xlwrite(xlsdata,transpose(squeeze(P_EV(6,:))),sht,'F2');
    xlwrite(xlsdata,transpose(squeeze(P_EV(7,:))),sht,'G2');
    xlwrite(xlsdata,transpose((squeeze(P_EV_A(:)))'),sht,'H2');
    xlwrite(xlsdata,transpose((squeeze(P_EV_SD_A(:)))'),sht,'I2');
    %----------------------------------------------------
    
    % Sum_EV
    sht = 'Sum_EV';
    xlwrite(xlsdata,{'ROI 1'},sht,'A1');
    xlwrite(xlsdata,{'ROI 2'},sht,'B1');
    xlwrite(xlsdata,{'ROI 3'},sht,'C1');
    xlwrite(xlsdata,{'ROI 4'},sht,'D1');
    xlwrite(xlsdata,{'ROI 5'},sht,'E1');
    xlwrite(xlsdata,{'ROI 6'},sht,'F1');
    xlwrite(xlsdata,{'ROI 7'},sht,'G1');
    
    xlwrite(xlsdata,{'Mean'},sht,'H1');
    xlwrite(xlsdata,{'Mean SD'},sht,'I1');
    
    xlwrite(xlsdata,transpose(squeeze(S_EV(1,:))),sht,'A2');
    xlwrite(xlsdata,transpose(squeeze(S_EV(2,:))),sht,'B2');
    xlwrite(xlsdata,transpose(squeeze(S_EV(3,:))),sht,'C2');
    xlwrite(xlsdata,transpose(squeeze(S_EV(4,:))),sht,'D2');
    xlwrite(xlsdata,transpose(squeeze(S_EV(5,:))),sht,'E2');
    xlwrite(xlsdata,transpose(squeeze(S_EV(6,:))),sht,'F2');
    xlwrite(xlsdata,transpose(squeeze(S_EV(7,:))),sht,'G2');
    xlwrite(xlsdata,transpose((squeeze(S_EV_A(:)))'),sht,'H2');
    xlwrite(xlsdata,transpose((squeeze(S_EV_SD_A(:)))'),sht,'I2');
    %----------------------------------------------------
    
    
    % Angle
    sht = 'Angle';
    xlwrite(xlsdata,{'ROI 1'},sht,'A1');
    xlwrite(xlsdata,{'ROI 2'},sht,'B1');
    xlwrite(xlsdata,{'ROI 3'},sht,'C1');
    xlwrite(xlsdata,{'ROI 4'},sht,'D1');
    xlwrite(xlsdata,{'ROI 5'},sht,'E1');
    xlwrite(xlsdata,{'ROI 6'},sht,'F1');
    xlwrite(xlsdata,{'ROI 7'},sht,'G1');
    
    xlwrite(xlsdata,{'Mean'},sht,'H1');
    xlwrite(xlsdata,{'Mean SD'},sht,'I1');
    
    xlwrite(xlsdata,transpose(squeeze(ANGL(1,:))),sht,'A2');
    xlwrite(xlsdata,transpose(squeeze(ANGL(2,:))),sht,'B2');
    xlwrite(xlsdata,transpose(squeeze(ANGL(3,:))),sht,'C2');
    xlwrite(xlsdata,transpose(squeeze(ANGL(4,:))),sht,'D2');
    xlwrite(xlsdata,transpose(squeeze(ANGL(5,:))),sht,'E2');
    xlwrite(xlsdata,transpose(squeeze(ANGL(6,:))),sht,'F2');
    xlwrite(xlsdata,transpose(squeeze(ANGL(7,:))),sht,'G2');
    xlwrite(xlsdata,transpose((squeeze(ANGL_A(:)))'),sht,'H2');
    xlwrite(xlsdata,transpose((squeeze(ANGL_SD_A(:)))'),sht,'I2');
    %----------------------------------------------------
        
    
    % Displacement
    sht = 'Displacement';
    xlwrite(xlsdata,{'ROI 1'},sht,'A1');
    xlwrite(xlsdata,{'ROI 2'},sht,'B1');
    xlwrite(xlsdata,{'ROI 3'},sht,'C1');
    xlwrite(xlsdata,{'ROI 4'},sht,'D1');
    xlwrite(xlsdata,{'ROI 5'},sht,'E1');
    xlwrite(xlsdata,{'ROI 6'},sht,'F1');
    xlwrite(xlsdata,{'ROI 7'},sht,'G1');
    
    xlwrite(xlsdata,{'Mean'},sht,'H1');
    xlwrite(xlsdata,{'Mean SD'},sht,'I1');
    
    xlwrite(xlsdata,transpose(squeeze(D_R(1,:))),sht,'A2');
    xlwrite(xlsdata,transpose(squeeze(D_R(2,:))),sht,'B2');
    xlwrite(xlsdata,transpose(squeeze(D_R(3,:))),sht,'C2');
    xlwrite(xlsdata,transpose(squeeze(D_R(4,:))),sht,'D2');
    xlwrite(xlsdata,transpose(squeeze(D_R(5,:))),sht,'E2');
    xlwrite(xlsdata,transpose(squeeze(D_R(6,:))),sht,'F2');
    xlwrite(xlsdata,transpose(squeeze(D_R(7,:))),sht,'G2');
    xlwrite(xlsdata,transpose((squeeze(D_R_A(:)))'),sht,'H2');
    xlwrite(xlsdata,transpose((squeeze(D_R_SD_A(:)))'),sht,'I2');
    %----------------------------------------------------
  
    % VX
    sht = 'Vx';
    xlwrite(xlsdata,{'ROI 1'},sht,'A1');
    xlwrite(xlsdata,{'ROI 2'},sht,'B1');
    xlwrite(xlsdata,{'ROI 3'},sht,'C1');
    xlwrite(xlsdata,{'ROI 4'},sht,'D1');
    xlwrite(xlsdata,{'ROI 5'},sht,'E1');
    xlwrite(xlsdata,{'ROI 6'},sht,'F1');
    xlwrite(xlsdata,{'ROI 7'},sht,'G1');
    
    xlwrite(xlsdata,{'Mean'},sht,'H1');
    xlwrite(xlsdata,{'Mean SD'},sht,'I1');
    
    xlwrite(xlsdata,transpose(squeeze(V_X(1,:))),sht,'A2');
    xlwrite(xlsdata,transpose(squeeze(V_X(2,:))),sht,'B2');
    xlwrite(xlsdata,transpose(squeeze(V_X(3,:))),sht,'C2');
    xlwrite(xlsdata,transpose(squeeze(V_X(4,:))),sht,'D2');
    xlwrite(xlsdata,transpose(squeeze(V_X(5,:))),sht,'E2');
    xlwrite(xlsdata,transpose(squeeze(V_X(6,:))),sht,'F2');
    xlwrite(xlsdata,transpose(squeeze(V_X(7,:))),sht,'G2');
    xlwrite(xlsdata,transpose((squeeze(V_X_A(:)))'),sht,'H2');
    xlwrite(xlsdata,transpose((squeeze(V_X_SD_A(:)))'),sht,'I2');
    %----------------------------------------------------
    
    % VY
    sht = 'Vy';
    xlwrite(xlsdata,{'ROI 1'},sht,'A1');
    xlwrite(xlsdata,{'ROI 2'},sht,'B1');
    xlwrite(xlsdata,{'ROI 3'},sht,'C1');
    xlwrite(xlsdata,{'ROI 4'},sht,'D1');
    xlwrite(xlsdata,{'ROI 5'},sht,'E1');
    xlwrite(xlsdata,{'ROI 6'},sht,'F1');
    xlwrite(xlsdata,{'ROI 7'},sht,'G1');
    
    xlwrite(xlsdata,{'Mean'},sht,'H1');
    xlwrite(xlsdata,{'Mean SD'},sht,'I1');
    
    xlwrite(xlsdata,transpose(squeeze(V_Y(1,:))),sht,'A2');
    xlwrite(xlsdata,transpose(squeeze(V_Y(2,:))),sht,'B2');
    xlwrite(xlsdata,transpose(squeeze(V_Y(3,:))),sht,'C2');
    xlwrite(xlsdata,transpose(squeeze(V_Y(4,:))),sht,'D2');
    xlwrite(xlsdata,transpose(squeeze(V_Y(5,:))),sht,'E2');
    xlwrite(xlsdata,transpose(squeeze(V_Y(6,:))),sht,'F2');
    xlwrite(xlsdata,transpose(squeeze(V_Y(7,:))),sht,'G2');
    xlwrite(xlsdata,transpose((squeeze(V_Y_A(:)))'),sht,'H2');
    xlwrite(xlsdata,transpose((squeeze(V_Y_SD_A(:)))'),sht,'I2');
    %----------------------------------------------------% VX
    
    % VZ
    sht = 'Vz';
    xlwrite(xlsdata,{'ROI 1'},sht,'A1');
    xlwrite(xlsdata,{'ROI 2'},sht,'B1');
    xlwrite(xlsdata,{'ROI 3'},sht,'C1');
    xlwrite(xlsdata,{'ROI 4'},sht,'D1');
    xlwrite(xlsdata,{'ROI 5'},sht,'E1');
    xlwrite(xlsdata,{'ROI 6'},sht,'F1');
    xlwrite(xlsdata,{'ROI 7'},sht,'G1');
    
    xlwrite(xlsdata,{'Mean'},sht,'H1');
    xlwrite(xlsdata,{'Mean SD'},sht,'I1');
    
    xlwrite(xlsdata,transpose(squeeze(V_Z(1,:))),sht,'A2');
    xlwrite(xlsdata,transpose(squeeze(V_Z(2,:))),sht,'B2');
    xlwrite(xlsdata,transpose(squeeze(V_Z(3,:))),sht,'C2');
    xlwrite(xlsdata,transpose(squeeze(V_Z(4,:))),sht,'D2');
    xlwrite(xlsdata,transpose(squeeze(V_Z(5,:))),sht,'E2');
    xlwrite(xlsdata,transpose(squeeze(V_Z(6,:))),sht,'F2');
    xlwrite(xlsdata,transpose(squeeze(V_Z(7,:))),sht,'G2');
    xlwrite(xlsdata,transpose((squeeze(V_Z_A(:)))'),sht,'H2');
    xlwrite(xlsdata,transpose((squeeze(V_Z_SD_A(:)))'),sht,'I2');
    %----------------------------------------------------
    
    
%save to mat file    
filename='results.mat';

%% save variables as mat file
save (filename,'N_EV')
save (filename,'P_EV','-append')
save (filename,'S_EV','-append')
save (filename,'P_EV','-append')
save (filename,'ANGL','-append')
save (filename,'D_R','-append')
save (filename,'V_X','-append')
save (filename,'V_Y','-append')
save (filename,'V_Z','-append')
save (filename,'V_Z','-append')
save (filename,'PatientID','-append')
save (filename,'Series_name','-append')
save (filename,'SliceLocation','-append')
save (filename,'force_mean','-append')

%%
%clear all
display('Analysis is completed!')

