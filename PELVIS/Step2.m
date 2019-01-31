%=========================================================================
%   Step2 for 2d Strain Analysis for Pelvis
%
%       part of 2d Strain Rate Toolkit
%=========================================================================
%
% INput: (output of Step
%
%           2)  ev_calculated_data.mat
%
% OUTput:   1) displacement report *.png
%_____________________________________________________
% required subroutines:
%           1) freezeColors
%           2) subplottight
%           3) multiwaitbar
%_____________________________________________________
% 
% written by Vadim Malis
% 02/15 at UCSD RIL
%==========================================================================

PathName = uigetdir('choose data folder','~/Desktop');

cd(PathName)
load('ev_calculated_data.mat','ANGLE','VectorF_blue','VectorF_red',...
    'Y_1','Y_2','Series_name','SliceLocation','dt','force_mean',...
    'force_ref','im_m','numphases','resolution','v_rl','v_ap','v_si',...
    'v_rl_sm','v_ap_sm','v_si_sm','PatientID')



V_R=sqrt(v_rl.^2+v_ap.^2+v_si.^2); %maginitude of the velocity
screensize = get( groot, 'Screensize' );

%% calculate vector magnitudes
Lambda1=permute(Y_1,[2,1,3]);Lambda1(~Lambda1)=NaN;     %negative EV
Lambda2=permute(Y_2,[2,1,3]);Lambda2(~Lambda2)=NaN;     %positive EV

NEV_u=Lambda1.*squeeze(VectorF_red(:,:,:,1));
NEV_v=Lambda1.*squeeze(VectorF_red(:,:,:,2));
PEV_u=Lambda2.*squeeze(VectorF_blue(:,:,:,1));
PEV_v=Lambda2.*squeeze(VectorF_blue(:,:,:,2));


%% get reference frame: force and min velocity aproaches (force is default)

%force_ref is loaded above   reference frame for lowes force in the cycle
force=resample(force_mean,numphases*10,size(force_mean,2));


% %min velocity
% V_R_mean=squeeze(mean(mean(V_R,1),2));
% V_R_std=squeeze(std(std(V_R,1,1),1,2));
% [~,velocity_ref]=min(V_R_mean+V_R_std);


%% Select main ROI
figure
imshow(mat2gray(im_m(:,:,force_ref)),'InitialMagnification', 200)
roi_size=75;
image_crop=imrect(gca,[100,100,roi_size,roi_size]);
%setResizable(image_crop,0)
wait(image_crop);

image_crop_pos=getPosition(image_crop);

CROP_X=int16([image_crop_pos(1),image_crop_pos(1)+image_crop_pos(3)-1]);
CROP_Y=int16([image_crop_pos(2),image_crop_pos(2)+image_crop_pos(4)-1]);

close

%% crop all volumes
ANGLE_cr    =   ANGLE(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
NEV_u_cr    =   NEV_u(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
NEV_v_cr    =   NEV_v(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
PEV_u_cr    =   PEV_u(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
PEV_v_cr    =   PEV_v(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
V_R_cr      =   V_R(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
v_rl_cr     =   v_rl(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
v_si_cr     =   v_si(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
v_ap_cr     =   v_ap(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),:);
IMAGE       =   zeros(size(v_ap_cr));

for i=1:numphases
    IMAGE(:,:,i) = mat2gray(im_m(CROP_Y(1):CROP_Y(2),CROP_X(1):CROP_X(2),i));
end


% [~,z_nev]=cart2pol(NEV_u,NEV_v);
% [~,z_pev]=cart2pol(PEV_u,PEV_v);

[X,Y]=meshgrid(1:CROP_X(2)-CROP_X(1)+1,1:CROP_Y(2)-CROP_Y(1)+1);



%% select ROI and calculate displacement

%get magnification for perfect fit
magn=floor(100*(double(screensize(4))/double((CROP_Y(2)-CROP_Y(1)))));

%ROI
figure
I=squeeze(IMAGE(:,:,force_ref));

imshow(I,'InitialMagnification',magn,'Border','tight');
h = imellipse;
wait(h);
mask   = createMask(h);
[row,col] = find(mask);

for i=1:size(row,1)
    if I(row(i),col(i))<0.1
        row(i)=NaN;
        col(i)=NaN;
    end
end

row(isnan(row))=[];
col(isnan(col))=[];

close


%tracking
[xs,ys] = track2dv4(col,row,v_rl_cr*-1,v_si_cr,dt,resolution,force_ref);


%% calculate displacement
delta_x=bsxfun(@minus,xs,xs(:,force_ref));
delta_y=bsxfun(@minus,ys,ys(:,force_ref));
delta_r=sqrt(delta_x.^2+delta_y.^2);

delta_r_image=zeros(size(IMAGE));

for i=1:numphases
    for j=1:size(delta_r,1)
        delta_r_image(row(j),col(j),i)=delta_r(j,i)*10/resolution;
    end
end

delta_r_image_alpha=delta_r_image(:,:,1);
delta_r_image_alpha(delta_r_image_alpha~=0)=0.2;



%% visualisation

mkdir('A_DRmap')
cd('A_DRmap')

% define colormap
cmax=0.85*max(delta_r_image(:));
cmin=min(delta_r_image(:));

% max y for force plot
y_max=max(force(:));

%SR Units
units_nev = texlabel('-10^3 sec^(-1)');
units_pev = texlabel('10^3 sec^(-1)');

I=mat2gray(I);

%%video tracking
% figure
% vid = VideoWriter('video_roi.avi');
% vid.Quality=100;
% open(vid);

hfig=figure('Visible','off');

multiWaitbar('Processing...', 0, 'Color', 'g');

for i=1:numphases
   
    
    %Tracking
    h1=subplottight(2,4,1);
    imshow(IMAGE(:,:,i),'InitialMagnification',magn,'border','tight');
    hold on
    freezeColors
    plot(xs(:,i),ys(:,i),'y.','markersize', 2);
    title('Tracked Voxels');
    
    %Displacement
    h2=subplottight(2,4,2);
    imshow(I,'InitialMagnification',magn,'border','tight');
    hold on;
    freezeColors
    h=imshow(delta_r_image(:,:,i));
    title('Displacement');
    colormap jet
    caxis([cmin cmax]);
    set(h, 'AlphaData', delta_r_image_alpha);
    cbar=colorbar('Southoutside');
    set(cbar,'Position',[.27 .57 .21 .01]);
    cbar.Label.String = '[mm]';
    
    %Strain Rate Negative
    h3=subplottight(2,4,3);
    imshow(IMAGE(:,:,i),'border','tight');
    hold on
    freezeColors
    quiver_color(X,Y,NEV_u_cr(:,:,i),NEV_v_cr(:,:,i),0.001:1800,units_nev,0);
    title('Strain Rate: Negative')
    cbar2=colorbar('Southoutside');
    set(cbar2,'Position',[.52 .57 .21 .01]);
    caxis([0.001,1700])
    set(cbar2,'ylim',[0,1800]);
    set(cbar2,'YTick',[0,600,1200,1800]);
    cbar2.Label.String = '-10^3 [sec^-^1]'; 
    hold off
    
    %Strain Rate Positive
    h4=subplottight(2,4,4);
    imshow(IMAGE(:,:,i),'border','tight');
    hold on
    freezeColors
    quiver_color(X,Y,PEV_u_cr(:,:,i),PEV_v_cr(:,:,i),0.001:1700,units_pev,0);
    title('Strain Rate: Positive')
    cbar3=colorbar('Southoutside');
    set(cbar3,'Position',[.77 .57 .21 .01]);
    caxis([0.001,1700])
    set(cbar3,'ylim',[0,1800]);
    set(cbar3,'YTick',[0,600,1200,1800]);
    cbar3.Label.String = '10^3 [sec^-^1]';
    hold off
    
    %Velocity X 
    h5=subplottight(2,4,5);
    imshow(abs(v_rl_cr(:,:,i)),'InitialMagnification',magn,'border','tight');
    colormap jet
    caxis([0 5]);
    hold on
    text(1,roi_size-3,'V_x','Color','w','FontSize',12,'FontWeight','bold')
    plot([roi_size,roi_size],[0,roi_size],'k-','LineWidth',1)
    
    %Velocity Y 
    h6=subplottight(2,4,6);
    imshow(abs(v_si_cr(:,:,i)),'InitialMagnification',magn,'border','tight');
    colormap jet
    caxis([0 5]);
    cbar4=colorbar('Southoutside');
    set(cbar4,'Position',[.125 .42 .5 .01]);
    cbar4.Label.String = '[cm/sec]';
    hold on
    text(1,roi_size-3,'V_y','Color','w','FontSize',12,'FontWeight','bold')
    plot([roi_size,roi_size],[0,roi_size],'k-','LineWidth',1)
    
    %Velocity Z 
    h7=subplottight(2,4,7);
    imshow(abs(v_ap_cr(:,:,i)),'InitialMagnification',magn,'border','tight');
    colormap jet
    caxis([0 5]);
    hold on
    text(1,roi_size-3,'V_z','Color','w','FontSize',12,'FontWeight','bold')
    
    %Force & INFO
    h8=subplottight(2,4,8);
    plot(force(1:i*10),'b-','LineWidth',1)
    hold on
    plot([i*10,i*10],[0,y_max*1.1],'r-','LineWidth',.2);
    plot([(i-1)*10,(i-1)*10],[0,y_max*1.1],'r-','LineWidth',.2);
    plot([0,numphases*(10)],[y_max*1.1,y_max*1.1],'k-','LineWidth',.5);
    xlim([0,numphases*(10)]);
    ylim([0,y_max*3]);  
    info=sprintf('Patient ID: %s\nSeries: %s\n\nSlice Location: %.2f\nMax Force: %.2f\n\nCurrent Frame: %d\nReference Frame: %d\n\n                          Force',PatientID,Series_name,SliceLocation,y_max,i,force_ref);
    text(10,y_max*2,info,'Interpreter', 'none','FontSize',8);
    set(gca,'XTick',[],'YTick',[]);
    axis square
    hold off
    
    export_fig(sprintf('report-%d.png', i),'-m4','-png');
    hold off
    
    cla(h1)
    cla(h2)
    cla(h3)
    cla(h4)
    cla(h5)
    cla(h6)
    cla(h7)
    cla(h8)
    clear cbar cbar2 cbar3 cbar4
    clf(hfig,'reset')
%     frame = getframe;
%     writeVideo(vid,frame);
    set(hfig,'visible','off')

multiWaitbar('Processing...', i/numphases);

end

% close(vid);

close

multiWaitbar('Processing...', 'Close');  



