%=========================================================================
%   Step1 for 2d Strain Analysis for Pelvis
%
%       part of 2d Strain Rate Toolkit
%=========================================================================
%
% INput:    1)  force recordings: mvc.csv and corresponding s*.csv
%           2)  *.dcm images
%
% OUTput:   1) sorted images mri_data.mat
%           2) ev_calculated.mat (ev decomposition volumes)
%           3) force analysis
%_____________________________________________________
% required subroutines:
%
%   1)vepc_2d_imsort
%   2)force_data
%   3)im_subtract
%   4)im_mean
%   5)anisodiff2D
%   6)multiWaitbar
%_____________________________________________________
% 
% written by Vadim Malis
% 02/15 at UCSD RIL
%==========================================================================

bpm=30;
force_samplerate=200;


window_info=sprintf('pick root data folder');
PathName = uigetdir('~/Desktop',window_info);
cd(PathName)

dicom_path=dir('*dcm');


%% Read original images and sort

im_data=vepc_2d_imsort(dicom_path);
load(im_data);

%% ------------------------------------------------------------------------
% % Force analysis
% path=pwd;
% force_filename=force_analysis(path,numphases,bpm,0);
% load(force_filename);

%% ------------------------------------------------------------------------
% Calculating SR and angle maps

% Magnitude image file 
magimage1 = im_m(:,:,1);

%   Allocate the gradient image buffers
%   Create 4 matrices to store the EigenValues of Every Voxel
f1 = zeros(size(v_ap),'single');
f2 = zeros(size(v_ap),'single');
f3 = zeros(size(v_ap),'single');
f4 = zeros(size(v_ap),'single');

%   Allocate arrays for filtered velocities
v_rl_sm=zeros(size(v_rl));
v_ap_sm=zeros(size(v_rl));
v_si_sm=zeros(size(v_rl));

%---------2D AnisotropicDiffusion Filter-----------------------------------

        num_iter=10; kappa=2; option=1; delta_t=1/7;
        % calling the anisodiff2d code; options are set above; 
        % less smoothing with lower iterations and smaller kappa.
       
        multiWaitbar('Filtering...', 0, 'Color', 'g');
        
for i=1:numphases;
         
        v_rl_sm(:,:,i) = anisodiff2D(v_rl(:,:,i),...
            im_m(:,:,i),num_iter,delta_t, kappa,option);
        v_ap_sm(:,:,i) = anisodiff2D(v_ap(:,:,i),...
            im_m(:,:,i),num_iter,delta_t, kappa,option);
        v_si_sm(:,:,i) = anisodiff2D(v_si(:,:,i),...
            im_m(:,:,i),num_iter,delta_t, kappa,option);
        
%----------Spatial derivative----------------------------------------------
%   2D_Strain Tensor    
         [f1(:,:,i), f2(:,:,i)] = gradient(v_ap_sm(:,:,i));
         [f3(:,:,i), f4(:,:,i)] = gradient(v_si_sm(:,:,i));
         multiWaitbar('Filtering...', i/numphases);      
end
     
multiWaitbar('Filtering...', 'Close');  

%% ------------------------------------------------------------------------
% Prealocation of variables for Eigen Value decomposition

% Create a Matrix to store the 4 components of the StrainTensor of every
% [STxx,STxy,STyx, STyy]
StrainT = zeros([size(v_ap) 4],'single');

% Create 2 matrices to store the EigenValues of every Voxel
Y_1 = zeros(size(v_ap),'single');
Y_2 = zeros(size(v_ap) ,'single');
Y_sqrt = zeros(size(v_ap) ,'single');
Y_sum = zeros(size(v_ap) ,'single');

% Create a maxtrix to store the (main) strain direction in each pixel
VectorF_red=zeros([size(v_ap) 2],'single');
VectorF_blue=zeros([size(v_ap) 2],'single');

%
angle=zeros(size(v_ap),'single');
x1angle=zeros(size(v_ap),'single');
y1angle=zeros(size(v_ap),'single');
x2angle=zeros(size(v_ap),'single');
y2angle=zeros(size(v_ap),'single');
angle_1=zeros(size(v_ap),'single');

%% ------------------------------------------------------------------------
% Eigenvalue problem CALCS

multiWaitbar('EV decomposition...', 0, 'Color', 'g');


A=size(v_ap,1);
B=size(v_ap,2);
C=size(v_ap,3);

for a=1:A
    for b=1:B
        for c=1:C
            
%             Calculating the Strain Tensor, Transpose, Eigenvectors and
%             values at each voxel, Sorting and Ordering Eigenvalues from
%             samllest to largest (Normalizing each gradient by its voxel
%             dimensions as well)

            if (magimage1(a,b))>50
                
            StrainTensor=[(f1(a,b,c))*(10/1.17) (f2(a,b,c))*(10/1.17);(f3(a,b,c))*(10/1.17) (f4(a,b,c))*(10/1.17)];
            StrainTensorTrans=[(f1(a,b,c))*(10/1.17) (f3(a,b,c))*(10/1.17);(f2(a,b,c))*(10/1.17) (f4(a,b,c))*(10/1.17)];
            L=0.5*(StrainTensor+StrainTensorTrans);
            [EigenVectors,S]=eig(L);EigenValues=diag(S);
            [t,index]=sort(EigenValues);
            
%Always second EigenValue is positive
            EigenValues=EigenValues(index);            
            EigenVectors = EigenVectors(:,index);   
            Y_1(a,b,c)=EigenValues(1)*1000;
            Y_2(a,b,c)=EigenValues(2)*1000;
            Y_sqrt(a,b,c)=sqrt(power(EigenValues(1),2)+power(EigenValues(2),2))*1000;
            Y_sum(a,b,c) = Y_1(a,b,c)+ Y_2(a,b,c);
       
%Bring all EigenVectors to first and second quadrant and calculate NEV
%angle

            %NEV
            if((EigenVectors(1,1)>0 && EigenVectors(2,1)>0)||(EigenVectors(1,1)<0 && EigenVectors(2,1)>0))
            VectorF_red(a,b,c,:)= EigenVectors(:,1);
            x1angle(a,b,c) = (180/pi)*acos(EigenVectors(1,1));
            elseif((EigenVectors(1,1)>0&&EigenVectors(2,1)<0)||(EigenVectors(1,1)<0 && EigenVectors(2,1)<0))
            VectorF_red(a,b,c,:)= -EigenVectors(:,1);
            x1angle(a,b,c) = 180-((180/pi)*acos(EigenVectors(1,1)));
            else
            end


            %PEV
            if((EigenVectors(1,2)>0 && EigenVectors(2,2)>0)||(EigenVectors(1,2)<0 && EigenVectors(2,2)>0))
            VectorF_blue(a,b,c,:)= EigenVectors(:,2);
            elseif((EigenVectors(1,2)>0 && EigenVectors(2,2)<0)||(EigenVectors(1,2)<0 && EigenVectors(2,2)<0))
            VectorF_blue(a,b,c,:)= -EigenVectors(:,2);
            else
            end
   
            end
        end
    end
    
    multiWaitbar('EV decomposition...', a/A);
    
end

 multiWaitbar('EV decomposition...', 'Close');

%Angle to 1st and 4th quadrant (NEV with +x)
x1angle(x1angle>90)=(x1angle(x1angle>90)-180);
x1angle=(-1)*x1angle;

%% ------------------------------------------------------------------------
% Saving calculated volumes


EV_Negative=Y_1;
EV_Positive=Y_2;
EV_Sum=Y_sum;
SR_Angle=x1angle;

% save ('ev_calculated_data.mat','EV_Negative')
% save ('ev_calculated_data.mat','EV_Positive','-append')
% save ('ev_calculated_data.mat','EV_Sum','-append')
% save ('ev_calculated_data.mat','SR_Angle','-append')
% save ('ev_calculated_data.mat','PatientID','-append')
% save ('ev_calculated_data.mat','Series_name','-append')
% save ('ev_calculated_data.mat','RotationMatrix','-append')
% 
% 

fid = fopen('negative_eig.dat','w+','l');
Y_1 = permute(Y_1,[2 1 3]);
fwrite(fid,Y_1,'float');
fclose(fid);
% 
fid = fopen('positive_eig.dat','w+','l');
Y_2 = permute(Y_2,[2 1 3]);
fwrite(fid,Y_2,'float');
fclose(fid);
% 
fid = fopen('sum_eig.dat','w+','l');
Y_sum = permute(Y_sum,[2 1 3]);
fwrite(fid,Y_sum,'float');
fclose(fid);
% 
fid = fopen('angle.dat','w+','l');
ANGLE = permute(x1angle,[2 1 3]);
fwrite(fid,ANGLE,'float');
fclose(fid);

%save all workspace data
save ('ev_calculated_data.mat')
