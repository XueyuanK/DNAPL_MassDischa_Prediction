% transform from 14*8*8 to 64*38*32 (the input of the NN)
clc
clear 
[K]=textread('3dsandbox.out','%f','headerlines',3);
K=reshape(K(1:89600),14,8,8,100);
K=permute(K,[4 1 2 3]);%size=100*14*8*8
%% upscaling
K_new=repmat(K,[1,4,4,4]);% size = (100,56,32,32)
K_new=reshape(K_new,100,14,4,8,4,8,4);
K_new=permute(K_new,[1 3 2 5 4 7 6]);
K_new=reshape(K_new,100,56,32,32);

K_new=cat(2,K_new(:,1:7,:,:),K_new(:,7,:,:),K_new(:,8:21,:,:),K_new(:,21,:,:),K_new(:,22:35,:,:),K_new(:,35,:,:),K_new(:,36:49,:,:),K_new(:,49,:,:),K_new(:,50:end,:,:));%size=100*60*32*32
K_new=cat(3,K_new(:,:,1:4,:),K_new(:,:,4,:),K_new(:,:,5:12,:),K_new(:,:,12,:),K_new(:,:,13:20,:),K_new(:,:,20,:),K_new(:,:,21:28,:),K_new(:,:,28,:),K_new(:,:,29:end,:));%size=100*60*36*32

K_new=cat(2,ones(100,2,36,32)*5,K_new,ones(100,2,36,32)*5);%size=100*64*36*32
K_new=cat(3,ones(100,64,2,32)*5,K_new,ones(100,64,2,32)*5);%size=100*64*40*32
%% plot the original and the transformed sample 
x_grid=[0:1:14];
z_grid=[0:1:8];
for i=1:8
K_plot=K(1,:,:,i);
figure(1),subplot(4,2,i),imagesc(x_grid,z_grid,reshape(K_plot,14,8)');
set(gca,'YDir','normal'); 
pbaspect([14 8 1])  % the ratio between x y z axis 
end

% the transformaed image
x_grid=[0:1:64];
z_grid=[0:1:40];
for i =1:8
K_plot=K_new(1,:,:,4*(i-1)+1);
figure(2),subplot(4,2,i),imagesc(x_grid,z_grid,reshape(K_plot,64,40)');
set(gca,'YDir','normal'); 
pbaspect([64 40 1])  % the ratio between x y z axis 
end

saveas( 1, 'original.jpg');
saveas( 2, 'transformed.jpg');
save('K100_64_40_32.mat','K_new','-v7.3')
