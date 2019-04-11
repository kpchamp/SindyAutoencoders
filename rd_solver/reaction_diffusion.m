clear all; close all; clc

% lambda-omega reaction-diffusion system
%  u_t = lam(A) u - ome(A) v + d1*(u_xx + u_yy) = 0
%  v_t = ome(A) u + lam(A) v + d2*(v_xx + v_yy) = 0
%
%  A^2 = u^2 + v^2 and
%  lam(A) = 1 - A^2
%  ome(A) = -beta*A^2


t=0:0.05:500;
d1=0.1; d2=0.1; beta=1.0;
L=20; n=100; N=n*n;
x2=linspace(-L/2,L/2,n+1); x=x2(1:n); y=x;
kx=(2*pi/L)*[0:(n/2-1) -n/2:-1]; ky=kx;

% INITIAL CONDITIONS

[X,Y]=meshgrid(x,y);
[KX,KY]=meshgrid(kx,ky);
K2=KX.^2+KY.^2; K22=reshape(K2,N,1);

m=1; % number of spirals

f = exp(-.01*(X.^2+Y.^2)); % circular mask to get rid of boundaries

u = zeros(length(x),length(y),length(t));
v = zeros(length(x),length(y),length(t));
uf = zeros(length(x), length(y), length(t));
vf = zeros(length(x), length(y), length(t));

du = zeros(length(x),length(y),length(t));
dv = zeros(length(x),length(y),length(t));
duf = zeros(length(x),length(y),length(t));
dvf = zeros(length(x),length(y),length(t));

u(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*cos(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));
v(:,:,1)=tanh(sqrt(X.^2+Y.^2)).*sin(m*angle(X+i*Y)-(sqrt(X.^2+Y.^2)));
uf(:,:,1)=f.*u(:,:,1);
vf(:,:,1)=f.*v(:,:,1);

% REACTION-DIFFUSION
uvt=[reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
uvt_rhs = reaction_diffusion_rhs(t(1),uvt,[],K22,d1,d2,beta,n,N);
du(:,:,1) = real(ifft2(reshape(uvt_rhs(1:N).',n,n)));
dv(:,:,1)= real(ifft2(reshape(uvt_rhs((N+1):(2*N)).',n,n)));

[t,uvsol]=ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);

for j=1:length(t)-1
ut=reshape((uvsol(j,1:N).'),n,n);
vt=reshape((uvsol(j,(N+1):(2*N)).'),n,n);
u(:,:,j+1)=real(ifft2(ut));
v(:,:,j+1)=real(ifft2(vt));

uvt_rhs = reaction_diffusion_rhs(t(j+1),uvsol(j,1:end).',[],K22,d1,d2,beta,n,N);
du(:,:,j+1)= real(ifft2(reshape(uvt_rhs(1:N).',n,n)));
dv(:,:,j+1)= real(ifft2(reshape(uvt_rhs((N+1):(2*N)).',n,n)));

uf(:,:,j+1)=f.*u(:,:,j+1);
vf(:,:,j+1)=f.*v(:,:,j+1);
duf(:,:,j+1)=f.*du(:,:,j+1);
dvf(:,:,j+1)=f.*dv(:,:,j+1);

% figure(1)
% pcolor(x,y,v(:,:,j+1)); shading interp; colormap(hot); colorbar; drawnow; 
end

t = t(3:end);
uf = uf(:,:,3:end);
vf = vf(:,:,3:end);
duf = duf(:,:,3:end);
dvf = dvf(:,:,3:end);

save('reaction_diffusion.mat','t','x','y','uf','vf','duf','dvf')

%%
% load reaction_diffusion_big
% pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)


