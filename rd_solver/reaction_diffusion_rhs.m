function rhs=reaction_diffusion_rhs(t,uvt,dummy,K22,d1,d2,beta,n,N);

% Calculate u and v terms
ut=reshape((uvt(1:N)),n,n);
vt=reshape((uvt((N+1):(2*N))),n,n);
u=real(ifft2(ut)); v=real(ifft2(vt));

% Reaction Terms
u3=u.^3; v3=v.^3; u2v=(u.^2).*v; uv2=u.*(v.^2);
utrhs=reshape((fft2(u-u3-uv2+beta*u2v+beta*v3)),N,1);
vtrhs=reshape((fft2(v-u2v-v3-beta*u3-beta*uv2)),N,1);

rhs=[-d1*K22.*uvt(1:N)+utrhs
     -d2*K22.*uvt(N+1:end)+vtrhs];

