[X Y Z] = meshgrid( [-Npix/2:Npix/2-1]*d2_bragg, ...
    [-Npix/2:Npix/2-1]*d2_bragg,...
    [-depth/2:depth/2-1]*d2_bragg);


[Xs Ys Zs] = meshgrid( [-Npixsamp/2:Npixsamp/2-1]*d2_bragg, ...
    [-Npixsamp/2:Npixsamp/2-1]*d2_bragg,...
    [-depth/2:depth/2-1]*d2_bragg);

obj = zeros(size(X));

ins = (Xs < grainwidth/2 & Xs > -grainwidth/2 ...
    & Ys < grainheight/2 & Ys > -grainheight/2 ...
    & Zs < filmthickness/2 & Zs > -filmthickness/2);
ins = find(ins>0); %make it a list from a boolean map

sup3d = zeros(size(Xs));
sup3d(ins) = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change this for different samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CDW_makestrain_4; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%make phases by doing dot product with Q vectors. 
%for ii = 1:nBraggpeaks   
%    HKL = Braggpeaklist(ii,:);
%    rho(ii).A = exp(i*2*pi*(HKL(1)*(Ux/lattice(1)) + HKL(2)*(Uy/lattice(2)) + HKL(3)*(Uz/lattice(3))));
%    rho(ii).A = rho(ii).A .* sup3d;
%end

