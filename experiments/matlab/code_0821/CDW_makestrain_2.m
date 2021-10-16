%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change this for different samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% randomly oriented CDW in plane
lam_cdw = 0.05; 	% CDW wavelength in microns
mag = 1.e-3;		% ampliitude of CDW

th = pi * ( -1 + 2*rand );	% random in plane orientation for CDW
unit = [ cos( th ) ; sin( th ) ]; % random unit vector in 2D

Ux = zeros( size( Xs ) );
Uy = zeros( size( Ys ) );

xm = mean( Xs(:) ); Xrel = Xs - xm;
ym = mean( Ys(:) ); Yrel = Ys - ym;
pts = [ Xrel(:) Yrel(:) ]';
projection = unit' * pts;
U = reshape( mag * cos( pi * projection / lam_cdw ), size( Ux ) );

Ux = U * cos( th ) .* sup3d;
Uy = U * sin( th ) .* sup3d;
Uz = zeros(size(Xs)) .* sup3d .* sign(Zs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

