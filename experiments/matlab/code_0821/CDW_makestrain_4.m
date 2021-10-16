%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change this for different samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

b = 1.; 			% lattice constant
v = 0.29;		% Poisson's ratio

% Two edge dislocations close to each other. 

coords_all = [ Xs(:) Ys(:) ];
f = find( sup3d(:) > 0.5 );
coords = coords_all( f, : );

%U = RotatedEdgeDislocation( coords_all', [ -0.175 0 ], pi/4, b, v );
%Ux = U(1,:);
%Uy = U(2,:);
%Uz = zeros( size( Ux ) );

[ Ux, Uy, Uz ] = TwoEdgeSlipSystem( coords_all', b, v );
Ux = sup3d .* reshape( Ux, size( Xs ) );
Uy = sup3d .* reshape( Uy, size( Xs ) );
Uz = sup3d .* reshape( Uz, size( Xs ) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

