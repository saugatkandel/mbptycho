%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change this for different samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% partial edge dislocation, randomly 
% located terminus, random direction 
% in plane.

% first getting random terminus
mag = 1.e-3 % magnitude of distortion

coords_all = [ Xs(:) Ys(:) ];
coords = coords_all( sup3d(:) > 0.5, : );
rp = randperm( size( coords, 1 ), 1 );
terminus = coords( rp, : );
th = pi * ( -1 + 2*rand );	% random in plane orientation for CDW
unit = [ cos( th ) ; sin( th ) ];		% along edge dislocation
unit_p = [ cos( th + pi/2 ) ; sin( th + pi/2 ) ];	% normal to edge dislocation

coords_rel = coords_all - repmat( terminus, size( coords_all, 1 ), 1 );
phi = atan2( coords_rel * unit_p, coords_rel * unit );
phi( phi < 0 ) = phi( phi < 0 ) + 2*pi;

Ux = sup3d .* reshape( ...
	mag * ( phi < pi/2 ) .* cos( th + pi/2 ) + ...
	mag * ( phi > pi/2 & phi < 3*pi/2 ) .* cos( th + phi ) + ...
	mag * ( phi > 3*pi/2 ) .* cos( th + 3*pi/2 ), ...
	size( Xs ) );
Uy = sup3d .* reshape( ...
	mag * ( phi < pi/2 ) .* sin( th + pi/2 ) + ...
	mag * ( phi > pi/2 & phi < 3*pi/2 ) .* sin( th + phi ) + ...
	mag * ( phi > 3*pi/2 ) .* sin( th + 3*pi/2 ), ...
	size( Ys ) );
	
Uz = zeros(size(Xs)) .* sup3d .* sign(Zs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

