%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change this for different samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% multiple point inclusions in sample, 
% result in a radially exponential strain 
% emanating from inclusion location

Nincl = 7; % number of inclusions
alpha = 0.025; % radial decay constant of distortion in um
mag = 2e-4;	% maximum distortion

coords_all = [ Xs(:) Ys(:) ];
%coords = unique( coords_all, 'rows' );
coords = coords_all( sup3d(:) > 0.5, : );
rp = randperm( size( coords, 1 ), Nincl );
coords_incl = coords( rp, : );
Ux = zeros( size( Xs ) );
Uy = zeros( size( Ys ) );

for n = 1:Nincl
	coords_rel = coords_all - repmat( coords_incl(n,:), size( coords_all, 1 ), 1 );
	r = reshape( sqrt( sum( coords_rel.^2, 2 ) ), size( Ux ) );
	th = reshape( atan2( coords_rel(:,2), coords_rel(:,1) ), size( Ux ) );
	U = mag * exp( -alpha * r );
	Ux_this = U .* cos( th );
	Uy_this = U .* sin( th );
	Ux = Ux + Ux_this;
	Uy = Uy + Uy_this;
end

Ux = Ux .* sup3d;
Uy = Uy .* sup3d;
Uz = zeros(size(Xs)) .* sup3d .* sign(Zs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

