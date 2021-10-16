function [ u ] = RotatedEdgeDislocation( pts, endpt, th, b, v )
	
	R1 = [ cos( th ) -sin( th ) ; sin( th ) cos( th ) ]; 		% R(th)
	R2 = [ -sin( th ) -cos( th ) ; cos( th ) -sin( th ) ];		% R(th+90)
	%ptsR = R'*( pts - R' * repmat( endpt', 1, size( pts, 2 ) ) );	% rotate grid passively
	ptsR = R2 * ( pts - repmat( R1'*endpt', 1, size( pts, 2 ) ) );
	r = sqrt( sum( ptsR.^2 ) );
	th = atan2( ptsR(2,:), ptsR(1,:) );
	ux = ( b/2/pi )*( th + sin( 2*th )/4/(1-v) );
	uy = ( b/2/pi )*( -( 1-2*v )*log( r.^2/b^2 )/( 4*(1-v) ) + ( cos(2*th)-1 )/( 4*(1-v ) ) );
	u = [ ux(:) uy(:) ]';
	u = R1 * u;

end
