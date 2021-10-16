function [ Ux, Uy, Uz ] = TwoEdgeSlipSystem( pts, b, v )

	mn = min( max( pts, [],	2 ) );
	ratio = 0.25
	th = pi*( -1 + 2*rand() );
	endpoint = ratio * mn * [ -1 0 ];
	u1 = RotatedEdgeDislocation( pts, endpoint, th, b, v );
	%u2 = zeros( size( u1 ) );
	
	u2 = -1*RotatedEdgeDislocation( -pts, endpoint, th, b, v );
	%u1 = zeros( size( u2 ) );

	U = u1 + u2;

	Ux = U(1,:);
	Uy = U(2,:);
	Uz = zeros( size( Ux ) );

end
