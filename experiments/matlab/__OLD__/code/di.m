function h = di(obj, isonum, color, X, Y,Z)

if nargin<2 h = displayisosurf(obj); return; end
if nargin<3 h = displayisosurf(obj, isonum); return; end
if nargin<4 h = displayisosurf(obj, isonum, color); return; end

h = displayisosurf(obj, isonum, color, X, Y,Z);