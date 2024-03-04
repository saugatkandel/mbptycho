function h = displayisosurf(obj, isonum, color, X, Y,Z)

%SH Oct 2008
%
% displays an isosurface defined by 'isonum' of a 3d intensity matrix (obj).  If
% isonum is positive, then that value will be used to extract an
% isosurface.  If it is negative, then it will be treated as a fraction of
% the maximum intensity in the obj matrix, where the isosurface will
% be extracted at an intensity of (-isonum)*(max intensity).  Color is a
% string. 


% if no X Y Z matrices were given, then we make them to fit the size of the
% object array
if nargin<4
    x=[1:size(obj,2)];
    y=[1:size(obj,1)];
    z=[1:size(obj,3)];
    
    [X,Y,Z]= meshgrid(x,y,z);
end

if nargin<3
    color='g';
end

if nargin<2
    isonum=-.5;
end

% if nargin >=4 %assume an irregular grid
%     
%     minx = min(min(min(X)));
%     miny = min(min(min(Y)));
%     minz = min(min(min(Z)));
% 
%     maxx = max(max(max(X)));
%     maxy = max(max(max(Y)));
%     maxz = max(max(max(Z)));
%     
%     rangex = maxx - minx;
%     rangey = maxy - miny;
%     rangez = maxz - minz;
%     
%     x = [minx: rangex/(1*size(obj,2)): maxx];
%     y = [miny: rangey/(1*size(obj,1)): maxy];
%     z = [minz: rangez/(1*size(obj,3)): maxz];
%     
%     [Xreg, Yreg, Zreg] = meshgrid(x,y,z);
%     
%     clf
%     hold on
%     for i=1:numel(X) plot3(X(i), Y(i), Z(i), 'bo');end
%     for i=1:numel(Xreg) plot3(Xreg(i), Yreg(i), Zreg(i), 'r*');end
%     hold off
%     
%     griddata3(X,Y,Z, abs(obj), Xreg, Yreg, Zreg);
%     
%     pause
% end

if isonum<0
    isonum = -isonum* max(max(max(abs(obj))));
end

obj = abs(obj);

isonum=double(isonum);

p = patch(isosurface(X,Y,Z,obj,isonum));
%isonormals(X,Y,Z,obj,p)
set(p,'FaceColor',color,'EdgeColor','none');
h=p;
daspect([1 1 1])
view(3); axis tight

if isempty(findobj(gca, 'Type', 'light'))
    camlight 
    camlight(-170.1962, - 53.9022);
end

%lighting( 'gouraud' );
