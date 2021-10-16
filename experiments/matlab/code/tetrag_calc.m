%% General tetragonal/cubic beamline diffraction calculator 

% Given an orientation and lattice, what is the sample and detector
% positioning necessary to reach a desired reflection
% Surface normal will be +x outboard at th 0 (LAB FRAME DIRECTION OF SURFACE NORMAL)

Ekev = 10.4; % beam energy in kev
%Ekev = 8.04778; % beam energy in kev (lab source)
%refl = [1 1 3];   %Desired reflection [h k l]

refl = [-1 -1 1];   %Desired reflection [h k l] CHANGE HERE

lat_surf = [0 0 1];  % Surface normal unit vector in [h k l] 
lambda = 12.39842/Ekev; % wavelength 1/A
kbeam = 2*pi*Ekev/12.39842; % beam momentum 1/A
ki = kbeam*[0 0 1];        %Incident k vector (+z lab frame)

lat = [5.12 5.12 5.12]; %YSZ
%lat = [4.24 4.24 4.24]; % Fe3O4 cubic lattice vectors a b c (A)
%lat = [3.36 3.36 5.9]; % TaS2 trigonal lattice vectors a b c (A)
%lat = [3.905 3.905 3.905]; % SrTiO3 cubic lattice vectors a b c (A)
%lat = [3.373 3.373 5.903]; % Sr2IrO4 tetragonal lattice vectors a b c (A)
%lat = [4.9424 4.9424 14.0161]; % V2O3 lattice vectors a b c (A)

%tetragonal case
%%{
astar = [2*pi/lat(1) 0 0];
bstar = [0 2*pi/lat(2) 0];
cstar = [0 0 2*pi/lat(3)];
%}
%rhombohedral corundum case
%{
avec = [lat(1) 0 0];
% testing phi compensation 
bvec = [lat(2)*cosd(123) lat(2)*sind(123) 0];
% bvec = [lat(2)*cosd(120) lat(2)*sind(120) 0];
cvec = [0 0 lat(3)];

vol1 = dot(avec,cross(bvec,cvec));
astar = 2*pi*cross(bvec,cvec)/vol1;
bstar = 2*pi*cross(cvec,avec)/vol1;
cstar = 2*pi*cross(avec,bvec)/vol1;
%}

q1 = refl(1)*astar+refl(2)*bstar+refl(3)*cstar;
surf_q = lat_surf(1)*astar+lat_surf(2)*bstar+lat_surf(3)*cstar;
display(q1);
display(surf_q);

th_b = asin(lambda*norm(q1)/(4*pi)); %Bragg theta
th_p = pi/2 - th_b;         %Polar angle of bragg Q relative to ki

%Polar angle of reflection relative to surface normal vector
sam_polar = acos((q1*surf_q')/(norm(q1)*norm(surf_q)));
display(sam_polar);
th_out=zeros(361,6);
for ii=0:360
%phi = 0 corresponds to total non-surface normal component oriented up (+y)
%positive phi then rotates this following right-hand-rule about outboard (+x)
%phi numbers correspond to a CCW rotation of sample relative to post when
%viewed on bench microscope

    samphi = ii*pi/180;
    x=fminsearch(@(x) findangles(x,0,samphi,th_p,sam_polar), 0);
    samth = x;
    surf_norm=[cos(samth) 0 -sin(samth)];
    vec1 = [cos(sam_polar) sin(sam_polar) 0];
    vec2 = [vec1(1) vec1(2)*cos(samphi) vec1(2)*sin(samphi)];
    vec3 = [(vec2(1)*cos(samth)+vec2(3)*sin(samth)) vec2(2) (-vec2(1)*sin(samth)+vec2(3)*cos(samth))];
    kf = ki+norm(q1)*vec3;
    th_out(ii+1,1) = x*180/pi;
    th_out(ii+1,2) = ii;
    th_out(ii+1,3) = 90-acos((kf*surf_norm')/norm(kf))*180/pi;
    th_out(ii+1,4) = atan(kf(1)/kf(3))*180/pi;
    th_out(ii+1,5) = 90-acos((kf*[0 1 0]')/norm(kf))*180/pi;
    th_out(ii+1,6) = (th_out(ii+1,1)>=0)*(th_out(ii+1,3)>=0)*(th_out(ii+1,4)>=0)*(th_out(ii+1,5)>=0);
end
figure(56);clf;plot(th_out(:,2),th_out(:,1));title('Sample theta (must be >0)');
figure(57);clf;plot(th_out(:,2),th_out(:,3));title('Exit angle (must be >0)');
figure(58);clf;plot(th_out(:,2),th_out(:,4));title('Detector 2theta (cylindrical)');
figure(59);clf;plot(th_out(:,2),th_out(:,5));title('Detector gamma (cylindrical)');
figure(60);clf;plot(th_out(:,2),th_out(:,6));title('Accessible reflection region');




