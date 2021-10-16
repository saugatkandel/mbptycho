global d2_bragg X Y Z ki_o 
warning off;

%experimental setup details
%(all distance units in microns throughout script)
pixsize = 55; %microns, for Merlin
lam = etolambda(9000)*1e-4;

%ZP details
zpdiam = 350;
outerzone = .05;
bsdiam = 100; %nominally 50, but the donut hole looks too small with 50
cutrad = .1;

%NW details
filmthickness = .1; %this is like the film thickness
grainwidth= .8; %edge-to-edge distance in x (as mounted at HXN)
grainheight = .5; %edge-to-edge distance in y 
 
Npix = 140; %presume a square ROI in the detector of size Npix Npix
depth = 100; %number of pixels of the numerical array in the 
Npixsamp = 200;
lattice = [.0003905 .0003905 .0003905]; %along XYZ directions as mounted

detdist = 0.350e6; %detector to sample distance, in microns
d2_bragg = detdist * lam /(Npix*pixsize);
defocus = 0;
display(['depth of numerical window covers ' num2str(d2_bragg * depth) ' microns']);

Braggpeaklist = [1 0 0;
                1 1 0;
                1 2 0];
nBraggpeaks = size(Braggpeaklist,1);            
motlist = zeros(size(Braggpeaklist));
for ii=1:nBraggpeaks
    H = Braggpeaklist(ii,1);
    K = Braggpeaklist(ii,2);
    L = Braggpeaklist(ii,3); 
    [m1 m2 m3] = HKLtoangles_thetaonly(H, K, L, lam, lattice(1)); %output is in theta, tth, gam APS NP convention: tth in-plane, gam oop
    motlist(ii,:) = [-m1 -m2 -m3];
end         

%%

CDW_makesample;
CDW_make_experimental_beam;
CDW_sim_data;
