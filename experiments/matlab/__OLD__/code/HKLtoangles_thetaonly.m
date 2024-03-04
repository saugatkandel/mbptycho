function [th tth gam] =  HKLtoangles_thetaonly(H, K, L, lambda, latspacing)

%adapt input for Laue geometry at HXN from APS nanoprobe mounting
%convention
Htemp = K;
Ktemp = L;
Ltemp = H;

H = Htemp;
L = Ltemp;
K = Ktemp;

angy = atand(L/H);
ang_sym = asind( (norm([H K L])/2)/(latspacing/lambda) );
ang2_sym = ang_sym*2;
det_phi = asind(cosd(angy)/sind(90-ang_sym));
temp = (sind(ang2_sym) * cosd(det_phi))^2 + cosd(ang2_sym)^2;

th = asind(cosd(90-ang_sym)/sind(angy));
tth = asind(sind(ang2_sym)*cosd(det_phi)/sqrt(temp));
gam = atand(sind(ang2_sym)*sind(det_phi)/sqrt(temp));

%display(['Detector angles: th=' num2str(th) ', tth=' num2str(tth) ', gam=' num2str(gam)]);
