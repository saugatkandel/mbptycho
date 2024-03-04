
% NW_make_exp_beam creates a beam for the experiment simulation and the
% image reconstruction from an experimentally measured beam.

namefile = 'recon_probe_29902.mat';
probe_load = load(namefile);

% load the beam in matrix and orientate it in the right position (from Xianjin email of 3/27/2017:
% The displaying orientation is in the lab frame, right-positive x, up-positive y.)
probe = fliplr(probe_load.prb');

probe = probe / max(abs(probe(:)));
prpixsize = 10e-3; % in micros
[Xp Yp] = meshgrid([-50:49]*prpixsize);

indin = find(abs(probe)>0.025);

vertsprobe = [Xp(indin) Yp(indin) ones(size(indin))*-2];
vertsprobe = [vertsprobe; 
                Xp(indin) Yp(indin) ones(size(indin))*2];
v=zeros(size(vertsprobe)); %temp variable that will be rotated to different places
            
for ii=1:nBraggpeaks
    th = motlist(ii,1);
    Rth = [cosd(-th) 0 sind(-th);
            0 1 0;
            -sind(-th) 0 cosd(-th)];
    v = Rth * vertsprobe.';
    v=v.';
    probes(ii).A = griddata( v(:,1), v(:,2), v(:,3), [probe(indin);probe(indin)], Xs(:), Ys(:), Zs(:) );
    probes(ii).A(isnan(probes(ii).A))=0;
    probes(ii).A = reshape(probes(ii).A, size(Xs));

    figure(10+ii); clf;
    plot3(v(:,1), v(:,2), v(:,3), 'ro', Xs(ins), Ys(ins), Zs(ins), 'k.');axis image
    hold on; di(probes(ii).A, .5, 'g', Xs,Ys,Zs); hold off
    pause(2);
end


            
