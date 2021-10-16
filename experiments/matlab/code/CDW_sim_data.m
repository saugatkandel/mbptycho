%set up ptycho scan:
[Xscan Yscan] = meshgrid([-5:5]*4*d2_bragg, [-5:5]*4*d2_bragg);

%presume same scan area for all Bragg peaks
for ii=1:nBraggpeaks
    scans(ii).xpos = Xscan(:);
    scans(ii).ypos = Yscan(:);
end

clear verts kis

%do sanity check to deterime what the q for each position looks like WRT
%film,

for ii=1:nBraggpeaks
        
    v = [Xs(ins) Ys(ins) Zs(ins)];
    th = motlist(ii,1);
    tth = motlist(ii,2);
    gam = motlist(ii,3);
    
    Rth = [cosd(th) 0 sind(th);
            0 1 0;
            -sind(th) 0 cosd(th)];

    Rtth = [cosd(-tth) 0 sind(-tth);
            0 1 0;
            -sind(-tth) 0 cosd(-tth)];
        
    Rgam = [1 0 0;
            0 cosd(-gam) -sind(-gam);
            0 sind(-gam) cosd(-gam)];
        
    verts(ii).A = ( Rgam * Rtth * Rth * v.').';

    ki = [0 0 1];
    kis(ii).A = ( Rgam * Rtth * ki.').';
    
    kf = [0 0 1];
    qs(ii).A = kf-kis(ii).A;
    
    figure(ii);clf;
    plot3(verts(ii).A(:,1), verts(ii).A(:,2), verts(ii).A(:,3), 'k.'); axis image;
    hold on; quiver3(0,0,0,kis(ii).A(1), kis(ii).A(2), kis(ii).A(3),2); hold off
    hold on; quiver3(0,0,0,kf(1), kf(2),kf(3),2); hold off
    hold on; quiver3(0,0,0,qs(ii).A(1), qs(ii).A(2), qs(ii).A(3),2); hold off
    xlabel('x'); ylabel('y'); zlabel('z');
    
    %now make sure beam makes sense
    v = [Xs(:) Ys(:) Zs(:)];
    v = ( Rgam * Rtth * Rth * v.').';
    di(probes(ii).A, .5, 'g', reshape(v(:,1),size(Xs)), reshape(v(:,2),size(Xs)), reshape(v(:,3), size(Xs)));
    
end

%now generate data at every probe position for all Bragg peaks

cnt =1;
[Xs2 Ys2] = meshgrid([-Npixsamp/2:Npixsamp/2] * d2_bragg);
figure(4); clf;

for ii = 1:nBraggpeaks
    
    display(['doing Bragg peak ' num2str(ii)]);
    
    for kk=1 %:numel(Xscan(:)) %cycle through probe positions
 
        %shift the rho object in the film frame, as in the HXN motor scheme
        %tempprobe = circshift( probes(ii).A, round([-Yscan(kk) -Xscan(kk)]/d2_bragg));
        tempprobe = probes(ii).A;
        
        projA = zeros(size(Xs2));
        projS = zeros(size(Xs2));
        
        for jj=1:numel(ins) %put the area weight of each rotated voxel into the projection plane
            
            field = tempprobe(ins(jj)) * rho(ii).A(ins(jj));
            
            pxcoord = [verts(ii).A(jj,1) verts(ii).A(jj,2)]/d2_bragg + Npixsamp/2;
            r=mod(pxcoord,1);
            
            projA(ceil(pxcoord(2)), ceil(pxcoord(1))) = projA(ceil(pxcoord(2)), ceil(pxcoord(1))) + r(1)*r(2) * field;
            projA(ceil(pxcoord(2)), floor(pxcoord(1))) = projA(ceil(pxcoord(2)), floor(pxcoord(1))) + (1-r(1))*r(2) * field;
            projA(floor(pxcoord(2)), ceil(pxcoord(1))) = projA(floor(pxcoord(2)), ceil(pxcoord(1))) + r(1)*(1-r(2)) * field;
            projA(floor(pxcoord(2)), floor(pxcoord(1))) = projA(floor(pxcoord(2)), floor(pxcoord(1))) + (1-r(1))*(1-r(2)) * field;

            projS(ceil(pxcoord(2)), ceil(pxcoord(1))) = projS(ceil(pxcoord(2)), ceil(pxcoord(1))) + r(1)*r(2);
            projS(ceil(pxcoord(2)), floor(pxcoord(1))) = projS(ceil(pxcoord(2)), floor(pxcoord(1))) + (1-r(1))*r(2);
            projS(floor(pxcoord(2)), ceil(pxcoord(1))) = projS(floor(pxcoord(2)), ceil(pxcoord(1))) + r(1)*(1-r(2));
            projS(floor(pxcoord(2)), floor(pxcoord(1))) = projS(floor(pxcoord(2)), floor(pxcoord(1))) + (1-r(1))*(1-r(2));

        end
            
        data(kk,ii).projS = projS;
        data(kk,ii).projA = projA;
        data(kk,ii).I = abs(fftshift(fftn(fftshift( projA ))));
        data(kk,ii).xpos = Xscan(kk);
        data(kk,ii).ypos = Yscan(kk);
                
    end
end