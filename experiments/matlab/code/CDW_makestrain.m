%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change this for different samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ux = .01*(cos(Xs)-1) .* sup3d .* sign(Xs);  %local lattice displacement in units of microns
Uy = .01*(cos(Ys*.5)-1) .* sup3d .* sign(Ys);
Uz = zeros(size(Xs)) .* sup3d .* sign(Zs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

