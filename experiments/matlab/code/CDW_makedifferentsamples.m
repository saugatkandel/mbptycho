% create different vector ptycho datasets, store in different workspaces.

for nn = 1:3
	clear -x nn; 
	outfile = sprintf( 'Sample_%d.mat', nn );
	system( 'cp CDW_makesample_orig.m CDW_makesample.m' );
	copycommand = sprintf( 'sed -i \"s/CDW_NOTHINGHERE/CDW_makestrain_\%d/g\" CDW_makesample.m', nn )
	system( copycommand );
	CDW_masterscript;
	save( outfile );
end
