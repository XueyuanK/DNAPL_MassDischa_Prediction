#BSUB -q mpi    
#BSUB -J interp_T2n_t1         
#BSUB -n 24           
#BSUB -o %J.out      
#BSUB -e %J.err      
python Inverse_VAE_recon.py