#BSUB -q gpu_v100   
#BSUB -J Evol3       
#BSUB -gpu "num=2"   
#BSUB -n 24           
#BSUB -o %J.out      
#BSUB -e %J.err     
python3.6 VAE_KS3D_conv.py