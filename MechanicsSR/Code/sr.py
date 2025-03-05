import os
os.chdir("C:\\Users\\czachry3\\Downloads\\Research\\MechanicsSR-main\\MechanicsSR-main\\MechanicsSR\\Code\\")
print(os.getcwd())

from S_NonNN_mechanicsSR import run_mechanicsSR

run_mechanicsSR("C:\\Users\\czachry3\\Downloads\\Research\\MechanicsSR-main\\MechanicsSR-main\\MechanicsSR\\example_data\\","DATA_J_ALPHA_BETA_GAMMA_NU_EPSILONY_N.txt",100,"14ops.txt", polyfit_deg=3)
