#!/bin/bash

cd $1
echo $PWD

python ../phit.py reuse_profile_mimic_1_cores_uniform.dat >& result_phit.txt
python ../phit.py reuse_profile_mimic_2_cores0.dat >> result_phit.txt
python ../phit.py reuse_profile_mimic_2_cores1.dat >> result_phit.txt
python ../phit.py reuse_profile_mimic_4_cores0.dat >> result_phit.txt
python ../phit.py reuse_profile_mimic_4_cores1.dat >> result_phit.txt
python ../phit.py reuse_profile_mimic_8_cores0.dat >> result_phit.txt
python ../phit.py reuse_profile_mimic_8_cores1.dat >> result_phit.txt
python ../phit.py reuse_profile_mimic_16_cores0.dat >> result_phit.txt
python ../phit.py reuse_profile_mimic_16_cores1.dat >> result_phit.txt
