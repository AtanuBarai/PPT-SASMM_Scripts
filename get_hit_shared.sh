#!/bin/bash

cd $1
echo $PWD

echo "1 Uniform"
python ../phit.py reuse_profile_mimic_1_cores_uniform.dat
echo "2 Uniform"
python ../phit.py reuse_profile_mimic_2_cores_uniform.dat
echo "2 RR"
python ../phit.py reuse_profile_mimic_2_cores_round_robin.dat
echo "4 Uniform"
python ../phit.py reuse_profile_mimic_4_cores_uniform.dat
echo "4 RR"
python ../phit.py reuse_profile_mimic_4_cores_round_robin.dat
echo "8 Uniform"
python ../phit.py reuse_profile_mimic_8_cores_uniform.dat
echo "8 RR"
python ../phit.py reuse_profile_mimic_8_cores_round_robin.dat
echo "16 Uniform"
python ../phit.py reuse_profile_mimic_16_cores_uniform.dat
echo "16 RR"
python ../phit.py reuse_profile_mimic_16_cores_round_robin.dat
