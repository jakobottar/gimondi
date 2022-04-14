#! /bin/bash

python train.py -c configs/u3o8-5.json -n u3o8_5_semi_1_final
python train.py -c configs/u3o8-5.json -n u3o8_5_semi_2_final
python train.py -c configs/u3o8-5.json -n u3o8_5_semi_3_final
python train.py -c configs/u3o8-5.json -n u3o8_5_semi_4_final
python train.py -c configs/u3o8-5.json -n u3o8_5_semi_5_final