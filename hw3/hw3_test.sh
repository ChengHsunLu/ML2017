#!/bin/bash
wget https://www.dropbox.com/s/95m2mt5b4f7bdyx/model_selftraining_10_v5.h5?dl=0 -O test_model1.h5
wget https://www.dropbox.com/s/u2wdrfj12jeb17t/model_selftraining_10_v6.h5?dl=0 -O test_model2.h5
wget https://www.dropbox.com/s/q1j4f9oyrbbt3ac/model_selftraining_10_v7.h5?dl=0 -O test_model3.h5
wget https://www.dropbox.com/s/1xsszaaj7mma3o3/model_selftraining_10_v8.h5?dl=0 -O test_model4.h5
python3 test.py $1 $2
