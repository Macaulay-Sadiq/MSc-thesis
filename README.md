# MSc-thesis

## PointNet with Local Support 

This work is intended to learn localized patterns from 3D face models of unordered point-set representations. This method will allow easy identification of discriminate features of occlusion on a 3D face model for effective recovery of the face model.
We implemented an auto-encoder network, based on PointNet architecture and introduced a local-support operation to learn local patterns on face models. We trained and tested our autoencoder network on the 
 <a href="http://bosphorus.ee.boun.edu.tr/HowtoObtain.aspx" target="_blank">Bosphorus 3D face dataset</a>.


## Installation

The code has been tested with Python 2.7, <a href="https://www.tensorflow.org/install/" target="_blank">TensorFlow gpu</a> (version>=1.14.0) and CUDA 10.0. 

For point cloud reconstruction loss function, we need to compile two custum TF operators under `tf_ops/nn_distance` (Chamfer's distance) and `tf_ops/approxmatch` (earth mover's distance). 
Check the `tf_compile_*.sh` script under these two folders, modify the TensorFlow and CUDA path accordingly before you run the shell script to compile the operators.

For visualization, we will need to install `VTK` :
    
    pip install vtk


### Usage

Extract the Bosphorus dataset into `data/` and run the 'bnt2xyz.m' script file the directory to save the 3D model in `.xyz` file format.

Once the directory of the `.xyz` files have been successfully created, the training files can be run to obtain the model from the network. THe training models can be written in the logs directory by default.
At hthe initial run of the train file, a `datafile.txt` is generated iin the project directory. It contains the training, validatin and test set splitted into 80%, 20% and 20% respectively.

To train 3D face model with CD loss function:

    python train_faces.py --model model --log_dir logs/<name_of_logdir>
    
To train 3D face model with EMD loss function:

    python train_faces.py --model model_emd --log_dir logs/<name_of_logdir>

To train network with local-support:

    python train_ls.py --model pointnet_ls --log_dir logs/<name_of_logdir> 

Some pre-trained models for the point-cloud reconstruction can be dounloaded from <a href="https://www.tensorflow.org/install/" target="_blank">here</a>.

To obtaine results from the trained model with local-support network for 3D faces, run:

    python test_ls.py --model pointnet_ls --model_path logs/log_FPS_10000_ls_exp6/model.ckpt --num_point 10000

this will write the target, input, and reconstructed point-cloud from the testing set into `results/` directory.

To obtain results from model with only PointNet and CD loss network for 3D faces, run:

    python test_faces.py --model model --model_path logs/log1_CD10000/model.ckpt --num_point 1000
    
this will write only the reconstructed point-cloud data from the testing set to the `results/` directory.

    
### Visualize Reconstruction

run:

    python eval_neutralization.py --exp_name results/
    
to visualize the written samples of poin-cloud data files in `results/` in four view ports of target, input, PointNet prediction and PointNet with local-support prediction of Face model.
Use key `N` and `B`to move to next and previous face sample, `P` to render points as 3D spheres and `S` to save current visualization window in `.png` format.
    
