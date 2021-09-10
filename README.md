# Source separation for Video sequences

example bash file to run code

`bash train.sh`

code contains:

train.py -- the main function to use to train the model

dataset.py -- dataset to be used to make dataloaders

conf.py -- configuration file to 

models
 |----- modules.py -- stores different network modules such as resnet and atlasnet 
 |----- network.py -- the main networks that exhibited the best performance with atlasnet decoder
 |----- affine_network.py -- the networks that uses affine transformations
 |----- fix_mapping_netowrk.py -- the networks that uses affine transformation and fixed mapping
