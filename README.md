# i-vector-tf
This repo to support the publication: Whisper to neutral mapping using cosine similarity maximization in i-vector space for speaker verification

Main components:
1. feature to i-vector as tensorflow layer: 
    i_vec_layer.py: file containing the layer definition. it can be used as follows:  
    ivec_layer=i_vec_layer(ivec_dim, fdmin, nmix,root+'ubm.gmm',root+'tvmat.h5');
    
2. cosine similarity objective function.
    look at train.py for example usage and the objective function.
