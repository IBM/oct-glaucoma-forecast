I am trying to create a minimal version of Suman's training code so that it can be run on CCC

# Created an environment on CCC
- exported the dependencies in pytorch_1.2.0_py3.6_x86_64_v1
- created a new environment "suman3" based on envornment.yml
- upgrade PyTorch becasue torchdiffeq needs PyTorch 1.3 (`conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch`)
- use pip install to installthe latest version of torchdiffeq. 
- install other dependencies
- make sure that "tempmvae_ld32" exists in trainining/

# Run main model pretraining
./run.sh -t 6h train_multimodal_latentodegru_pretrain.py

# Run main model training (Suman's original version that only reads 4 visits)
./run.sh -t 12h train_multimodal_latentodegru_sync.py