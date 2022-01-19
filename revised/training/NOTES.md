jbsub -mem 24g -cores 1+1 -queue x86_1h python train_multimodal_latentodegru_sync_pretrain.py

jbadmin -kill XXX
