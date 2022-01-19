from train_multimodal_latentodegru_sync import MultimodalTimeSeriesData

data = MultimodalTimeSeriesData(fold_seed = 2)
print('Training data size', data.size_train())
print('Val data size', data.size_val())
