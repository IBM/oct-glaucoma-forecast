from train_multimodal_latentodegru_pretrain import MultimodalTimeSeriesData

from train_multimodal_latentodegru import MultimodalTimeSeriesData as MultimodalTimeSeriesData2

data = MultimodalTimeSeriesData(fold_seed = 2)
print('Training data size', data.size_train())
print('Val data size', data.size_val())

data = MultimodalTimeSeriesData2(fold_seed = 2)
print('Training data size', data.size_train())
print('Val data size', data.size_val())
