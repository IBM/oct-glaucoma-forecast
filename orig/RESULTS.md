multimodal training: 
Epoch: 80.00, trainloss: 258.04, kl_loss: 4.84, valloss: 421.62, gm_mae_rec: 2.43, gm_mae_pred: 3.68, time/ep: 588.73


['rnfl', 'gcl', 'vft'] forecast mae 3.68+-3.03  recon mae2.43+-2.08
['rnfl', 'gcl', None] forecast mae 3.70+-3.15  recon mae2.47+-2.19
[None, 'gcl', 'vft'] forecast mae 5.52+-4.07  recon mae4.86+-4.02
['rnfl', None, 'vft'] forecast mae 3.14+-3.10  recon mae2.01+-1.41
['rnfl', None, None] forecast mae 3.14+-3.15  recon mae1.75+-1.28
[None, 'gcl', None] forecast mae 6.01+-4.65  recon mae5.21+-4.47
[None, None, 'vft'] forecast mae 8.89+-6.29  recon mae8.64+-5.8



training using only rnfl: 
Epoch: 80.00, trainloss: 13.64, kl_loss: 0.15, valloss: 17.61, gm_mae_rec: 1.00, gm_mae_pred: 2.53, time/ep: 70.55

#In second run the forecasting error = 2.45



#training on rnvl+vft

['rnfl', None, 'vft'] forecast mae 3.14+-3.13  recon mae1.81+-1.48
['rnfl', None, None] forecast mae 2.93+-3.30  recon mae1.35+-1.11
[None, None, 'vft'] forecast mae 9.45+-6.79  recon mae9.29+-6.21


#there is no advantage of combining vft with rnfl, in which scenerio cvt
is important? Missing rnfl timepoints in test time? For example, using
one visit rnfl and vft only?


GRU only=> forecasting error = 2.74

ODE-GRU => 2.82 with adamsmethod


Note: ode_map_multimodal_poe.py on oct_ofecasting Epoch: 80, trainloss:
83.22, kl_loss: 0.70, dx_loss: 0.73, dx_auc: 0.48, valloss: 107.56,
dx_aucval: 0.53, gm_mae_rec: 0.87, gm_mae_pred: 2.47 time/ep:161.90 sec
This uses odegru in decoder as well and reconstructs only RNFL

 

1. Story: Combining different modaloty may not give sigificant imporvements
over individual but can help one of the modality or view is missing to
predict the others.

2. In case the image is corrupted for one view, then can we still use
   the corrupted images in multimodal settings i.e. if voea rnfl is
   corrupted can we use onh + corrupted fovea to forecast the fovea rnfl
   
3. For this at least the combination should be as good as individual
   use. One possible reason that may not work is if there is no common
   information between the view. If the RNLF or ONH and fovea scans
   correlated?
4. How can the dx and visual field be used. Using visual field to
   constrain the output ma be more appropriate. So the forecasting can
   can be the viual field as well, ie the visual field is also
   forecasted. This is important as the visual field denotes important
   aspect of vision. Challenge: not all oct have corresponding visual
   field. What can we do in this case?

BUt the first question is would fovea+onh work? what are common
information there? How correlated are they? 

Main argument can be missing and corrupted view. For this evaluation
have to be done in different dataset that shows the missng data or
incomplete data.

Multiview structure and function forecastig from multiview data 


Experiment with RNFL nad RNFL_GCL
['rnfl', 'gcl', None] forecast mae 2.56+-2.61  recon mae1.56+-1.28
['rnfl', None, None] forecast mae 2.53+-2.75  recon mae1.32+-1.02
[None, 'gcl', None] forecast mae 2.80+-2.66  recon mae2.05+-1.65


above exp but the loss term contains reconsruction of RNFL only ie
glcrnfl loss is not inlcuded
 
['rnfl', 'gcl', None] forecast mae 2.60+-2.93 recon mae1.22+-1.01
['rnfl', None, None] forecast mae 2.66+-2.98 recon mae0.93+-0.76
[None, 'gcl', None] forecast mae 2.75+-2.78 recon mae1.82+-1.34

#Now gclrnfl loss is included as well but durin training all training
all time points are used for recosntruction, but during testimg only 3
time points are used to predict 4th - 4 visit inputs during training

['rnfl', 'gcl', None] forecast mae 2.78+-3.04 recon mae1.12+-0.92
['rnfl', None, None] forecast mae 2.63+-2.98 recon mae0.87+-0.68
[None, 'gcl', None] forecast mae 2.76+-2.81 recon mae1.62+-1.29

#above second run - 4 visit input in training

['rnfl', 'gcl', None] forecast mae 3.00+-3.04  recon mae1.35+-1.11
['rnfl', None, None] forecast mae 2.93+-3.09  recon mae1.26+-0.88
[None, 'gcl', None] forecast mae 2.99+-2.90  recon mae1.79+-1.42



#rnfl + rnflgcl loss with forecasting ie 3 visits inuput during trainig
#NV 3 ['rnfl', 'gcl', None] forecast mae 2.65+-2.70 recon mae1.46+-1.24
['rnfl', None, None] forecast mae 2.68+-2.86 recon mae1.15+-0.94
[None, 'gcl', None] forecast mae 2.89+-2.66 recon mae1.96+-1.56

Second run
['rnfl', 'gcl', None] forecast mae 2.76+-3.04  recon mae1.15+-1.06
['rnfl', None, None] forecast mae 2.67+-2.91  recon mae0.85+-0.75
[None, 'gcl', None] forecast mae 2.87+-2.78  recon mae1.70+-1.48


#now with 4 visitss during training
['rnfl', 'gcl', None] forecast mae 2.84+-2.96  recon mae1.16+-0.95

['rnfl', None, None] forecast mae 2.73+-2.94  recon mae0.94+-0.76

[None, 'gcl', None] forecast mae 2.98+-2.89  recon mae1.67+-1.41



#multimodalpoe_odegru_64110 - with 3 visits during training
['rnfl', 'gcl', None] forecast mae 2.46+-2.85  recon mae0.88+-0.67

['rnfl', None, None] forecast mae 2.53+-2.95  recon mae0.80+-0.64

[None, 'gcl', None] forecast mae 10.26+-7.87  recon mae9.46+-7.09


#now trying by removing prior expert when only one modality is in
forward pass
['rnfl', 'gcl', None] forecast mae 2.45+-2.79  recon mae0.92+-0.75

['rnfl', None, None] forecast mae 2.52+-3.01  recon mae0.74+-0.57

[None, 'gcl', None] forecast mae 10.85+-6.83  recon mae9.92+-5.77




