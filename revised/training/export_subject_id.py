from scripts.datautils import MatchedTestSet
import numpy

fold_seed = 5

data = MatchedTestSet(seed = fold_seed)

sid_ = data.val_sid
sid = sid_.cpu().detach().numpy()
sid = sid[:,0]
print(sid.shape)
numpy.savetxt("results_" + str(fold_seed) + "/subjects.csv", sid.astype(int), delimiter=",", fmt="%d")

age_at_visit_ = data.age_at_vd_val_0
age_at_visit = age_at_visit_.cpu().detach().numpy()
print(age_at_visit.shape)
numpy.savetxt("results_" + str(fold_seed) + "/subjects_age_at_visit.csv", age_at_visit, delimiter=",", fmt = "%.4f")
