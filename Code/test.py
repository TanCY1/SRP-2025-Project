y_true = [1,0]
from model import model

model = model()


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


fpr,tpr,threshold = roc_curve(y_true, y_score)

plt.plot(fpr,tpr)
plt.show()