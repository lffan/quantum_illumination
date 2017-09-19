import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

df_two = pd.read_csv('../output/expr_3_r1_p51.csv')

RA = np.linspace(0, 1, 51)
RB = np.linspace(0, 1, 51)
RA, RB = np.meshgrid(RA, RB)
qcb_pcs = df_two['Helstrom'].values.reshape(RA.shape)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(RA, RB, qcb_pcs,
                rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)

ax.set_xlabel('$r_a$')
ax.set_ylabel('$t_b$')
ax.set_zlabel('$QCB$')
ax.set_title('$N_{th}=0.1$, $N_s=0.01$, $\eta=0.01$')

ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.4f'))

# fig.colorbar(qcb_pcs, shrink=0.5, aspect=5)
plt.show()
