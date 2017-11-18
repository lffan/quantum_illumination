import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# df_two = pd.read_csv('../output/expr_2_r2_p51.csv')
#
# RA = np.sqrt(np.linspace(0, 1, 51))
# RB = np.sqrt(np.linspace(0, 1, 51))
# RA, RB = np.meshgrid(RA, RB)
# qcb_pcs = df_two['Helstrom'].values.reshape(RA.shape)
#
# fig = plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(RA, RB, qcb_pcs, \
#                 rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
# ax.set_xlabel('$r_a$')
# ax.set_ylabel('$t_b$')
# ax.set_zlabel('$QCB$')
# ax.set_title('$N_{th}=0.1$, $N_s=0.01$, $\eta=0.01$')
#
# ax.zaxis.set_major_locator(LinearLocator(5))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
#
# # fig.colorbar(qcb_pcs, shrink=0.5, aspect=5)
# plt.show()

df2 = pd.read_csv('../output/expr_2_r2_p51.csv')
df3 = pd.read_csv('../output/expr_3_r1_p51.csv')
df = pd.concat([df2, df3])

sns.heatmap(df3['Helstrom'].values.reshape(51, 51), cmap="RdBu_r")
plt.title("Helstrom")
plt.show()

sns.heatmap(df3['Chernoff'].values.reshape(51, 51), cmap="RdBu_r")
plt.show()

sns.heatmap(df3['VN_Entropy'].values.reshape(51, 51), cmap="RdBu_r")
plt.show()

sns.heatmap(df3['Aver_N'].values.reshape(51, 51), cmap="RdBu_r")
plt.show()

sns.heatmap(df3['A_N'].values.reshape(51, 51), cmap="RdBu_r")
plt.show()

sns.heatmap(df3['B_N'].values.reshape(51, 51), cmap="RdBu_r")
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xs = df['ra'].values
# ys = df['rb'].values
# zs = df['Helstrom'].values
# ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.Spectral)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xs = df['ra'].values
# ys = df['rb'].values
# zs = df['VN_Entropy'].values
# ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.Spectral)
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xs = df['ra'].values
# ys = df['rb'].values
# zs = df['Aver_N'].values
# ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.Spectral)
# plt.show()

