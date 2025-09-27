

import matplotlib.pyplot as plt
import numpy as np

# 创建一个全黑的二维数组（海绵）
sponge = np.zeros((100, 100))  # 100x100的全黑图像

# 使用imshow显示图像
plt.imshow(sponge, cmap='gray', vmin=0, vmax=1)

# 隐藏坐标轴
plt.axis('off')
# plt.savefig("./tges.png")
# 显示图像
plt.show()