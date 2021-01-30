# %%
from IPython.display import display  # noqa

from tensile.tensile import *

if __name__ == '__main__':
    logger.level = logging.DEBUG
    data1 = TensileTest(fp='data/test.csv')
    data2 = TensileTest(fp='data/test2.csv')

    # data3 = data1[['1 _ 3', '1 _ 5', 3]] + data2[['2 _ 1', 1]]
    # data3.calculate()
    # plt.figure('Initial modulus', facecolor='white')
    # res = data3.cal_init_modulus(xlim=[0.05, 0.1])
    # display(res)
