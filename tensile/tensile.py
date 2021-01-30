import logging
import math
import re

from matplotlib import ticker
from scipy.integrate import simps
from scipy.signal import find_peaks

from base.base import *

logging.basicConfig(format='%(filename)s [line:%(lineno)d] %(levelname)s: %(message)s')
logger = logging.getLogger('tensile')
logger.level = logging.INFO


def to_si_unit(value, unit: str):
    unit = unit.lower()

    if unit in ('mm', '毫米'):
        return np.multiply(value, 0.001)
    elif unit in ('cm', '厘米', '%'):
        return np.multiply(value, 0.01)
    elif unit in ('m', '米', 's', '秒', 'n', '牛', 'pa'):
        return value
    elif unit in ('kpa',):
        return np.multiply(value, 1000)
    elif unit in ('mpa',):
        return np.multiply(value, 1000000)
    else:
        logger.error(f'unrecognized unit: {unit}')
        return value


def verify_unique(data: Iterable, throw=True, err_msg=None):
    data = list(data)
    if len(data) == len(set(data)):
        return
    logger.warning(f'Duplicated values found in: {data}')
    if err_msg:
        logger.warning(err_msg)
    if throw:
        raise ValueError(data)


def get_index_range(data: pd.Series, _range):
    start_index, end_index = -1, -1
    a, b = _range
    for index in data.index:
        if start_index < 0 < (data[index] - a) * (b - a):
            start_index = index
        if end_index < 0 < (data[index] - b) * (b - a):
            end_index = index
    return start_index, end_index


def scale_axis(x_scale=1, y_scale=1):
    # don't use
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * x_scale)))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * y_scale)))


def statistics_of_columns(df: pd.DataFrame):
    stat = pd.DataFrame(columns=df.columns)
    stat.loc['mean'] = df.mean()
    stat.loc['min'] = df.min()
    stat.loc['max'] = df.max()
    # 总体样本位置，自由度=N-ddof=N-1
    stat.loc['std'] = df.std(ddof=1)
    return stat


# %%
class TensileTest:
    def __init__(self, serial_name='Tensile', fp=None, info=None, data=None):
        """

        :param serial_name: 样品系列名称，并不是每一根曲线的名称
        :param fp: csv文件路径
        :param info:
        :param data:
        """
        # type:(str,str,pd.DataFrame,List[pd.DataFrame])
        assert fp is None or (info is None and data is None), '不能同时提供文件名和数据'
        self.serial_name = serial_name
        self.info: pd.DataFrame = info if info is not None else pd.DataFrame()
        self.tensile_data: List[pd.DataFrame] = data if data is not None else []
        self.units: Dict[str, str] = {
            'length': 'm',
            'stress': 'Pa',
        }
        if fp:
            self.read_csv(fp)

    def read_csv(self, fps: Union[str, List[str]], encoding='GBK'):
        """
        原始数据：每行宽度不等，大都没有","结尾，每个单元格有双引号围着"cell"
        Excel另存为: 每行宽度相等，空单元格直接空字串补全，即结尾出现n个逗号,,,,,

        :param fps: 可以是一个文件名或多个文件名：a.csv, 或 [a-1.csv, a-2.csv]
        :param encoding: 岛津软件采用中文界面，导出含中文的csv文件，编码默认使用GBK或GB2313，一般文件为UTF8
        :return:
        """
        if isinstance(fps, str):
            fps = [fps]
        src_text = ''
        for fp in fps:
            with open(fp, encoding=encoding)as fd:
                csv.reader(fp)
                src_text += fd.read()
        src_text = re.sub(',+(?=[\r,\n])', '', src_text)
        src_text = src_text.strip()
        # 默认采用的是2个换行符
        # blocks = src_text.split('\n\n')
        # why need ?:
        blocks = re.split(r'(?:\r|\n|\r\n){2,}', src_text)
        # block 1 - test info
        # blocks[0]

        # block 2 - sample size
        rows = parse_csv_str(blocks[1])
        shape = rows[1][0]
        info = []
        if shape in ('棒材',):  # add english later
            assert tuple(rows[2]) == ('名称', '直径', '标距')
            _, diameter_unit, length_unit = rows[3]
            for row in rows[4:]:
                name, diameter, length = self.parse_types(row, (str, float, float))
                area = math.pi * to_si_unit(diameter / 2, diameter_unit) ** 2
                length = to_si_unit(length, length_unit)
                info.append([name, area, length])
            self.info = pd.DataFrame(info, columns=['name', 'area', 'length'])
        elif shape in ('板材',):
            raise NotImplementedError(shape)
        else:
            raise NotImplementedError(shape)

        if len(set(self.info['name'])) != self.info['name'].size:
            logger.warning('样品名称重复，请使用数字序号作为样品编号（从0开始）或重新命名(rename函数)')

        # block 3 - calculated table
        # blocks[2]

        # block 4~end: data
        for sample_index, block in enumerate(blocks[3:]):
            rows = parse_csv_str(block)
            sample_name = rows[0][0]
            logger.debug(f'reading No.{sample_index}-"{sample_name}"')
            assert sample_name == self.info.loc[sample_index, 'name'], (
                sample_name, self.info.loc[sample_index, 'name'])
            assert tuple(rows[1]) == ('时间', '载荷', '行程'), rows[1]
            data_units = rows[2]
            df = pd.DataFrame(self.parse_types(rows[3:], float), columns=['time', 'load', 'displacement'])
            logger.debug(f'No.{sample_index}-"{sample_name}": {df.size}')
            for col, unit in zip(df.columns, data_units):
                df[col] = to_si_unit(df[col], unit)
            self.tensile_data.append(df)

    @staticmethod
    def parse_types(cells: Union[List[str], List[List[str]]], types: Union[Tuple, Type] = None):
        """

        :param cells: One row or multiple rows(table)
        :param types: e.g. float - for all cells, [str, float, float] for every row
        :return:
        """
        if not cells or types is None:
            return cells
        if isinstance(cells[0], str):
            # one row
            if not isinstance(types, tuple):
                types = [types] * len(cells)
            assert len(cells) == len(types), f'Length must be equal! {(cells, types)}'
            cells = [t(c) if t else c for c, t in zip(cells, types)]
        elif isinstance(cells[0], (list, tuple)):
            # table
            # noinspection PyTypeChecker
            cells = [TensileTest.parse_types(cell, types) for cell in cells]
        # logger.debug(f'split line: {cells}')
        return cells

    def rename(self, names: Union[List[str], Callable] = None):
        """
        If names is :
         - None     : set names as str(index), equals callable "str"
         - Callable : names(index), lambda i: 'sample-'+str(i)
        :param names:
        :return:
        """
        if names is None:
            names = [str(i) for i in self.info.index]
        elif isinstance(names, Callable):
            names = [names(i) for i in self.info.index]
        if len(names) != self.info.shape[0]:
            raise ValueError(f'样品名数量不一致: {len(names)}!={self.info.shape[0]}')
        if len(set(names)) != len(names):
            raise ValueError(f'存在重复样品名: {names}')
        self.info['name'] = names

    def __str__(self):
        return f'<{self.__class__.__name__} {self.serial_name} ({len(self.tensile_data)} samples)>'

    def __len__(self):
        return len(self.tensile_data)

    def __add__(self, other):
        assert isinstance(other, TensileTest)
        info = pd.concat([self.info, other.info], axis=0).reset_index(drop=True)
        res = TensileTest(info=info, data=self.tensile_data + other.tensile_data)
        return res

    def __getitem__(self, item):
        """
        Examples:

        - a[0], a[0:3], a[0:5:2]
        - a['name0'], a['name0':'name3'], a['name0':'name5':2]
        - a[(1,3,'name2',6,'name0')]

        :param item:
        :return: A new sliced TensileTest instance
        """
        names = self.info['name']
        if isinstance(item, slice):
            start_index = self.name_to_index(item.start) if item.start is not None else None
            stop_index = self.name_to_index(item.stop) if item.stop is not None else None
            int_slice = slice(start_index, stop_index, item.step)
            indices = range(*int_slice.indices(len(self)))
        elif isinstance(item, str):
            indices = [names[names == item].index[0]]
        elif isinstance(item, int):
            indices = [item]
        elif isinstance(item, Iterable):
            indices = self.name_to_index(item)
        else:
            raise KeyError(item)
        info = self.info.loc[indices].reset_index(drop=True)
        dfs = [self.tensile_data[i].copy() for i in indices]
        return TensileTest(info=info, data=dfs, serial_name=self.serial_name)

    def name_to_index(self, keys: Union[int, str, Iterable[Union[int, str]]] = None) -> Union[int, List[int]]:
        """
        若样品名重复，则只取第一个
        :param keys: key can be index(int) or name(str)
        :return:
        """
        names = self.info['name']
        if isinstance(keys, int):
            return keys
        elif isinstance(keys, str):
            return names[names == keys].index[0]
        else:
            return [names[names == key].index[0] if isinstance(key, str) else key for key in keys]

    def set_stress_unit(self, unit='Pa'):
        """
        WARNING: re-calculate after set new unit!!!

        :param unit:
        :return:
        """
        assert unit in ('Pa', 'kPa', 'MPa'), unit
        self.units['stress'] = unit
        # p_unit = self.units.get('stress', 'Pa')
        # ratio = to_si_unit(1.0, p_unit) / to_si_unit(1.0, unit)
        # for df in self.tensile_data:
        #     df['stress'] = df['stress'] * ratio

    # not used yet
    def filter_data_by_index(self, remains: list = None, drops: list = None):
        assert remains is not None and drops is not None, 'only remains or only drops'
        if remains is None and drops is None:
            return
        if drops is not None:
            drops = self.name_to_index(drops)
            remains = [i for i in self.info.index if i not in drops]
        else:
            remains = self.name_to_index(remains)
        self.info = self.info.loc[remains].reset_index(drop=True)
        self.tensile_data = [self.tensile_data[i] for i in remains]

    # 数据处理
    def smooth(self, method='savgol', **kwargs):
        """Smooth origin `load` column data, please ensure cal_stress_strain() after smooth
        DON'T use when there is break change or TODO: set smooth range?
        """
        if method == 'savgol':
            from scipy.signal import savgol_filter
            window_length = kwargs.pop('window_length', 11)
            polyorder = kwargs.pop('polyorder', 5)
            for df in self.tensile_data:
                df['load'] = savgol_filter(df['load'], window_length, polyorder, **kwargs)
        else:
            raise NotImplementedError

    # calculation
    def calculate(self):
        stress_ratio = 1.0 / to_si_unit(1.0, self.units.get('stress', 'Pa'))
        for i in self.info.index:
            # logger.debug(f'calculate ss: index={i}')
            df = self.tensile_data[i]
            strain = df['strain'] = df['displacement'] / self.info.loc[i, 'length']
            stress = df['stress'] = df['load'] / self.info.loc[i, 'area'] * stress_ratio
            df['true_strain'] = np.log(1 + df['strain'])
            df['true_stress'] = np.multiply(df['stress'], 1 + df['strain'])
            self.info.loc[i, 'fracture_energy'] = np.multiply(self.info.loc[i, 'length'], simps(stress, strain))
            # break_peaks = find_peaks(stress)[0]
            # break_index = break_peaks[-1] if len(break_peaks) > 0 else (len(stress) - 1)
            # self.info.loc[i, 'break_elongation'] = strain[break_index]
            # self.info.loc[i, 'break_stress'] = stress[break_index]

    def cal_init_modulus(self, x0=None, xlim=(0.05, 0.1), deg=3, plot=True):
        """

        :param x0: 求x处的初始模量，默认 xlim[0]
        :param xlim: 拟合区间，一般测试起始阶段不稳定，不宜选用类似[0,*]的范围
        :param deg: 多项式拟合的次数
        :param plot: 显示拟合的多项式曲线
        :return:
        """
        if x0 is None:
            x0 = xlim[0]

        for i in self.info.index:
            strain, stress = self.tensile_data[i]['strain'], self.tensile_data[i]['stress']

            a, b = get_index_range(strain, xlim)
            x, y = strain.loc[a:b], stress.loc[a:b]
            formula = np.poly1d(np.polyfit(x, y, deg))

            xx = np.linspace(0, xlim[1], 50)
            yy = formula(xx)

            self.info.loc[i, 'initial_modulus'] = (formula(x0 + 1e-6) - formula(x0)) / 1e-6
            if plot:
                color = next(get_color_cycler())['color']
                plt.plot(strain.loc[0:b], stress.loc[0:b], color=color, linestyle='-', label=self.info.loc[i, 'name'])
                plt.plot(xx, yy, color=color, linestyle='--', linewidth=0.5)
                plt.gca().set_xlim(left=0, right=None)
                # plt.gca().set_ylim(bottom=0, top=None)

        if plot:
            plt.legend()
            plt.title('Initial Modulus Fitting')
            plt.xlabel('λ')
            plt.ylabel(f'σ ({self.units.get("stress", "Pa")})')
            plt.axvspan(*xlim, facecolor='yellow', alpha=0.5)
            plt.axvline(x0, color='red', linewidth=2, linestyle='--')

    def cal_break_point(self, method='maxproduct', plot=True):
        """

        :param method: maxproduct(default), max, peak
        :param plot:
        :return:
        """
        for i, df in enumerate(self.tensile_data):
            stress = df['stress']
            strain = df['strain']
            if method == 'maxproduct':
                break_index = (stress * strain).idxmax()
            elif method == 'peak':
                break_peaks = find_peaks(stress)[0]
                break_index = break_peaks[-1] if len(break_peaks) > 0 else (len(stress) - 1)
            elif method == 'max':
                break_index = stress.idxmax()
            elif isinstance(method, float):
                # TODO:
                # gradient = np.divide(np.gradient(stress), stress)  # 每一采样时刻相对当前值的变化率
                # cmp = gradient[gradient < -method]
                # if True in cmp:
                #     break_index = len(cmp) - 1 - list(reversed(cmp)).index(True)
                # else:
                #     break_index = len(cmp) - 1
                raise NotImplementedError
            else:
                raise ValueError(method)
            self.info.loc[i, 'break_elongation'] = strain[break_index]
            self.info.loc[i, 'break_strength'] = stress[break_index]
            if plot:
                color = next(get_color_cycler())['color']
                plt.plot(strain, stress, color=color, markevery=[break_index], marker='o', markersize=7,
                         markerfacecolor='none', label=self.info.loc[i, 'name'])
        if plot:
            plt.gca().set_xlim(left=0, right=None)
            plt.gca().set_ylim(bottom=0, top=None)
            plt.title('Breaking Point')
            plt.xlabel('λ')
            plt.ylabel(f'σ ({self.units.get("stress", "Pa")})')
            plt.legend()

    def draw(self, true_ss=False, color=None):
        """
        Draw multiple stress-strain line in one plot
        :return:
        """
        x_key, y_key = ['true_strain', 'true_stress'] if true_ss else ['strain', 'stress']
        all_indices = list(self.info.index)
        verify_unique(all_indices)
        verify_unique([self.info.loc[i, 'name'] for i in all_indices])

        plt.figure('ss', facecolor='white')
        for i in all_indices:  # type:int
            plt.plot(self.tensile_data[i][x_key], self.tensile_data[i][y_key],
                     label=self.info.loc[i, 'name'], color=color)
        plt.gca().set_xlim(left=0, right=None)
        plt.gca().set_ylim(bottom=0, top=None)
        plt.title(f'Stress-strain curves of {self.serial_name}')
        plt.xlabel('λ')
        plt.ylabel(f'σ ({self.units.get("stress", "Pa")})')
        plt.legend()

    # noinspection PyUnusedLocal
    @classmethod
    def barplot(cls, group, key, error_bar_type='', plot_type='line'):
        # type: (List[TensileTest],str,str,str)->None
        """
        group 中的TensileTest实例必须已完成各类计算
        :param group:
        :param key: 'break_strength'
        :param error_bar_type:
        :param plot_type:
        :return:
        """
        plt.figure(f'Barplot-{key}', facecolor='white')
        stats = [statistics_of_columns(item.info) for item in group]
        y = [s.loc['mean', key] for s in stats]
        yerr = [s.loc['std', key] for s in stats]
        x = np.arange(len(y))
        x_labels = [item.serial_name for item in group]
        if plot_type == 'line':
            plt.errorbar(x, y, yerr=yerr, linestyle='-')
        elif plot_type == 'bar':
            plt.bar(x, y, yerr=yerr)
        else:
            raise NotImplementedError(plot_type)
        plt.gca().set_xticks(x)
        plt.gca().set_xticklabels(x_labels)
        plt.ylabel(key)
        plt.title('Statistics')
        pass


# %%
if __name__ == '__main__':
    pass
