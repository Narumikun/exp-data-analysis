import logging
import math
import pickle
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
    def __init__(self, serial_name='Tensile', fps=None, info=None, data=None, units=None):
        # type:(str,Union[str,List[str]],pd.DataFrame,List[pd.DataFrame],Dict)->TensileTest
        """定义一组拉伸测试。

        Parameters
        ----------
        serial_name : str
            测试名称
        fps : str, List[str]
            若提供测试数据csv文件的路径，则读取数据
        info :

        data :


        Returns
        -------

        """
        assert fps is None or (info is None and data is None), '不能同时提供文件名和数据'
        self.serial_name = serial_name
        self.info: pd.DataFrame = info if info is not None else pd.DataFrame()
        self.tensile_data: List[pd.DataFrame] = data if data is not None else []
        self.units: Dict[str, str] = units or {'length': 'm', 'stress': 'Pa'}
        if fps:
            self.read_csv(fps)

    def read_csv(self, fps: Union[str, List[str]], encoding: str = 'GBK'):
        """读取拉伸机软件经"导出所有"这一选项导出的csv数据文件

        文件从上至下可分割为以下几个groups:
        0-测试信息, 1-样品信息, 2-计算结果, 3~n-各曲线数据。每个group之间有一空行分割。

        Parameters
        ----------
        fps :
            一个或多个被分割的文件路径, 若经过手动修改，务必保证格式不变，如同一列两组数据之间有一空行等。
            如 a.csv, [a-1.csv, a-2.csv]
        encoding : str, default 'GBK'
            岛津软件采用中文界面，导出含中文的csv文件，编码默认使用GBK或GB2313，流行使用的为UTF8为主，若出现乱码再修改

        Returns
        -------


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
                name, diameter, length = self._parse_types(row, (str, float, float))
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
            df = pd.DataFrame(self._parse_types(rows[3:], float), columns=['time', 'load', 'displacement'])
            logger.debug(f'No.{sample_index}-"{sample_name}": {df.size}')
            for col, unit in zip(df.columns, data_units):
                df[col] = to_si_unit(df[col], unit)
            self.tensile_data.append(df)

    def export(self, samples: Iterable = None, cols=('strain', 'stress'), clipboard: bool = None, fp: str = None,
               **kwargs):
        """将指定样品samples的数据列cols合并成一张表格拷贝到剪切板或保存到文件

        Parameters
        ----------
        samples : sequence
            list of sample name or index, 默认所有样品
        cols : sequence, default ('strain', 'stress')
            列名，默认strain和stress两列 (Default value = ('strain')
        clipboard : bool
            True则保存到剪切板以便在Origin等中复制
        fp : str
            保存到CSV文件
        **kwargs :
            其余可选参数传递给DataFrame.to_clipboard()或to_csv()，
            若使用额外关键字参数，建议两者不要同时使用clipboard和fp，不能区分kwargs该传递给哪一个

        Returns
        -------

        """
        assert not (kwargs and clipboard and fp), '使用额外关键字参数时不能同时保存到剪切板和文件'
        if samples is None:
            indices = range(len(self))
        else:
            indices = self.name_to_index(samples)
        # assert
        df = pd.DataFrame()
        for index in indices:
            new_cols = [self.info.loc[index, 'name'] + '_' + col for col in cols]
            df[new_cols] = self.tensile_data[index][list(cols)]
        if clipboard:
            df.to_clipboard(**kwargs)
        if fp:
            df.to_csv(fp, encoding='utf8', **kwargs)

    def dump(self, fp: str):
        """利用pickle保存TensileTest实例

        Parameters
        ----------
        fp : str
            保存的文件名，建议使用后缀 *.pkl

        Returns
        -------

        """
        with open(fp, 'wb')as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(fp: str):
        """从pickle文件中加载对象

        Parameters
        ----------
        fp: str
            文件名, dump()所保存的pkl文件

        Returns
        -------

        """
        with open(fp, 'rb')as fd:
            return pickle.load(fd)

    @staticmethod
    def _parse_types(cells: Union[List[str], List[List[str]]], types: Union[Sequence, Type] = None):
        """对一行或多行数据进行类型转换。

        Parameters
        ----------
        cells : sequence
            一行(一位数组)或多行数据(二维数组)
        types :
            需要转化的类型，有效类型: str, int, float 或Callable自定义转换函数。
            - 若为单个类型，则所有单元格转换为该类型
            - 若为一组类型，如(str,int,float)，则每行分别转化为对应类型，确保长度一致

        Returns
        -------

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
            cells = [TensileTest._parse_types(cell, types) for cell in cells]
        # logger.debug(f'split line: {cells}')
        return cells

    def rename(self, names: Union[List[str], Callable] = None):
        """重命名各条曲线。

        Parameters
        ----------
        names : List[str], Callable, default None
            若为None, 则以序号作为名称，0开始。
            若为一组str，保证长度一致。
            若为Callable, 即函数或lambda表达式，如 lambda index: 'sample'+index，输入为序号(int,从0开始)，输出为str

        Returns
        -------

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

    def remove(self, index_or_name: Union[int, str]):
        if isinstance(index_or_name, str):
            assert index_or_name in self.info['name'], f'{index_or_name} 不在列表中'
            index = list(self.info['name']).index(index_or_name)
        else:
            assert 0 <= index_or_name < len(self), f'{index_or_name} not in [0, {len(self.tensile_data)})'
            index = index_or_name
        self.tensile_data.pop(index)
        self.info.drop(index, inplace=True)
        self.info.reset_index(drop=True, inplace=True)

    def __str__(self):
        return f'<{self.__class__.__name__} {self.serial_name} ({len(self.tensile_data)} samples)>'

    def __len__(self):
        return len(self.tensile_data)

    def __add__(self, other):
        """加法实现

        注意：`serial_name`和`units`均采用前者的数据，执行加法前请确保`units`一致
        """
        assert isinstance(other, self.__class__)
        info = pd.concat([self.info, other.info], axis=0).reset_index(drop=True)
        res = TensileTest(serial_name=self.serial_name, units=self.units, info=info,
                          data=self.tensile_data + other.tensile_data)
        return res

    def __getitem__(self, item):
        """
        索引及切片实现

        有效的索引格式:

        ========  =================  ==================
        Type      Example            Description
        ========  =================  ==================
        int       data[0]            序号索引
        str       data["s1"]         样品名索引
        Sequence  data[[1,"s1"]]     以上两种索引的组合
        slice     data[0:5:2]        切片，start和stop均
                  data["s1":"s5":2]  可使用序号或样品名
        ========  =================  ==================

        Returns
        -------
        data: TensileTest
            A new TensileTest instance with sliced data

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
        """返回一组样品名(str)对应的索引(int), 允许样品名和索引同时存在

        Parameters
        ----------
        keys : Sequence
            样品名(str)或整数索引(int)的列表

        Returns
        -------

        """
        names = self.info['name']
        if isinstance(keys, int):
            return keys
        elif isinstance(keys, str):
            return names[names == keys].index[0]
        else:
            return [names[names == key].index[0] if isinstance(key, str) else key for key in keys]

    @property
    def stress_unit(self):
        return self.units.get('stress', 'Pa')

    @stress_unit.setter
    def stress_unit(self, value: str):
        """设置应力/模量单位，有效值: Pa, kPa, MPa.

        修改单位后，必须重新计算所有数据。
        """
        valid = ('Pa', 'kPa', 'MPa')
        unit = validate_unit(valid, value)
        assert unit is not None, f'单位无效: {value}, 可选值: {valid}'
        self.units['stress'] = unit

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
        """平滑原始数据

        有断裂或有滑移时不能使用 TODO: set smooth range?

        Parameters
        ----------
        method : str
            平滑方法, 可选值: 'savgol', (Default value = 'savgol')
        **kwargs :


        Returns
        -------

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
        """计算初始模量

        Parameters
        ----------
        x0 : float, default `xlim[0]`
            求x处的初始模量，默认 `xlim[0]`
        xlim : Tuple[float], default (0.05,0.1)
            拟合区间，一般测试起始阶段不稳定，不宜选用类似[0,*]的范围
        deg : int, default 3
            多项式拟合的次数
        plot : bool default True
            显示拟合的多项式曲线

        Returns
        -------

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

    def cal_break_point(self, methods='maxproduct', plot=True):
        # type:(Union[str,Dict[Union[int,str],str]],bool)->None
        """寻找断裂点

        计算结果保存在`info`中

        Parameters
        ----------
        methods: str, Dict[Any, str], default 'maxproduct'
            寻找断裂点的方法, 可选值: 'maxproduct'(default), 'max', 'peak', 'skip'(跳过)

            - 若是一个str值，则将该方法应用于所有曲线
            - 若是一个dict，如```{1:"max","line1":"peak","default":"maxpeak"}```，则分别为每条曲线指定不同的方法，
              注意为所有曲线指定值或提供默认值(default)
        plot: bool, default True
            是否画图

        Returns
        -------

        """
        if isinstance(methods, str):
            methods: Dict[Union[int, str], str] = {"default": methods}
        np.multiply()
        for i, df in enumerate(self.tensile_data):
            stress = df['stress']
            strain = df['strain']
            method = methods.get(i) or methods.get(self.info.loc[i, 'name']) or methods.get('default')
            assert method is not None, f'为{i}-{self.info.loc[i, "name"]} 指定method或提供默认值default'

            if method == 'maxproduct':
                break_index = (stress * strain).idxmax()
            elif method == 'peak':
                break_peaks = find_peaks(stress)[0]
                break_index = break_peaks[-1] if len(break_peaks) > 0 else (len(stress) - 1)
            elif method == 'max':
                break_index = stress.idxmax()
            elif method == 'skip':
                continue
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

    def draw(self, true_ss=False, colors=None):
        """Draw multiple stress-strain line in one plot

        Parameters
        ----------
        true_ss : bool, default False
            是否使用真实应力应变，默认使用工程应力应变 (Default value = False)
        colors :
            指定曲线颜色，颜色格式具体参考matplotlib.colors，常用的有:
            - 内建的8中颜色: 'b','g','r','c','m','y','k','w'
            - 灰度: '0.2', 注意是str而非float
            - html颜色: '#00ff00'
            - RGB/RGBA元组: (0.1, 0.9, 0.9)
            可指定一个或一组颜色:
            - 若为None(default), 则使用默认的color_cycler轮换颜色
            - 若只有一种颜色，则所有曲线采用相同颜色，通常在不同组数据(多个TensileTest实例)在同一张图上绘制时使用
            - 若为一组颜色，则确保和样品数一致

        Returns
        -------

        """
        if isinstance(colors, str) or (isinstance(colors, (list, tuple)) and isinstance(colors[0], float)):
            colors = [colors] * len(self)
        x_key, y_key = ['true_strain', 'true_stress'] if true_ss else ['strain', 'stress']
        all_indices = list(self.info.index)
        verify_unique(all_indices)
        verify_unique([self.info.loc[i, 'name'] for i in all_indices])

        plt.figure('ss', facecolor='white')
        for i in all_indices:  # type:int
            plt.plot(self.tensile_data[i][x_key], self.tensile_data[i][y_key],
                     label=self.info.loc[i, 'name'], color=colors[i])
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
        """Error bar plot

        Parameters
        ----------
        group: List[TensileTest]
            多组测试数据
        key: str
            需要绘制的物理参数，为`info.columns`中的一个
        error_bar_type: str
            error bar计算方法，目前采用标准差std_err
        plot_type: str
            可选: 'line'-折线图, 'bar'-柱状图

        Returns
        -------

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
