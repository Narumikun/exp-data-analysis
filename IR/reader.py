import re

from base.base import *


def from_organchem_html(html_code: str, name='', use_transmittance=True):
    """
    化学专业数据库谱图页面(需登录)，右键view-source/查看源码,确保里面有很长一串tstr='4000/989;3999/989;3998/989;3997/990;...400/988'
    :param html_code:
    :param name:
    :param use_transmittance:
    :return:
    """
    res: List[str] = re.findall(r"var\s+tstr\s*=\s*'(\d+/\d+;){100,}';", html_code)
    assert len(res) > 0, 'No IR data (tstr field) in the html_code'
    points = res[0].strip(';').split(';')
    data = [[float(x) for x in p.split('/')] for p in points]
    x_name = (name + '_' if name else '') + 'wavenumber'
    y_name = (name + '_' if name else '') + ('Transmittance' if use_transmittance else 'Absorption')
    df = pd.DataFrame(data=data, columns=[x_name, y_name])
    df[y_name] = np.multiply(df[y_name], 0.1)
    return df


# %%
if __name__ == '__main__':
    df = from_organchem_html(s, 'ABS')
    df.to_clipboard(excel=False, index = False)
