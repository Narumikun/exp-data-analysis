import csv
from io import StringIO
from typing import List, Tuple, Dict, Sequence, Iterable, Optional, Union, Callable, Type  # noqa

import matplotlib.pyplot as plt
import numpy as np  # noqa
import pandas as pd  # noqa


def get_color_cycler(ax=None):
    if ax is None:
        ax = plt.gca()
    # noinspection PyProtectedMember
    return ax._get_lines.prop_cycler


def parse_csv_str(text: str, drop_empty_trailing=True):
    """
    """
    reader = csv.reader(StringIO(text))
    rows = []
    for row in reader:
        if drop_empty_trailing:
            if row and row[-1] == '':
                row.pop()
        rows.append(row)
    # print(rows[0:min(10, len(rows))])
    return list(rows)


def validate_unit(valid: Iterable[str], unit: str) -> Optional[str]:
    for i, v in enumerate(valid):
        if v.lower() == unit.lower():
            return v
