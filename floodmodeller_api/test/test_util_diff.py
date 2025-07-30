from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from floodmodeller_api._base import FMFile
from floodmodeller_api.diff import (
    check_dict_with_dataframe_equal,
    check_item_with_dataframe_equal,
    check_list_with_dataframe_equal,
)
from floodmodeller_api.util import FloodModellerAPIError, get_associated_file, handle_exception


class Dummy(FMFile):
    _filetype = "DUMMY"
    _suffix = ".dum"

    def _write(self):
        return ""

    def _read(self):
        pass

    def update(self):
        pass

    def save(self, filepath: str | Path) -> None:
        pass

    @handle_exception(when="dummy")
    def risky(self, fail=False):
        if fail:
            msg = "boom"
            raise ValueError(msg)
        return "ok"


def test_handle_exception_wraps_error():
    f = Dummy()
    with pytest.raises(FloodModellerAPIError) as exc:
        f.risky(True)
    assert "dummy DUMMY file" in str(exc.value)


def test_handle_exception_no_error():
    f = Dummy()
    assert f.risky(False) == "ok"


def test_get_associated_file(tmp_path: Path):
    original = tmp_path / "test.zzn"
    assoc = tmp_path / "test.zzl"
    original.write_text("x")
    assoc.write_text("y")
    result = get_associated_file(original, ".zzl")
    assert result == assoc


def test_get_associated_file_missing(tmp_path: Path):
    original = tmp_path / "missing.zzn"
    original.write_text("x")
    with pytest.raises(FileNotFoundError):
        get_associated_file(original, ".zzl")


def test_check_item_with_dataframe_equal_dataframe():
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"a": [1, 2], "b": [3, 5]})
    res, diff = check_item_with_dataframe_equal(df1, df2, "df", [])
    assert not res
    assert diff
    assert "left: 4" in diff[0][1]


def test_check_dict_with_dataframe_equal_missing_keys():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 1, "c": 2}
    res, diff = check_dict_with_dataframe_equal(d1, d2, "dict", [], ())
    assert not res
    assert ("dict", "Key: 'b' missing in other") in diff
    assert ("dict", "Key: c missing from first object") in diff


def test_check_list_with_dataframe_equal_value_and_length():
    res, diff = check_list_with_dataframe_equal([1, 2, 3], [1, 2, 4], "lst", [], ())
    assert not res
    assert ("lst->itm[2]", "3 != 4") in diff

    res, diff = check_list_with_dataframe_equal([1, 2], [1, 2, 3], "lst", [], ())
    assert not res
    assert ("lst", "Mismatch in list length") in diff


def test_check_item_with_dataframe_equal_special_type():
    class Special:
        def __init__(self, val):
            self.val = val

        def _get_diff(self, other):
            return False, [("val", f"{self.val}!={other.val}")]

    a = Special(1)
    b = Special(2)
    res, diff = check_item_with_dataframe_equal(a, b, "obj", [], special_types=(Special,))
    assert not res
    assert diff == [("obj->val", "1!=2")]
