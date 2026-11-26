"""`finding_text`: a finding's numbers in the style, the house style by default (#36).

Oracle: the number written by hand at the style's decimals.
"""

from dazflow.core.finding import Finding
from dazflow.render import finding_text, load_style

RAW = 0.3939399476640034


def finding():
    return Finding(template="a scaled error of {{ mase }}", values={"mase": RAW})


def test_the_text_keeps_the_raw_number():
    assert finding().text == f"a scaled error of {RAW}"


def test_the_default_style_writes_two_decimals():
    assert finding_text(finding()) == "a scaled error of 0.39"


def test_decimals_3_writes_three():
    style = load_style({"numbers": {"decimals": 3}})
    assert finding_text(finding(), style) == "a scaled error of 0.394"


def test_no_style_is_the_house_style():
    assert finding_text(finding()) == finding_text(finding(), load_style(None))
