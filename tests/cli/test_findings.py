"""The command line prints findings as the reports write them (#36).

Oracle: the finding sentence of ``reports/<step>/S*.md``, written through
`render/format.py` by `Project.write`, and the style's decimals.
"""

import re
import shutil

from typer.testing import CliRunner

from dazflow.cli import app

runner = CliRunner()

# A number written with more than the house style's two decimals.
LONG_FLOAT = re.compile(r"\d\.\d{3,}")


def show(project, artefact_id):
    result = runner.invoke(app, ["show", artefact_id, "-p", str(project)])
    assert result.exit_code == 0, result.output
    return result.stdout.strip()


def test_show_prints_the_reports_finding(ran_example):
    written = (ran_example / "reports" / "select" / "S6.1.md").read_text(encoding="utf-8")
    shown = show(ran_example, "S6.1")
    assert LONG_FLOAT.search(shown) is None, shown
    assert shown.replace("\\", "") in written.replace("\\", "")


def test_show_uses_the_projects_style(ran_example, tmp_path):
    project = tmp_path / "loss-rate"
    shutil.copytree(ran_example, project)
    (project / "style.yaml").write_text("numbers:\n  decimals: 3\n", encoding="utf-8")
    config = project / "project.yaml"
    config.write_text(
        config.read_text(encoding="utf-8").replace("style: null", "style: style.yaml"),
        encoding="utf-8",
    )
    assert re.search(r"\d\.\d{3}\b", show(project, "S6.1"))


def test_run_prints_no_long_floats(example):
    result = runner.invoke(app, ["run", "-p", str(example)])
    assert result.exit_code == 0, result.output
    findings = [line for line in result.stdout.splitlines() if line.startswith("  ")]
    assert len(findings) >= 7, result.stdout
    assert LONG_FLOAT.search(result.stdout) is None, result.stdout
