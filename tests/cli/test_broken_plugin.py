"""A broken plugin: the command line still starts, warns and keeps every built-in command.

The CLI builds one command per registered step when it is imported, and the
step registry loads plugins first. A plugin whose import raises must not stop
`dazflow`: the import warns on standard error with the plugin's name, and the
built-in commands are kept (design §6, §11). The plugin is a real installed
distribution, a `*.dist-info` directory on `PYTHONPATH`, and the command line
runs in its own interpreter, because the commands are fixed at import.
"""

import os
import subprocess
import sys

BUILT_IN = ("ingest", "clean", "explore", "build", "select", "forecast", "diagnose", "report")
COMMANDS = ("init", "validate", "run", "show", "plugins", "replay", *BUILT_IN)


def dazflow(site, *args):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]),
    }
    code = "import sys; from dazflow.cli import main; sys.argv = ['dazflow', *sys.argv[1:]]; main()"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def commands(site):
    """The command names the CLI registered at import, one interpreter of its own."""
    code = (
        "import typer.main; from dazflow.cli import app; "
        "print(' '.join(sorted(typer.main.get_command(app).commands)))"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=environment(site),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def environment(site):
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]),
    }


def install_broken(site):
    info = site / "dzt_broken-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: dzt-broken\nVersion: 0.1\n", encoding="utf-8"
    )
    (info / "entry_points.txt").write_text(
        "[dazflow.plugins]\nbroken = dzt_broken_plugin\n", encoding="utf-8"
    )
    (site / "dzt_broken_plugin.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")


def test_a_broken_plugin_warns_and_keeps_every_built_in_command(tmp_path):
    install_broken(tmp_path)

    result = commands(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "warning: the dazflow plugin 'broken' failed to load" in result.stderr
    assert "dzt-broken" in result.stderr
    assert "boom" in result.stderr
    assert set(COMMANDS) <= set(result.stdout.split()), result.stdout


def test_a_step_command_with_a_broken_plugin_exits_1_naming_it(tmp_path):
    install_broken(tmp_path)

    result = dazflow(tmp_path, "build", "-p", str(tmp_path / "nowhere"))

    assert result.returncode == 1
    assert "the dazflow plugin 'broken' failed to load" in result.stderr


def test_without_the_broken_plugin_there_is_no_warning(tmp_path):
    result = commands(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "warning" not in result.stderr
    assert set(COMMANDS) <= set(result.stdout.split()), result.stdout
