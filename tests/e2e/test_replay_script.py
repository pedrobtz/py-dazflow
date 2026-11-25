"""The replay script written to reports/document/ runs on its own (design §7.10).

Oracle: the script's exit status as a separate Python process, run from
another working directory: 0 on the example's untouched folder, and not 0
once an input has changed, so it is the replay that decides.
"""

import subprocess
import sys


def run_script(root, cwd):
    return subprocess.run(
        [sys.executable, str(root / "reports" / "document" / "replay.py")],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_the_generated_replay_script_runs_as_a_subprocess_and_exits_0(ran_example, tmp_path):
    done = run_script(ran_example, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "matches its record" in done.stdout


def test_the_script_exits_non_zero_when_an_input_changed(ran_copy, tmp_path):
    series = ran_copy / "data" / "raw" / "series.csv"
    text = series.read_text(encoding="utf-8")
    series.write_text(text.replace("1995Q2,2.009,", "1995Q2,2.019,"), encoding="utf-8")
    done = run_script(ran_copy, tmp_path)
    assert done.returncode != 0
    assert "inputs.ingest.macro.fingerprint" in done.stdout
