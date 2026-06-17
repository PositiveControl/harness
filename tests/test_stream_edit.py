"""Tests for stream_edit (Candidate A of harness-bw27).

Cover three boundaries: the sandbox (workspace-escape rejection,
metachar rejection, args validation), the capture path (paths and
stdin sources, oversized output truncation), and the in-place path
(per-file overwrite, error paths). The bench's Phase-3 model-in-loop
work scores end-to-end behaviour; this file pins the unit-level
guarantees those benches will rely on.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.tools.stream_edit import StreamEditArgs, StreamEditTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def tool(workspace: Path) -> StreamEditTool:
    return StreamEditTool(root=workspace)


# --- construction -----------------------------------------------------------


def test_constructor_resolves_real_binaries(tool: StreamEditTool) -> None:
    for verb in ("awk", "sed", "cut", "tr"):
        assert tool.binaries[verb].endswith(verb), f"{verb} bin lookup looks wrong"
        assert Path(tool.binaries[verb]).exists(), f"{verb} bin missing"


def test_constructor_with_injected_binaries(workspace: Path) -> None:
    """Tests can stub the binary map; production calls let __post_init__
    resolve real paths."""
    real_awk = shutil.which("awk")
    assert real_awk is not None
    tool = StreamEditTool(
        root=workspace,
        binaries={"awk": real_awk, "sed": real_awk, "cut": real_awk, "tr": real_awk},
    )
    assert tool.binaries["awk"] == real_awk


def test_constructor_rejects_incomplete_binaries(workspace: Path) -> None:
    with pytest.raises(ValueError, match="missing verbs"):
        StreamEditTool(root=workspace, binaries={"awk": "/usr/bin/awk"})


# --- args validation --------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["1 | xargs cat"],
        ["{print} | sort"],
        ['{print | "sort"}'],
    ],
)
def test_metachars_rejected(tool: StreamEditTool, workspace: Path, args: list[str]) -> None:
    """Only `|` is blocked — the rest of the shell-ish chars
    (`;`, `&`, `$`, `>`, `<`, backtick) are core awk/sed syntax or
    inert replacement data."""
    (workspace / "in.txt").write_text("hello\n")
    with pytest.raises(ValueError, match="disallowed shell metachars"):
        tool.call(tool="awk", args=args, paths=["in.txt"])


@pytest.mark.parametrize(
    ("verb", "args"),
    [
        ("awk", ["{print $3}"]),  # `$` is field-ref
        ("awk", ["{print $1; print $2}"]),  # `;` is awk statement separator
        ("sed", ["s/foo/&/g"]),  # `&` is sed back-reference
        ("awk", ['{print > "out.txt"}']),  # `>` is awk redirect to a sandboxed file
        ("awk", ["{print `x`}"]),  # backtick is inert with shell=False (4d11ea3f)
    ],
)
def test_awk_sed_syntax_passes_metachar_filter(
    tool: StreamEditTool, workspace: Path, verb: str, args: list[str]
) -> None:
    """Real awk/sed scripts must not be rejected by the metachar gate.
    Pairs with ``test_metachars_rejected`` to lock the boundary."""
    (workspace / "in.txt").write_text("a b c\n")
    tool.call(tool=verb, args=args, paths=["in.txt"])


def test_backtick_template_literal_replacement_passes(
    tool: StreamEditTool, workspace: Path
) -> None:
    """loop_run=4d11ea3f: a sed substitution writing a JS template
    literal — backticks + `${score}` in the REPLACEMENT — was rejected
    by the metachar gate, knocking the model off its edit path. With
    shell=False the backtick is an inert byte; the substitution must
    run and land the literal in the file."""
    (workspace / "game.js").write_text('ctx.fillText("Score: 0", 10, 20);\n')
    result = tool.call(
        tool="sed",
        args=['s/ctx\\.fillText("Score: 0", 10, 20);/ctx.fillText(`Score: ${score}`, 10, 20);/'],
        paths=["game.js"],
        in_place=True,
    )
    assert "rewrote 1 file(s)" in result
    assert (workspace / "game.js").read_text() == "ctx.fillText(`Score: ${score}`, 10, 20);\n"


def test_empty_args_rejected(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "in.txt").write_text("hello\n")
    with pytest.raises(ValueError, match="non-empty list"):
        tool.call(tool="awk", args=[], paths=["in.txt"])


def test_multiline_arg_redirects_to_block_editors(tool: StreamEditTool, workspace: Path) -> None:
    """loop_run=ed1e2582: a literal newline in an args element is the
    'pasting a multi-line block through sed s///' tell. Reject it BEFORE
    the verb chokes, and steer to edit_file / write_file / python_stream."""
    (workspace / "game.js").write_text("function update(dt) {}\n")
    with pytest.raises(ValueError, match="literal newline") as exc:
        tool.call(tool="sed", args=["s/x/a\n      b/"], paths=["game.js"])
    msg = str(exc.value)
    assert "edit_file" in msg
    assert "write_file" in msg


def test_metachar_message_steers_to_block_editors(tool: StreamEditTool, workspace: Path) -> None:
    """The `|` rejection now names the right tools for a multi-line insert
    instead of the unhelpful 'compose multiple calls' (loop_run=ed1e2582
    crammed a 40-line block + pipe into one sed program)."""
    (workspace / "in.txt").write_text("hello\n")
    with pytest.raises(ValueError, match="disallowed shell metachars") as exc:
        tool.call(tool="awk", args=["{print} | sort"], paths=["in.txt"])
    assert "edit_file" in str(exc.value)


def test_args_as_string_rejected_with_corrective_message(
    tool: StreamEditTool, workspace: Path
) -> None:
    # harness-vszp: the model sometimes passes args as a single joined
    # string instead of an argv list. The error must show the expected
    # list shape (corrective), not just the type mismatch.
    (workspace / "in.txt").write_text("hello\n")
    with pytest.raises(TypeError, match=r"list of argv strings"):
        tool.call(tool="sed", args="s/old/new/g", paths=["in.txt"])  # type: ignore[arg-type]


def test_jsonish_string_list_coerced_via_schema() -> None:
    """loop_run=ed1e2582 / 135f0d99: small models emit a JSON-stringified
    list for a list field (args / paths). The before-validator unwraps it
    instead of failing with a bare 'Input should be a valid list'."""
    validated = StreamEditArgs.model_validate(
        {"tool": "sed", "args": '["-E", "s/a/b/"]', "paths": '["game.js"]'}
    )
    assert validated.args == ["-E", "s/a/b/"]
    assert validated.paths == ["game.js"]


def test_jsonish_coercion_preserves_int_to_str() -> None:
    """A stringified list of line numbers coerces to strings, matching
    the existing coerce_numbers_to_str contract for real-list args."""
    validated = StreamEditArgs.model_validate({"tool": "sed", "args": "[290, 320]"})
    assert validated.args == ["290", "320"]


def test_plain_string_args_still_rejected_by_schema() -> None:
    """A non-JSON joined string is NOT a stringified list — it must still
    fail validation. Only the genuine serialized-list shape (`[...]`) is
    unwrapped. The rejection carries corrective steering (the argv-list
    shape) rather than the bare 'Input should be a valid list'."""
    with pytest.raises(ValidationError, match=r"one element per argv slot"):
        StreamEditArgs.model_validate({"tool": "sed", "args": "s/old/new/g"})


def test_bracketed_sed_program_string_rejected_with_steer() -> None:
    """loop_run=fec79051: the model emitted a sed line range wrapped in
    brackets as one string — `[`/`]`-delimited but neither valid JSON nor
    valid sed. It must reject with the argv-shape steer, not coerce."""
    with pytest.raises(ValidationError, match=r"no surrounding brackets"):
        StreamEditArgs.model_validate({"tool": "sed", "args": "[/^start/,/^}/]s/old/new/"})


def test_plain_string_paths_rejected_with_path_steer() -> None:
    """A bare path string fails too, but with a path-flavored hint that
    names `paths=[...]` — not the args/argv message."""
    with pytest.raises(ValidationError, match=r"paths=\["):
        StreamEditArgs.model_validate({"tool": "sed", "args": ["s/a/b/"], "paths": "game.js"})


def test_oversized_arg_rejected(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "in.txt").write_text("hello\n")
    huge = "a" * (4 * 1024 + 1)
    with pytest.raises(ValueError, match="bytes"):
        tool.call(tool="awk", args=[huge], paths=["in.txt"])


def test_argv_length_capped(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "in.txt").write_text("hello\n")
    with pytest.raises(ValueError, match="argv has"):
        tool.call(tool="awk", args=["x"] * 33, paths=["in.txt"])


def test_unknown_verb_rejected(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "in.txt").write_text("hello\n")
    with pytest.raises(ValueError, match="unknown tool"):
        tool.call(tool="grep", args=["x"], paths=["in.txt"])


# --- path validation --------------------------------------------------------


def test_path_outside_workspace_rejected(tool: StreamEditTool, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    with pytest.raises(ValueError, match="escapes workspace"):
        tool.call(tool="awk", args=["{print}"], paths=["../outside.txt"])


def test_missing_path_raises(tool: StreamEditTool) -> None:
    with pytest.raises(FileNotFoundError):
        tool.call(tool="awk", args=["{print}"], paths=["nope.txt"])


def test_directory_rejected(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "subdir").mkdir()
    with pytest.raises(IsADirectoryError):
        tool.call(tool="awk", args=["{print}"], paths=["subdir"])


# --- input source mutex -----------------------------------------------------


def test_both_paths_and_stdin_rejected(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "in.txt").write_text("hi\n")
    with pytest.raises(ValueError, match="either"):
        tool.call(tool="awk", args=["{print}"], paths=["in.txt"], stdin="x\n")


def test_neither_paths_nor_stdin_rejected(tool: StreamEditTool) -> None:
    with pytest.raises(ValueError, match="one of"):
        tool.call(tool="awk", args=["{print}"])


def test_oversized_stdin_rejected(tool: StreamEditTool) -> None:
    too_big = "x" * (256 * 1024 + 1)
    with pytest.raises(ValueError, match="max 262144"):
        tool.call(tool="awk", args=["{print}"], stdin=too_big)


# --- happy-path capture -----------------------------------------------------


def test_awk_extract_column_from_file(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "log.txt").write_text("a b c\nd e f\ng h i\n")
    out = tool.call(tool="awk", args=["{print $2}"], paths=["log.txt"])
    assert out == "b\ne\nh\n"


def test_awk_extract_column_from_stdin(tool: StreamEditTool) -> None:
    out = tool.call(tool="awk", args=["{print $1}"], stdin="alpha 1\nbeta 2\n")
    assert out == "alpha\nbeta\n"


def test_sed_substitute_capture(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "in.txt").write_text("foo foo\nfoo bar\n")
    out = tool.call(tool="sed", args=["s/foo/baz/g"], paths=["in.txt"])
    assert out == "baz baz\nbaz bar\n"


def test_cut_delimited_field(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "csv.txt").write_text("a,b,c\nd,e,f\n")
    out = tool.call(tool="cut", args=["-d", ",", "-f", "2"], paths=["csv.txt"])
    assert out == "b\ne\n"


def test_tr_uppercase(tool: StreamEditTool) -> None:
    out = tool.call(tool="tr", args=["a-z", "A-Z"], stdin="hello\n")
    assert out == "HELLO\n"


def test_multiple_paths_concatenate(tool: StreamEditTool, workspace: Path) -> None:
    """awk's ``FILENAME`` builtin proves the verb saw both inputs in order."""
    (workspace / "a.txt").write_text("one\n")
    (workspace / "b.txt").write_text("two\n")
    out = tool.call(tool="awk", args=["{print FILENAME, $0}"], paths=["a.txt", "b.txt"])
    assert "a.txt one" in out
    assert "b.txt two" in out


# --- in-place ---------------------------------------------------------------


def test_in_place_requires_paths(tool: StreamEditTool) -> None:
    with pytest.raises(ValueError, match="in_place"):
        tool.call(tool="sed", args=["s/x/y/"], stdin="x\n", in_place=True)


def test_in_place_sed_rewrites_each_file(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "a.py").write_text("foo and foo\n")
    (workspace / "b.py").write_text("foo bar\n")
    summary = tool.call(
        tool="sed",
        args=["s/foo/bar/g"],
        paths=["a.py", "b.py"],
        in_place=True,
    )
    assert "rewrote 2 file(s)" in summary
    assert (workspace / "a.py").read_text() == "bar and bar\n"
    assert (workspace / "b.py").read_text() == "bar bar\n"


def test_in_place_reports_per_file_delta(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "a.py").write_text("foo and foo\n")
    summary = tool.call(tool="sed", args=["s/foo/x/g"], paths=["a.py"], in_place=True)
    # "foo and foo" (11) -> "x and x" (7); newline preserved; delta = -4
    assert "a.py: -4 bytes" in summary
    assert (workspace / "a.py").read_text() == "x and x\n"


# --- failure surface --------------------------------------------------------


def test_nonzero_exit_returned_as_text(tool: StreamEditTool, workspace: Path) -> None:
    (workspace / "in.txt").write_text("hi\n")
    out = tool.call(tool="awk", args=["-bogus-flag"], paths=["in.txt"])
    assert "exit=" in out
    assert "stderr" in out


def test_capture_truncates_oversized_stdout(tool: StreamEditTool, workspace: Path) -> None:
    """Generate >512 KB of output from sed and confirm the truncation
    marker shows up."""
    src = workspace / "big.txt"
    src.write_text("x\n" * 200_000)
    out = tool.call(tool="sed", args=["s/x/yyyyy/"], paths=["big.txt"])
    assert "[truncated at" in out
    assert len(out.encode("utf-8")) <= 512 * 1024 + 200  # cap + marker


def test_timeout_message_returned(workspace: Path) -> None:
    """A short timeout against a deliberately spinning awk program
    produces a timeout marker."""
    tool = StreamEditTool(root=workspace, timeout_seconds=0.2)
    (workspace / "in.txt").write_text("x\n")
    out = tool.call(
        tool="awk",
        args=["BEGIN{while(1){}} {print}"],
        paths=["in.txt"],
        timeout=0.2,
    )
    assert "timed out" in out
