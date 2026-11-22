"""Exercise the released wheel through the same uvx command used by Codex."""

import argparse
import asyncio
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prepare_wheelhouse import PROJECT, TARGETS


async def check(storage, plot):
    target = TARGETS[(sys.platform, platform.machine().lower())]
    wheel, = (PROJECT / "wheels").glob("optuna_mcp_py314-*.whl")
    parameters = StdioServerParameters(
        command="uvx",
        env={key: os.environ[key] for key in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR")
             if key in os.environ},
        args=[
            "--no-config", "--isolated", "--offline", "--no-index",
            "--no-build", "--no-python-downloads", "--python", "3.14",
            "--find-links", str(PROJECT / "wheelhouse" / target),
            "--from", str(wheel), "optuna-mcp", "--storage", storage,
        ],
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            initialized = await session.initialize()
            assert "ask the user" in initialized.instructions
            tools = {tool.name for tool in (await session.list_tools()).tools}
            assert len(tools) == 27 and "ask_startup_trials" in tools, tools

            async def call(name, arguments):
                result = await session.call_tool(name, arguments)
                assert not result.isError, result.content
                return result

            missing = await session.call_tool("create_study", {"study_name": "smoke"})
            assert missing.isError and "minimize or maximize" in missing.content[0].text
            created = await call("create_study", {"study_name": "smoke", "directions": ["minimize"]})
            assert created.structuredContent["sampler_name"] == "GPSampler"
            assert created.structuredContent["n_startup_trials"] == 10
            missing = await session.call_tool("ask", {"search_space": {}})
            assert missing.isError and "Ask the user" in missing.content[0].text
            asked = await call(
                "ask",
                {"search_space": {"x": {
                    "name": "FloatDistribution", "attributes": {"low": 0, "high": 1}
                }}},
            )
            trial = asked.structuredContent["trial_number"]
            assert trial == 0
            await call("tell", {"trial_number": trial, "values": 0.25})
            best = await call("best_trial", {})
            assert best.structuredContent["values"] == [0.25]
            if plot:
                result = await call("plot_optimization_history", {})
                assert result.content[0].mimeType == "image/png"

            study = {"study_name": "gp-smoke"}
            created = await call(
                "create_study", {**study, "directions": ["minimize"], "n_startup_trials": 3}
            )
            assert created.structuredContent["n_startup_trials"] == 3
            updated = await call("create_study", {**study, "n_startup_trials": 5})
            assert updated.structuredContent["n_startup_trials"] == 5
            await call("create_study", {"study_name": "smoke"})
            resumed = await call("create_study", study)
            assert resumed.structuredContent["n_startup_trials"] == 5
            search_space = {
                "x": {"name": "FloatDistribution", "attributes": {"low": -10, "high": 10}}
            }
            arguments = {"search_space": search_space}
            batch = (await call("ask_startup_trials", arguments)).structuredContent["result"]
            assert [trial["trial_number"] for trial in batch] == list(range(5)), batch
            assert all(-10 <= trial["params"]["x"] <= 10 for trial in batch), batch
            for index, trial in enumerate(batch):
                x = trial["params"]["x"]
                await call("tell", {"trial_number": trial["trial_number"], "values": (x - 0.4) ** 2})
                if index == 0:
                    remaining = await call("ask_startup_trials", arguments)
                    assert remaining.structuredContent["result"] == batch[1:]
            finished = await call("ask_startup_trials", arguments)
            assert finished.structuredContent["result"] == []

            # The first model-based ask loads SciPy unless startup preloaded it.
            async with asyncio.timeout(30):
                asked = await call("ask", {"search_space": search_space})
            trial = asked.structuredContent
            assert trial["trial_number"] == 5, trial
            x = trial["params"]["x"]
            assert -10 <= x <= 10, trial
            await call("tell", {"trial_number": trial["trial_number"], "values": (x - 0.4) ** 2})


async def main(plot):
    with TemporaryDirectory() as directory:
        db = Path(directory) / "smoke.db"
        async with asyncio.timeout(120):
            await check(f"sqlite:///{db.as_posix()}", plot)
        assert db.is_file()
    print("MCP stdio, 27 tools, SQLite startup batch, and model-based GPSampler verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Also verify PNG rendering.")
    asyncio.run(main(parser.parse_args().plot))
