from pathlib import Path
import sys
import tomllib
from urllib.parse import unquote, urlsplit

from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

manifests = Path(sys.argv[1]).resolve() / "packaging"
with (manifests / "pylock.toml").open("rb") as file:
    lock = tomllib.load(file)

targets = {
    "windows-x64": ("win32", "AMD64", ["win_amd64"]),
    "linux-x64": (
        "linux",
        "x86_64",
        [f"manylinux_2_{minor}_x86_64" for minor in range(28, 4, -1)]
        + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"],
    ),
}
for target, (platform, machine, platforms) in targets.items():
    env = {
        "implementation_name": "cpython",
        "platform_machine": machine,
        "platform_system": "Windows" if platform == "win32" else "Linux",
        "python_version": "3.14",
        "python_full_version": "3.14.10",
        "sys_platform": platform,
    }
    ordered_tags = [
        *cpython_tags((3, 14), ["cp314"], platforms),
        *compatible_tags((3, 14), "cp314", platforms),
    ]
    ranks = {tag: rank for rank, tag in enumerate(ordered_tags)}
    requirements = ["# Generated from pylock.toml for CPython 3.14 " + target]
    for package in lock["packages"]:
        if "marker" in package and not Marker(package["marker"]).evaluate(env):
            continue
        candidates = []
        for wheel in package["wheels"]:
            url = wheel["url"]
            filename = unquote(Path(urlsplit(url).path).name)
            _, _, _, tags = parse_wheel_filename(filename)
            compatible = tags & ranks.keys()
            if compatible:
                rank = min(ranks[tag] for tag in compatible)
                candidates.append((rank, url, wheel["hashes"]["sha256"]))
        if not candidates:
            raise ValueError(f"No wheel for {package['name']} on {target}")
        _, url, sha256 = min(candidates)
        requirements.append(f"{package['name']} @ {url}#sha256={sha256}")
    (manifests / f"wheels-{target}.txt").write_text(
        "\n".join(requirements) + "\n", encoding="utf-8", newline="\n"
    )
