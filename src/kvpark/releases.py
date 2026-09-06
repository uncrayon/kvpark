"""Discover and verify published packages from the project's fixed release source."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

REPOSITORY = "https://github.com/uncrayon/kvpark"
RELEASES_API = "https://api.github.com/repos/uncrayon/kvpark/releases?per_page=100"
MAX_DOWNLOAD = 25 * 1024**2


def fetch(url, limit=MAX_DOWNLOAD):
    req = Request(url, headers={"User-Agent": "kvpark-updater", "Accept": "application/vnd.github+json"})
    with urlopen(req, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("release download exceeds the size limit")
    return data


def latest(installed):
    candidates = []
    for release in json.loads(fetch(RELEASES_API, 2 * 1024**2)):
        tag = release.get("tag_name", "")
        if release.get("draft") or not re.fullmatch(r"v[0-9][0-9A-Za-z.\-]*", tag):
            continue
        try:
            version = Version(tag[1:])
        except InvalidVersion:
            continue
        if version.is_prerelease and not Version(installed).is_prerelease:
            continue
        name = f"kvpark-{version}-py3-none-any.whl"
        for asset in release.get("assets", []):
            url = f"{REPOSITORY}/releases/download/{tag}/{name}"
            digest = asset.get("digest") or ""
            if (asset.get("name") == name and asset.get("browser_download_url") == url
                    and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)):
                candidates.append(dict(version=str(version), url=url, sha256=digest[7:],
                                       name=name, release_url=f"{REPOSITORY}/releases/tag/{tag}"))
    return max(candidates, key=lambda item: Version(item["version"]), default=None)


def download(release, destination):
    data = fetch(release["url"])
    if hashlib.sha256(data).hexdigest() != release["sha256"]:
        raise ValueError("release checksum mismatch; nothing was installed")
    path = Path(destination) / release["name"]
    path.write_bytes(data)
    try:
        validate_wheel(path, release["version"])
    except (KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise ValueError("release wheel metadata is invalid; nothing was installed") from exc
    return path


def validate_wheel(path, version):
    from email.parser import BytesParser
    with zipfile.ZipFile(path) as wheel:
        info = BytesParser().parsebytes(wheel.read(f"kvpark-{version}.dist-info/METADATA"))
        if info["Name"] != "kvpark" or Version(info["Version"]) != Version(version):
            raise ValueError("release package identity mismatch")
        compatibility = json.loads(wheel.read("kvpark/update_compat.json"))
        if compatibility != {"updater_protocol": 1, "snapshot_format": 2, "control_version": 4}:
            raise ValueError("this release requires a manual compatibility upgrade; see its release notes")
        current_python = ".".join(map(str, sys.version_info[:3]))
        if current_python not in SpecifierSet(info.get("Requires-Python", "")):
            raise ValueError("this release needs a different Python version")
        for raw in info.get_all("Requires-Dist", []):
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                installed = None
            if requirement.url or installed is None or installed not in requirement.specifier:
                raise ValueError(f"update dependency {requirement.name} in Hermes' environment first; nothing was installed")
