"""Create an allowlisted code-only archive and verify all recorded digests."""
from pathlib import Path
from datetime import date
import argparse
import hashlib
import json
import re
import zipfile


ALLOWLIST = [
    ".gitignore", "README.md", "README_中文.md", "NOTICE.md", "RELEASE_NOTES.md", "requirements.txt",
    "requirements-lock.txt", "reanalysis.py", "verify_results.py", "schema.py",
    "requirements-plot.txt", "plot_revision_figures_1009.py",
    "render_environment_reference.json",
    "generate_synthetic.py", "smoke_test.py", "build_release.py",
    "synthetic_smoke_validation.json", "examples/SYNTHETIC_NOTICE.json",
    "examples/synthetic_main.csv", "examples/synthetic_extended26.csv",
    "examples/synthetic_facilities.csv",
]


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release-date", type=date.fromisoformat, default=date(2026, 10, 16))
    args = parser.parse_args()
    package, out = args.package.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((package / "synthetic_smoke_validation.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["synthetic_only"] and not report["scientific_findings_validated"]
    assert report["packaged_examples_match_synthetic_regeneration"]
    notice = json.loads((package / "examples/SYNTHETIC_NOTICE.json").read_text(encoding="utf-8"))
    assert notice["synthetic"] and not notice["real_data_read"]
    rendering = json.loads((package / "render_environment_reference.json").read_text(encoding="utf-8"))
    assert rendering["public_copy_full_render_in_separate_plotting_environment"]
    assert rendering["all_seven_public_pngs_byte_and_pixel_identical_to_frozen_figures"]
    assert rendering["public_png_match_count"] == 7
    assert rendering["private_render_outputs_excluded_from_release"]
    assert rendering["public_script_sha256"] == sha256((package / "plot_revision_figures_1009.py").read_bytes())
    content = {}
    for name in ALLOWLIST:
        path = package / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe allowlisted file: {name}")
        raw = path.read_bytes()
        if path.suffix != ".csv":
            text = raw.decode("utf-8-sig")
            if re.search(r"(?i)(?:(?<![A-Za-z0-9])[A-Z]:[\\/]|wxid_[a-z0-9]{3,}|xwechat_[a-z0-9]{3,}|/Users/[a-z0-9]|/home/[a-z0-9]|ghp_[a-z0-9]|github_pat_[a-z0-9])", text):
                raise ValueError(f"Potential private path or secret in {name}")
        content[name] = raw
    sums = "".join(f"{sha256(raw)}  {name}\n" for name, raw in sorted(content.items()))
    (package / "SHA256SUMS.txt").write_bytes(sums.encode("utf-8"))
    content["SHA256SUMS.txt"] = sums.encode("utf-8")
    archive_path = out / f"park-sports-vitality-code_{args.release_date.isoformat()}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(content.items()):
            member = zipfile.ZipInfo("park-sports-vitality-code/" + name, date_time=(*args.release_date.timetuple()[:3], 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, raw)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(content)
        for name, raw in content.items():
            assert archive.read("park-sports-vitality-code/" + name) == raw
    archive_sha = sha256(archive_path.read_bytes())
    (out / "ZIP_SHA256.txt").write_text(f"{archive_sha}  {archive_path.name}\n", encoding="utf-8")
    manifest = {"archive": archive_path.name, "archive_sha256": archive_sha, "archive_bytes": archive_path.stat().st_size, "file_count": len(content), "files": [{"name": name, "sha256": sha256(raw), "size": len(raw)} for name, raw in sorted(content.items())], "synthetic_only_data": True, "real_inputs_included": False, "published": None, "remote_publication_verified": False, "publication_status": "Local archive built; this command does not check or change remote publication status", "plot_rendering_status": "Historical validation: full adapted public plotting interface checked in a separate recorded environment; seven PNGs matched frozen figures; no new model fitting or rendering performed by this archive builder", "integrity": "All ZIP members checked byte-for-byte; CRC test passed"}
    (out / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    publication_status = {
        "status": "local_archive_built_remote_status_not_checked", "published": None,
        "planned_repository_name": "changsha-facility-vitality-revision",
        "url": "https://github.com/Shumin0829/changsha-facility-vitality-revision",
        "repository_url": "https://github.com/Shumin0829/changsha-facility-vitality-revision",
        "commit_sha": None, "repository_created": None, "code_uploaded": None,
        "remote_state_checked": False,
        "reason": "This local archive builder performs no authentication, upload, or remote verification. The repository URL alone does not establish publication of this archive or a specific commit.",
        "next_step": "Verify the actual public repository and exact commit separately. No additional software reuse licence is included.",
        "code_package_ready": True, "archive": archive_path.name,
        "archive_sha256": archive_sha, "real_inputs_included": False,
        "credentials_printed_or_written": False,
    }
    (out / "publication_status.json").write_text(json.dumps(publication_status, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()

