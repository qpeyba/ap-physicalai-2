"""Restore the bundled Panda assets into the installed Playground package."""
from pathlib import Path
import tarfile
from mujoco_playground._src import mjx_env

archive = Path(__file__).resolve().parent / "panda_assets.tar.gz"
target = Path(mjx_env.MENAGERIE_PATH)
target.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as bundle:
    for member in bundle.getmembers():
        path = (target / member.name).resolve()
        if target.resolve() not in path.parents or member.issym() or member.islnk():
            raise ValueError(f"Unsafe archive entry: {member.name}")
    bundle.extractall(target)
print(f"Panda assets restored: {target}")
