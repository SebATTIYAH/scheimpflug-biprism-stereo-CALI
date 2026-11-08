"""Export only reviewed source categories; never upload or initialize git."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ROOT_FILES = ('README.md', 'pyproject.toml', 'requirements-tested.txt',
              '.gitignore', 'calibrate.py',
              'calibrate.cmd', 'CONTRIBUTING.md', 'CHANGELOG.md',
              'MANIFEST.in', '.gitattributes')


def public_files(root=ROOT):
    root = Path(root).resolve()
    files = {name: root/name for name in ROOT_FILES}
    for directory, pattern in (('bbs_calib', '*.py'), ('bbs_calib/resources', '*.json'),
                               ('tests', 'test_*.py'), ('candidate_affine', '*.py'),
                               ('experiments', '*.py'), ('results', '*.json'), ('results', '*.csv')):
        for path in (root/directory).rglob(pattern):
            files[path.relative_to(root).as_posix()] = path
    for name in ('docs/public_usage.md', 'docs/model.md', 'docs/limitations.md', 'docs/optional_q_scale.md',
                 'docs/REPRODUCIBILITY.md',
                 'candidate_affine/README.md', 'experiments/README.md', '.github/workflows/tests.yml',
                 'tools/export_public.py', 'tools/audit_git_tree.py',
                 'tools/__init__.py'):
        files[name] = root/name
    # Test fixture compatibility, sourced from the packaged neutral examples.
    for name in ('real.example.json', 'cropped.example.json', 'synthetic.json'):
        files['configs/'+name] = root/'bbs_calib/resources'/name
    brown_root = root/'extensions/brown_v4'
    for path in (brown_root/'bbs_calib').glob('*.py'):
        files[path.relative_to(root).as_posix()] = path
    for name in ('joint_calibrate.py', 'joint_reconstruct.py', 'README.md',
                 'requirements.txt', 'SOURCE_MANIFEST.json', 'VALIDATION.json'):
        files['extensions/brown_v4/'+name] = brown_root/name
    files['LICENSE'] = root/'LICENSE'
    for name, path in files.items():
        if not path.is_file() or path.is_symlink() or root not in path.resolve().parents:
            raise ValueError('Missing file or unsafe source: '+name)
    return dict(sorted(files.items()))


def export(output, root=ROOT):
    files = public_files(root)
    contents = {name: path.read_bytes() for name, path in files.items()}
    # A narrow safety net, not a full secret/privacy audit.
    for name, data in contents.items():
        content = data.decode('utf-8-sig')
        if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----', content):
            raise ValueError('Private-key marker in '+name)
        if re.search(r'[A-Za-z]:[/\\]Users[/\\](?!example[/\\]|test[/\\])[^/\\\s]+[/\\]', content):
            raise ValueError('Personal user path in '+name)
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    status = dict(schema_version=1, files_sha256=manifest)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        def add(name, data):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
        for name, data in contents.items():
            add(name, data)
        add('SOURCE_MANIFEST.json', json.dumps(status, indent=2)+'\n')
    return output, len(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New source ZIP; existing files are never overwritten')
    args = parser.parse_args()
    try:
        path, count = export(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, 'ERROR: '+str(exc)+'\n')
    print('Exported %d source files: %s'%(count,path))


if __name__ == '__main__':
    main()
