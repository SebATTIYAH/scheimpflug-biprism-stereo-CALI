"""Packaged examples, independent of the working directory."""
import json
from pathlib import Path
from importlib import resources


def template_config(kind):
    names = {'cropped': 'cropped.example.json', 'full-frame': 'real.example.json', 'synthetic': 'synthetic.json'}
    if kind not in names:
        raise ValueError('Unknown template: ' + str(kind))
    with resources.open_text('bbs_calib.resources', names[kind], encoding='utf-8') as stream:
        return json.load(stream)


def write_template(path, kind='cropped'):
    config = template_config(kind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return path
