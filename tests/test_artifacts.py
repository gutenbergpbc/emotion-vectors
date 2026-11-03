"""Downloaded evidence cannot escape its destination or overwrite changed data."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

SPEC=importlib.util.spec_from_file_location('artifact_tools',Path(__file__).resolve().parents[1]/'tools/artifacts.py')
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def archive(tmp_path,names):
    path=tmp_path/'asset.tar.gz'
    with tarfile.open(path,'w:gz') as out:
        for name,body in names:
            info=tarfile.TarInfo(name);info.size=len(body)
            out.addfile(info,io.BytesIO(body))
    return path


def test_valid_archive_and_modified_local_file(tmp_path):
    name='data/example.json';body=b'{"id":1}\n'
    path=archive(tmp_path,[(name,body)])
    files={name:{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}}
    root=tmp_path/'repo';root.mkdir()
    MODULE.extract(path,{'files':[name]},files,root)
    assert (root/name).read_bytes()==body
    (root/name).write_bytes(b'local edit')
    with pytest.raises(ValueError,match='Refusing to replace'):
        MODULE.extract(path,{'files':[name]},files,root)


@pytest.mark.parametrize('name',['../escape','/absolute'])
def test_path_traversal_rejected(tmp_path,name):
    path=archive(tmp_path,[(name,b'x')])
    with pytest.raises(ValueError,match='Unsafe member'):
        MODULE.extract(path,{'files':[name]},{name:{'bytes':1}},tmp_path/'repo')


def test_corruption_and_unlisted_members_are_rejected(tmp_path):
    path=archive(tmp_path,[('data/a',b'x')])
    with pytest.raises(ValueError,match='Corrupt member'):
        MODULE.extract(path,{'files':['data/a']},{'data/a':{'bytes':1,'sha256':'wrong'}},tmp_path/'repo')
    with pytest.raises(ValueError,match='members differ'):
        MODULE.extract(path,{'files':[]},{},tmp_path/'repo')
