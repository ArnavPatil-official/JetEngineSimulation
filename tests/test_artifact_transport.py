"""Tiny lossless-publication fixtures; no study files or computations."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import pytest

from scripts.phase8 import artifact_transport as transport


def fixture(tmp_path):
    root=tmp_path/'source';root.mkdir()
    metadata=root/'outputs/meta';artifacts=root/'outputs/artifacts';artifacts.mkdir(parents=True)
    payload=random.Random(412).randbytes(1800)
    large=artifacts/'study.json';large.write_bytes(payload)
    small=artifacts/'small.json';small.write_bytes(b'small unchanged')
    paths=[p.relative_to(root).as_posix() for p in (large,small)]
    return root,metadata,artifacts,payload,paths


def clone(tmp_path,root):
    destination=tmp_path/'clone';destination.mkdir()
    shutil.copytree(root/'outputs/meta',destination/'outputs/meta')
    return destination,destination/'outputs/meta'


def save_manifest(metadata,value):
    (metadata/'artifact_transport.json').write_text(json.dumps(value))


def test_prepare_preserves_originals_bounds_chunks_and_is_deterministic(tmp_path):
    root,metadata,artifacts,payload,paths=fixture(tmp_path)
    published,manifest=transport.prepare(root,metadata,paths,threshold_bytes=100,chunk_bytes=73)
    entry=manifest['files'][paths[0]]
    chunks=[root/item['path'] for item in entry['chunks']]
    encoded=b''.join(path.read_bytes() for path in chunks)
    assert gzip.decompress(encoded)==payload and encoded[4:8]==b'\0'*4
    assert all(0<path.stat().st_size<=73 for path in chunks)
    assert entry['sha256']==hashlib.sha256(payload).hexdigest()
    assert entry['compressed_sha256']==hashlib.sha256(encoded).hexdigest()
    assert paths[0] not in published and paths[1] in published
    assert 'outputs/meta/artifact_transport.json' in published
    assert (root/paths[0]).read_bytes()==payload and (root/paths[1]).read_bytes()==b'small unchanged'
    before=(metadata/'artifact_transport.json').read_bytes()
    repeated,repeated_manifest=transport.prepare(root,metadata,paths,threshold_bytes=100,chunk_bytes=73)
    assert repeated==published and repeated_manifest==manifest
    assert before==(metadata/'artifact_transport.json').read_bytes()
    assert not list(metadata.rglob('.part-*')) and not list(metadata.glob('.transport-manifest-*'))


def test_clone_restores_exact_bytes_and_identical_existing_files(tmp_path):
    root,metadata,artifacts,payload,paths=fixture(tmp_path)
    transport.prepare(root,metadata,paths,threshold_bytes=100,chunk_bytes=64)
    copied,copied_metadata=clone(tmp_path,root)
    assert transport.restore(copied,copied_metadata,[copied/'outputs/artifacts'])==[paths[0]]
    assert (copied/paths[0]).read_bytes()==payload
    assert transport.restore(copied,copied_metadata,['outputs/artifacts'])==[]
    assert not list(copied.rglob('.restore-*'))


def test_restore_refuses_different_existing_original_without_overwrite(tmp_path):
    root,metadata,artifacts,payload,paths=fixture(tmp_path)
    transport.prepare(root,metadata,paths,threshold_bytes=100,chunk_bytes=64)
    copied,copied_metadata=clone(tmp_path,root)
    original=copied/paths[0];original.parent.mkdir(parents=True);original.write_bytes(b'different')
    with pytest.raises(ValueError,match='Existing file differs'):
        transport.restore(copied,copied_metadata,['outputs/artifacts'])
    assert original.read_bytes()==b'different' and not list(copied.rglob('.restore-*'))


@pytest.mark.parametrize('existing',[False,True])
def test_corrupt_chunk_is_refused_even_with_identical_original(tmp_path,existing):
    root,metadata,artifacts,payload,paths=fixture(tmp_path)
    _,manifest=transport.prepare(root,metadata,paths,threshold_bytes=100,chunk_bytes=64)
    copied,copied_metadata=clone(tmp_path,root)
    chunk=copied/manifest['files'][paths[0]]['chunks'][0]['path'];chunk.write_bytes(b'corrupt')
    if existing:
        original=copied/paths[0];original.parent.mkdir(parents=True);original.write_bytes(payload)
    with pytest.raises(ValueError,match='Existing file differs'):
        transport.restore(copied,copied_metadata,['outputs/artifacts'])
    assert not (copied/paths[0]).exists() or (copied/paths[0]).read_bytes()==payload


@pytest.mark.parametrize('change',['original_sha','compressed_sha','chunk_order','destination_escape','destination_unowned','chunk_escape'])
def test_restore_rejects_manifest_corruption_and_scope_escape(tmp_path,change):
    root,metadata,artifacts,payload,paths=fixture(tmp_path)
    _,manifest=transport.prepare(root,metadata,paths,threshold_bytes=100,chunk_bytes=64)
    copied,copied_metadata=clone(tmp_path,root)
    entry=manifest['files'][paths[0]]
    if change=='original_sha':entry['sha256']='0'*64
    if change=='compressed_sha':entry['compressed_sha256']='0'*64
    if change=='chunk_order':entry['chunks'].reverse()
    if change=='chunk_escape':entry['chunks'][0]['path']='outputs/elsewhere.gz'
    if change=='destination_escape':manifest['files']={'../outside':entry}
    if change=='destination_unowned':manifest['files']={'outputs/unowned/study.json':entry}
    save_manifest(copied_metadata,manifest)
    with pytest.raises(ValueError):transport.restore(copied,copied_metadata,['outputs/artifacts'])
    assert not (copied/paths[0]).exists() and not list(copied.rglob('.restore-*'))


def test_prepare_and_restore_refuse_symlink_components(tmp_path):
    root,metadata,artifacts,payload,paths=fixture(tmp_path)
    link=artifacts/'link.json';link.symlink_to(artifacts/'study.json')
    with pytest.raises(ValueError,match='symlink'):transport.prepare(root,metadata,['outputs/artifacts/link.json'])
    transport.prepare(root,metadata,paths,threshold_bytes=100,chunk_bytes=64)
    copied,copied_metadata=clone(tmp_path,root)
    outside=tmp_path/'outside';outside.mkdir()
    (copied/'outputs/artifacts').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):transport.restore(copied,copied_metadata,['outputs/artifacts'])
    assert not list(outside.iterdir())


def test_restore_exclusive_creation_cannot_overwrite_racing_file(tmp_path,monkeypatch):
    root,metadata,artifacts,payload,paths=fixture(tmp_path)
    transport.prepare(root,metadata,paths,threshold_bytes=100,chunk_bytes=64)
    copied,copied_metadata=clone(tmp_path,root)
    original=copied/paths[0];link=os.link
    def raced_link(source,target):
        if Path(target)==original:original.write_bytes(b'racing owner')
        return link(source,target)
    monkeypatch.setattr(transport.os,'link',raced_link)
    with pytest.raises(ValueError,match='Existing file differs'):
        transport.restore(copied,copied_metadata,['outputs/artifacts'])
    assert original.read_bytes()==b'racing owner' and not list(copied.rglob('.restore-*'))


def test_small_files_stay_direct_and_default_chunk_cap_is_enforced(tmp_path):
    root,metadata,artifacts,payload,paths=fixture(tmp_path)
    published,manifest=transport.prepare(root,metadata,paths,threshold_bytes=1800,chunk_bytes=17)
    assert manifest['files']=={} and set(paths)<=set(published)
    assert transport.restore(root,metadata,['outputs/artifacts'])==[]
    with pytest.raises(ValueError,match='chunk_bytes'):
        transport.prepare(root,metadata,paths,chunk_bytes=transport.MAX_CHUNK_BYTES+1)
    with pytest.raises(ValueError):transport.prepare(root,metadata,['../outside'])
