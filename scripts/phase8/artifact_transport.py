"""Lossless, checksum-verified transport for artifacts above GitHub's file cap.

Original files remain untouched. The manifest and deterministic gzip chunks are
ordinary Git files; restoration needs only Python's standard library.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import stat
import uuid
from pathlib import Path, PurePosixPath

MAX_CHUNK_BYTES = 50 * 1024**2
SCHEMA = 'pc-artifact-transport-v1'
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def _integer(value, name, *, minimum=1, maximum=None):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'Invalid {name}')
    return value


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or '\x00' in value:
        raise ValueError('Transport paths must be relative POSIX paths')
    parts = value.split('/')
    if any(part in ('', '.', '..') for part in parts) or PurePosixPath(value).is_absolute():
        raise ValueError('Transport path leaves its owned directory')
    return value


def _owned(root, value, *, allow_root=False):
    """Resolve ownership without following any symlink component."""
    path = Path(value)
    if path.is_absolute():
        try:
            name = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError('Transport path leaves repository') from exc
    else:
        name = path.as_posix()
    if name == '.' and allow_root:
        return root
    name = _relative(name)
    current = root
    for part in PurePosixPath(name).parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Transport refuses symlink: '+name)
    if not current.resolve().is_relative_to(root):
        raise ValueError('Transport path leaves repository')
    return current


def _regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Transport requires a regular file: '+str(path))


def _hash(path):
    _regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _identical(path, digest, size):
    _regular(path)
    if path.stat().st_size != size or _hash(path) != digest:
        raise ValueError('Existing file differs from transport: '+str(path))


def _install(temporary, target, digest, size):
    """An atomic exclusive create; an existing identical file is idempotent."""
    try:
        os.link(temporary, target)
        return True
    except FileExistsError:
        _identical(target, digest, size)
        return False


class _Chunks:
    def __init__(self, root, directory, limit):
        self.root, self.directory, self.limit = root, directory, limit
        self.stream, self.temporary, self.size = None, None, 0
        self.items, self.digest = [], hashlib.sha256()
        self.chunk_digest = None
        self.total = 0

    def write(self, data):
        length = len(data)
        view = memoryview(data)
        self.digest.update(data); self.total += length
        while view:
            if self.stream is None:
                self.temporary = self.directory / ('.part-'+uuid.uuid4().hex)
                self.stream = self.temporary.open('xb')
                self.size, self.chunk_digest = 0, hashlib.sha256()
            count = min(len(view), self.limit-self.size)
            self.stream.write(view[:count]); self.chunk_digest.update(view[:count])
            view = view[count:]; self.size += count
            if self.size == self.limit:
                self.close()
        return length

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def close(self):
        if self.stream is None:
            return
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()
        self.stream = None
        target = self.directory / f'{len(self.items):05d}.gz.part'
        try:
            digest = self.chunk_digest.hexdigest()
            _install(self.temporary, target, digest, self.size)
            self.items.append({'path':target.relative_to(self.root).as_posix(), 'sha256':digest, 'size':self.size})
        finally:
            self.temporary.unlink(missing_ok=True)

    def abort(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None
        if self.temporary is not None:
            self.temporary.unlink(missing_ok=True)


def prepare(root, metadata, paths, *, threshold_bytes=75*1024**2, chunk_bytes=MAX_CHUNK_BYTES):
    """Return ``(publish_paths, manifest)`` and preserve all original bytes.

    Files strictly larger than threshold_bytes use gzip chunks. Repeated calls
    for the same unchanged artifact set are idempotent. A different existing
    manifest or chunk is refused; publication cannot silently replace it.
    """
    root = Path(root).resolve()
    metadata = _owned(root, metadata)
    threshold_bytes = _integer(threshold_bytes, 'threshold_bytes', minimum=0)
    chunk_bytes = _integer(chunk_bytes, 'chunk_bytes', maximum=MAX_CHUNK_BYTES)
    directory = metadata/'artifact_transport'
    manifest_path = metadata/'artifact_transport.json'
    _owned(root, directory); _owned(root, manifest_path)
    originals = sorted({_owned(root, name).relative_to(root).as_posix() for name in paths})
    for name in originals:
        path = root/name
        if path == manifest_path or path.is_relative_to(directory):
            raise ValueError('Transport-managed files cannot be original inputs')
        _regular(path)
    metadata.mkdir(parents=True, exist_ok=True)
    manifest = {'schema':SCHEMA, 'metadata_path':metadata.relative_to(root).as_posix(),
                'threshold_bytes':threshold_bytes, 'chunk_bytes':chunk_bytes, 'gzip_mtime':0, 'files':{}}
    published = []
    for name in originals:
        original = _owned(root, name)
        size = original.stat().st_size
        if size <= threshold_bytes:
            published.append(name)
            continue
        expected = _hash(original)
        output = _owned(root, directory/expected)
        output.mkdir(parents=True, exist_ok=True)
        chunks = _Chunks(root, output, chunk_bytes)
        observed, count = hashlib.sha256(), 0
        try:
            with original.open('rb') as source, gzip.GzipFile(filename='', mode='wb', fileobj=chunks, mtime=0) as compressed:
                for block in iter(lambda:source.read(1024**2), b''):
                    observed.update(block); count += len(block); compressed.write(block)
            chunks.close()
        except BaseException:
            chunks.abort()
            raise
        if observed.hexdigest() != expected or count != size or _hash(original) != expected or original.stat().st_size != size:
            raise ValueError('Original changed while preparing transport: '+name)
        entry = {'sha256':expected, 'size':size, 'encoding':'gzip', 'chunks':chunks.items,
                 'compressed_sha256':chunks.digest.hexdigest(), 'compressed_size':chunks.total}
        manifest['files'][name] = entry
        published.extend(item['path'] for item in chunks.items)
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    temporary = metadata/('.transport-manifest-'+uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        _install(temporary, manifest_path, hashlib.sha256(payload).hexdigest(), len(payload))
    finally:
        temporary.unlink(missing_ok=True)
    published.append(manifest_path.relative_to(root).as_posix())
    return sorted(set(published)), manifest


class _Joined(io.RawIOBase):
    """Bounded-memory concatenation of already verified compressed chunks."""
    def __init__(self, paths):
        super().__init__()
        self.paths, self.stream = iter(paths), None
        self.digest, self.size = hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.stream is None:
                try:
                    self.stream = next(self.paths).open('rb')
                except StopIteration:
                    return 0
            count = self.stream.readinto(buffer)
            if count:
                self.digest.update(memoryview(buffer)[:count]); self.size += count
                return count
            self.stream.close(); self.stream = None

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None
        super().close()


def _digest(value):
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError('Invalid transport SHA256')
    return value


def restore(root, metadata, allowed_roots):
    """Restore missing originals exclusively; validate existing originals too.

    Caller-provided allowed_roots own destination paths. Every chunk must live
    inside this metadata directory and match its declared size and SHA256.
    Different existing originals, symlinks and malformed paths are refused.
    """
    root = Path(root).resolve()
    metadata = _owned(root, metadata)
    manifest_path = _owned(root, metadata/'artifact_transport.json')
    if not manifest_path.exists():
        return []
    _regular(manifest_path)
    if isinstance(allowed_roots, (str, Path)):
        raise ValueError('allowed_roots must be a sequence of owned directories')
    owned = [_owned(root, path, allow_root=True) for path in allowed_roots]
    if not owned:
        raise ValueError('No destination ownership roots supplied')
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if (manifest.get('schema') != SCHEMA or manifest.get('metadata_path') != metadata.relative_to(root).as_posix()
            or manifest.get('gzip_mtime') != 0 or not isinstance(manifest.get('files'), dict)):
        raise ValueError('Foreign or malformed transport manifest')
    limit = _integer(manifest.get('chunk_bytes'), 'chunk_bytes', maximum=MAX_CHUNK_BYTES)
    records = []
    for name, entry in manifest['files'].items():
        _relative(name)
        original = _owned(root, name)
        if not any(original.is_relative_to(folder) for folder in owned):
            raise ValueError('Original leaves caller-owned artifact roots: '+name)
        if original == manifest_path or original.is_relative_to(metadata/'artifact_transport'):
            raise ValueError('Original overlaps transport-owned files')
        digest = _digest(entry.get('sha256'))
        size = _integer(entry.get('size'), 'original size')
        compressed_digest = _digest(entry.get('compressed_sha256'))
        compressed_size = _integer(entry.get('compressed_size'), 'compressed size')
        chunks = entry.get('chunks')
        if entry.get('encoding') != 'gzip' or not isinstance(chunks, list) or not chunks:
            raise ValueError('Unsupported or missing transport encoding/chunks')
        paths, total, compressed_hash = [], 0, hashlib.sha256()
        for index, item in enumerate(chunks):
            expected = (metadata/'artifact_transport'/digest/f'{index:05d}.gz.part').relative_to(root).as_posix()
            if item.get('path') != expected:
                raise ValueError('Chunk path/order leaves its transport ownership')
            path = _owned(root, _relative(item['path']))
            count = _integer(item.get('size'), 'chunk size', maximum=limit)
            _identical(path, _digest(item.get('sha256')), count)
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024**2), b''):
                    compressed_hash.update(block)
            paths.append(path); total += count
        if total != compressed_size or compressed_hash.hexdigest() != compressed_digest:
            raise ValueError('Compressed transport SHA256/size differs')
        if original.exists():
            _identical(original, digest, size)
        records.append((name, original, digest, size, compressed_digest, compressed_size, paths))
    restored = []
    for name, original, digest, size, compressed_digest, compressed_size, paths in records:
        if original.exists():
            continue
        original.parent.mkdir(parents=True, exist_ok=True)
        _owned(root, name)
        temporary = original.parent/('.restore-'+uuid.uuid4().hex)
        observed, count = hashlib.sha256(), 0
        joined = _Joined(paths)
        try:
            with io.BufferedReader(joined) as raw, gzip.GzipFile(fileobj=raw, mode='rb') as compressed, temporary.open('xb') as target:
                for block in iter(lambda:compressed.read(1024**2), b''):
                    count += len(block)
                    if count > size:
                        raise ValueError('Restored file exceeds original size')
                    observed.update(block); target.write(block)
                target.flush(); os.fsync(target.fileno())
            if (count != size or observed.hexdigest() != digest or joined.size != compressed_size
                    or joined.digest.hexdigest() != compressed_digest):
                raise ValueError('Restored transport SHA256/size differs: '+name)
            _owned(root, name)
            if _install(temporary, original, digest, size):
                restored.append(name)
        finally:
            joined.close(); temporary.unlink(missing_ok=True)
    return restored
