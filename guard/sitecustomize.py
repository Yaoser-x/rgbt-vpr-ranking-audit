"""Deny network and reads outside the shipped package, new result tree, and Python runtime."""
import atexit
import json
import os
from pathlib import Path
import site
import sys

if os.environ.get('PAPER_A_READ_ROOTS'):
    roots = [Path(p).resolve() for p in json.loads(os.environ['PAPER_A_READ_ROOTS'])]
    roots += [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    roots += [Path(p).resolve() for p in site.getsitepackages()]
    seen = set()
    denied = []
    logbase = Path(os.environ['PAPER_A_ACCESS_LOG'])
    logfile = logbase.with_name(logbase.stem+'-'+str(os.getpid())+'.json')

    def guard(event, args):
        if event in ('socket.connect', 'socket.bind', 'socket.getaddrinfo'):
            raise RuntimeError('Offline reconstruction forbids network: '+event)
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if str(path).casefold().replace('\\','/').rstrip('/') in ('nul','//./nul'):
            return
        if not any(path.is_relative_to(root) for root in roots):
            denied.append(str(path))
            raise RuntimeError('Reconstruction attempted access outside allowed roots: '+str(path))
        seen.add(str(path))

    def finish():
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.write_text(json.dumps(dict(network='DENIED', outside_reads_denied=denied,
            allowed_roots=[str(p) for p in roots], accessed_files=sorted(seen)), indent=2), encoding='utf-8')

    sys.addaudithook(guard)
    atexit.register(finish)
