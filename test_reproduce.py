import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reproduce import REPRESENTATIONS, compare_value, new_output, sha, validate


class ReconstructionContract(unittest.TestCase):
    def test_tampered_input_is_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); member=root/'frozen.bin';member.write_bytes(b'frozen')
            record={'schema':1,'representations':REPRESENTATIONS,'files':{'frozen.bin':{'sha256':sha(member),'bytes':6}}}
            (root/'INPUT_MANIFEST.json').write_text(json.dumps(record))
            member.write_bytes(b'edited')
            with self.assertRaisesRegex(RuntimeError,'SHA mismatch'):validate(root)

    def test_missing_input_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'INPUT_MANIFEST.json').write_text(json.dumps({'schema':1,'representations':REPRESENTATIONS,
                'files':{'missing.npz':{'sha256':'0'*64,'bytes':1}}}))
            with self.assertRaisesRegex(RuntimeError,'Missing input'):validate(root)

    def test_wrong_representation_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'INPUT_MANIFEST.json').write_text(json.dumps({'schema':1,'files':{},
                'representations':{'SALAD':['dr_c','dt_c'],'Clique':['dr_s','dt_s']}}))
            with self.assertRaisesRegex(RuntimeError,'representation mapping'):validate(root)

    def test_member_cannot_escape_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'INPUT_MANIFEST.json').write_text(json.dumps({'schema':1,'representations':REPRESENTATIONS,
                'files':{'../outside':{'sha256':'0'*64,'bytes':1}}}))
            with self.assertRaisesRegex(RuntimeError,'escapes package'):validate(root)

    def test_existing_and_sealed_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'input';source.mkdir(); old=root/'old';old.mkdir()
            (old/'untouched').write_text('retain')
            with self.assertRaisesRegex(RuntimeError,'already exist'):new_output(source,old)
            (root/'MANIFEST.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'sealed evidence'):new_output(source,root/'new')
            self.assertEqual((old/'untouched').read_text(),'retain')

    def test_integer_and_float_comparison_contract(self):
        with self.assertRaisesRegex(RuntimeError,'Integer mismatch'):compare_value('532','531','count')
        with self.assertRaisesRegex(RuntimeError,'Float mismatch'):compare_value(1.0,1.0+1e-9,'interval')
        compare_value(1.0,1.0+1e-12,'interval')
        with self.assertRaisesRegex(RuntimeError,'Float mismatch'):compare_value(float('nan'),0.,'interval')

    def test_guard_allows_null_device_and_process_pool_but_denies_external_reads_and_network(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve(); env=os.environ.copy()
            env.update(PYTHONPATH=str(Path(__file__).resolve().parent/'guard'),PYTHONDONTWRITEBYTECODE='1',
                       PAPER_A_READ_ROOTS=json.dumps([str(root),str(Path(__file__).resolve().parent)]),
                       PAPER_A_ACCESS_LOG=str(root/'access.json'))
            script='''
import os, socket, sys
from concurrent.futures import ProcessPoolExecutor
if __name__ == '__main__':
    with open(os.devnull,'r') as stream: stream.read()
    with ProcessPoolExecutor(max_workers=1) as pool: assert list(pool.map(abs,[-1,-2])) == [1,2]
    for operation in (lambda:open(sys.argv[1],'r'),
                      lambda:socket.socket().connect(('127.0.0.1',9))):
        try: operation()
        except RuntimeError: pass
        else: raise AssertionError('Guard failed to reject operation')
'''
            result=subprocess.run([sys.executable,'-B','-c',script,str(root.parent/'guard-denied-fixture')],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
