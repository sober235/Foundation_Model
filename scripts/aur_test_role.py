#!/usr/bin/env python
"""Explicit final-gate historical-test qualification; never changes training roles."""
import argparse
import json
import sys
from pathlib import Path

from anatobind.aur.data_contract import file_sha256

from anatobind.aur.test_role import build_historical_test_role, read_test_role


class _Parser(argparse.ArgumentParser):
    def error(self,message):self.exit(2,'Invalid historical-test qualification arguments\n')


def main(argv=None):
    p=_Parser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--qualified-source');p.add_argument('--replay-only',action='store_true')
    args=p.parse_args(argv)
    try:
        if args.replay_only:
            result=read_test_role(args.out,qualified_source_root=args.qualified_source)
            parent=Path(args.parent).expanduser().resolve()
            parent=parent.parent if parent.is_file() else parent
            if file_sha256(parent/'contract.json')!=result['parent_contract_sha256']:
                raise ValueError('Historical replay parent differs')
            print(json.dumps({'integrity_verified':True,'rows':len(result['rows']),'contract_sha256':result['contract_sha256']}))
        else:
            print(json.dumps(build_historical_test_role(args.parent,args.out,qualified_source_root=args.qualified_source),allow_nan=False))
        return 0
    except Exception:
        print('Historical-test qualification failed; private evidence preserved',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
