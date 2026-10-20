#!/usr/bin/env python
"""Freeze private unified-training roles and validation target qualification.

Outputs are immutable. Replay verifies every committed file without decoding labels.
"""
import argparse
import json
import sys

from anatobind.aur.data_contract import private_output, qualify, verify_contract, finalize_qualification


class AnonymousParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2,'Data qualification argument error\n')


def main(argv=None):
    parser=AnonymousParser(description=__doc__)
    parser.add_argument('--out',required=True,help='New private directory under the home root, outside Git')
    parser.add_argument('--replay-only',action='store_true',help='Verify all frozen input/output/code hashes')
    parser.add_argument('--finalize-from',help='Immutable direct-parent execution for metadata-only qualification finalization')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=parser.parse_args(argv)
    try:
        out=private_output(args.out)
        if args.replay_only and args.finalize_from: raise ValueError('Conflicting qualification modes')
        if args.finalize_from:
            print(json.dumps(finalize_qualification(out,args.finalize_from),sort_keys=True))
        elif args.replay_only:
            contract=verify_contract(out)
            print(json.dumps({'version':contract['version'],'integrity_verified':True,'summary':contract['summary']},sort_keys=True))
        else:
            print(json.dumps(qualify(out,workers=args.workers),sort_keys=True))
    except Exception:
        print('Data qualification failed: private contract or integrity check rejected',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
