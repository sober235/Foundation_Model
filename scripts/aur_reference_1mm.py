#!/usr/bin/env python
"""Rescore frozen S7 references on the qualified AUR 1 mm grid (NOT_EVIDENCE)."""
import argparse
import json
import sys

from anatobind.aur.reference_1mm import run_reference


class _Parser(argparse.ArgumentParser):
    def error(self,message):self.exit(2,'Invalid reference arguments\n')


def main(argv=None):
    p=_Parser(description=__doc__)
    for name in ('contract','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--qualified-source');p.add_argument('--test-role')
    p.add_argument('--sources',nargs='+',choices=('pdgm','bmsr','isles'),default=['pdgm','bmsr','isles'])
    args=p.parse_args(argv)
    try:
        result=run_reference(args.contract,args.out,qualified_source_root=args.qualified_source,
                             test_role=args.test_role,sources=tuple(args.sources))
        for summary in result['sources'].values():
            print(json.dumps({key:summary[key] for key in ('source','n_scans','excluded_cases','sensitivity','fp_per_scan')},allow_nan=False))
        return 0
    except Exception:
        print('Reference qualification failed; private evidence preserved',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
