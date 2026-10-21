#!/usr/bin/env python
"""Run frozen, mode-explicit development A/U/R evaluation (NOT_EVIDENCE)."""
import argparse
import json
import sys

from anatobind.aur.unified_evaluation import run_evaluation


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, 'Invalid development evaluation arguments\n')


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument('--contract',required=True)
    parser.add_argument('--export',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--modes',nargs='+',choices=('2d','3d'),default=['2d','3d'])
    parser.add_argument('--score-threshold',type=float,default=.5)
    parser.add_argument('--qualified-source',help='Exact qualified source archive, verified against original hashes')
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(args.contract,args.export,args.out,device=args.device,modes=tuple(args.modes),
                                score_threshold=args.score_threshold,qualified_source_root=args.qualified_source)
        print(json.dumps({key:result[key] for key in ('version','evidence','quality_certified','n_observations','n_patients','n_sources')}))
        return 0
    except Exception:
        print('Development evaluation failed; private evidence preserved',file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
