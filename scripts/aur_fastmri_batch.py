"""Explicit assumed-pixel-grid inference; only anonymous summaries reach stdout."""
import argparse
import json
from pathlib import Path
from anatobind.aur.fastmri_batch import run_batch


class PrivateParser(argparse.ArgumentParser):
    def error(self,message):self.exit(2,'invalid fastMRI batch arguments\n')


def main(argv=None):
    parser=PrivateParser(description=__doc__)
    for name in ('root','export','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--modes',nargs='+',choices=('2d','3d'),default=['2d','3d'])
    parser.add_argument('--score-threshold',type=float)
    parser.add_argument('--gate-registration')
    parser.add_argument('--expected-registration-sha256')
    parser.add_argument('--device',default='cpu');parser.add_argument('--limit',type=int)
    parser.add_argument('--crops-2d',nargs=3,type=int,default=[1,160,160])
    parser.add_argument('--crops-3d',nargs=3,type=int,default=[128,160,160])
    args=parser.parse_args(argv)
    try:
        registration=json.loads(Path(args.gate_registration).read_text()) if args.gate_registration else None
        summary=run_batch(args.root,args.export,args.out,modes=args.modes,
            crops={'2d':args.crops_2d,'3d':args.crops_3d},score_threshold=args.score_threshold,device=args.device,limit=args.limit,
            gate_registration=registration,expected_registration_sha256=args.expected_registration_sha256)
        print(json.dumps(summary,sort_keys=True));return 0
    except Exception:
        print(json.dumps({'status':'FAILED','reason':'fastmri_batch_failed'}));return 1


if __name__=='__main__':raise SystemExit(main())
