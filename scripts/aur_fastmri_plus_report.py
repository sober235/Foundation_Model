"""Report fastMRI+ diagnostic box coverage with the official fixed convention."""
import argparse
import json
from anatobind.aur.fastmri_plus_report import write_report


class PrivateParser(argparse.ArgumentParser):
    def error(self,message):self.exit(2,'invalid fastMRI report arguments\n')


def main(argv=None):
    parser=PrivateParser(description=__doc__)
    for key in ('batch','annotations','dicom-root','out'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--audit-limit',type=int,default=20)
    args=parser.parse_args(argv)
    try:
        report=write_report(args.batch,args.annotations,args.dicom_root,args.out,audit_limit=args.audit_limit)
        print(json.dumps(report,sort_keys=True));return 0
    except Exception:
        print(json.dumps({'status':'FAILED','reason':'fastmri_report_failed'}));return 1


if __name__=='__main__':raise SystemExit(main())
