#!/usr/bin/env python
"""Run the private, frozen dual-mode G1 engineering gate (NOT_EVIDENCE)."""
import argparse
import json
import sys

from anatobind.aur.ssl.unified_eval import run


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, 'Invalid dual G1 arguments\n')


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--crop-2d', type=int, nargs=3, default=[1, 160, 160])
    parser.add_argument('--crop-3d', type=int, nargs=3, default=[128, 160, 160])
    parser.add_argument('--calibration', type=int, default=96)
    parser.add_argument('--validation', type=int, default=48)
    parser.add_argument('--lesion', type=int, default=48)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--readout-epochs', type=int, default=300)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        report = run(args.contract, args.checkpoint, args.out, device=args.device,
            crops={'2d': args.crop_2d, '3d': args.crop_3d}, n_calibration=args.calibration,
            n_validation=args.validation, n_lesion=args.lesion, seeds=tuple(args.seeds),
            readout_epochs=args.readout_epochs, n_bootstrap=args.bootstrap)
        print(json.dumps({'version': report['version'], 'default_recipe_match': report['default_recipe_match'],
                          'modes': {mode: item['status'] for mode, item in report['modes'].items()}}))
        return 0
    except Exception:
        print('Dual G1 evaluation failed; private evidence preserved', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
