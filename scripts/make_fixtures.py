"""Generate the TRXScan fixtures named in tests/fixtures/trxscan_fixtures.txt and derive unpile's inputs.

    python scripts/make_fixtures.py [name ...]      (default: phasediff rpe; dest: $UNPILE_FIXTURES or testdata/fixtures)

Needs `pip install "trxscan==<TRXSCAN_VERSION>" nibabel nilearn scikit-image scipy`; the phantom is fetched by TRXScan.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, '..', 'tests', 'fixtures', 'trxscan_fixtures.txt')


def recipes():
    out = {}
    for line in open(SPEC):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('TRXSCAN_VERSION'):
            continue
        name, rest = line.split(':', 1)
        recipe, *sets = rest.split()
        out[name.strip()] = (recipe, sets)
    return out


def main(names):
    dest = os.environ.get('UNPILE_FIXTURES', os.path.join('testdata', 'fixtures'))
    os.makedirs(dest, exist_ok=True)
    known = recipes()
    for n in names or ['phasediff', 'rpe']:
        recipe, sets = known[n]
        target = os.path.join(dest, n)
        if os.path.exists(os.path.join(target, 'unpile', 'meta.json')):
            print(f'{n}: cached in {target}')
            continue
        shutil.rmtree(target, ignore_errors=True)
        cmd = ['trxscan-fixture', recipe, '--out', target] + sets
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
        subprocess.run([sys.executable, os.path.join(HERE, 'prep_fixture.py'), target], check=True)


if __name__ == '__main__':
    main(sys.argv[1:])
