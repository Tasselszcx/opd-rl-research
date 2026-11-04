"""Check maintained navigation, provenance, and accidental private/run artifacts."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'opd-rl-research'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ['href', 'src'] and value:
                self.links.append(value)


def local_target(source, href):
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    prefix = '/' + SLUG + '/'
    if path.startswith(prefix):
        return ROOT / path[len(prefix):]
    if path.startswith('/'):
        return ROOT / path.lstrip('/')
    return source.parent / path


def main():
    errors = []
    maintained = [ROOT / p for p in ['README.md', 'THIRD_PARTY_NOTICES.md', 'archive/README.md',
                  'reading-notes/README.md', 'experiments/student-state-guidance/README.md']]
    maintained += sorted((ROOT / 'docs').glob('*.md')) + sorted((ROOT / 'results/g3a').glob('*.md'))
    link_count = 0
    for source in maintained:
        text = re.sub(r'```.*?```', '', source.read_text(), flags=re.S)
        for href in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            href = href.strip().strip('<>')
            target = local_target(source, href)
            if target is not None:
                link_count += 1
                if not target.exists():
                    errors.append(str(source.relative_to(ROOT)) + ': missing ' + href)
    for source in [ROOT / 'index.html', ROOT / '404.html', *sorted((ROOT / 'reading-notes').glob('*.html'))]:
        parser = Links()
        parser.feed(source.read_text())
        for href in parser.links:
            target = local_target(source, href)
            if target is not None:
                link_count += 1
                if not target.exists():
                    errors.append(str(source.relative_to(ROOT)) + ': missing ' + href)

    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    files = {ROOT / p.decode() for p in paths.split(b'\0') if p}
    secrets = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    private = re.compile(r'/(?:mnt/dolphinfs|Users/tassels|home/hadoop-[^/]+)/|\b10[.]245[.]\d+[.]\d+')
    for path in sorted(files):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        if path.stat().st_size >= 95 * 1024 * 1024:
            errors.append(rel + ': exceeds publication size limit')
        upstream_fixture = rel == 'experiments/student-state-guidance/verl/tests/e2e/arithmetic_sequence/model/model.safetensors'
        # Keep the small, public upstream toy fixture; its exact hash is checked below.
        if upstream_fixture and path.stat().st_size > 2 * 1024 * 1024:
            errors.append(rel + ': upstream fixture unexpectedly enlarged')
        if path.name in ['.env', 'RUN_TARGET.json', 'RUN_HANDOFF.md'] or (path.suffix in ['.safetensors', '.pt', '.pth'] and not upstream_fixture):
            errors.append(rel + ': run/credential/weight artifact is not publishable')
        try:
            text = path.read_text()
        except UnicodeError:
            continue
        if secrets.search(text):
            errors.append(rel + ': credential-like content; inspect locally')
        if private.search(text):
            errors.append(rel + ': machine-specific deployment path/address')
    source = ROOT / 'experiments/student-state-guidance'
    manifest = json.loads((source / 'configs/source-snapshot-files.json').read_text())
    provenance = json.loads((source / 'PUBLICATION_PROVENANCE.json').read_text())
    modified = {x['path']: x['public_sha256'] for x in provenance['modified_snapshot_files']}
    for item in manifest['files']:
        path = source / item['public_path']
        expected = modified.get(item['public_path'], item['sha256'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('source snapshot missing or unrecorded change: ' + item['public_path'])
    python_count = 0
    for folder in [ROOT / 'tools', source / 'state_guidance', source / 'guidance', source / 'length_audit', source / 'reproduction', source / 'tools']:
        for path in folder.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            python_count += 1
    if errors:
        print(json.dumps({'status': 'failed', 'errors': errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({'status': 'passed', 'local_links_checked': link_count,
                      'snapshot_files_checked': len(manifest['files']), 'python_files_parsed': python_count}))


if __name__ == '__main__':
    main()
