#!/usr/bin/env python3
from __future__ import annotations

import html
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

SIDEBAR_RE = re.compile(
    r'<(?:div|nav)\s+class=["\']sidebar["\'][^>]*>.*?</(?:div|nav)>',
    re.IGNORECASE | re.DOTALL,
)
ANCHOR_RE = re.compile(r'<a\b[^>]*>', re.IGNORECASE | re.DOTALL)
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
ACTIVE_RE = re.compile(r'class=["\'][^"\']*\bactive\b[^"\']*["\']', re.IGNORECASE)
FRONT_MATTER_RE = re.compile(r'\A---\s*\n(.*?)\n---\s*\n', re.DOTALL)

SECTION_BY_SUFFIX = {
    'shortcv.html': 'cv',
    'publ.html': 'publications',
    'alect/index.html': 'invited-lectures',
    'nalect.html': 'non-academic-lectures',
    'videos.html': 'videos',
    'mg.html': 'metrology',
    'stgraph/index.html': 'stgraph',
    'echat/index.html': 'echat',
    'chatting/index.html': 'chatting',
    'blog/index.html': 'blog',
}

ROBOTO_LINES = [
    re.compile(r'^\s*<link\s+rel=["\']preconnect["\']\s+href=["\']https://fonts\.googleapis\.com["\'][^>]*>\s*\n?', re.I | re.M),
    re.compile(r'^\s*<link\s+rel=["\']preconnect["\']\s+href=["\']https://fonts\.gstatic\.com["\'][^>]*>\s*\n?', re.I | re.M),
    re.compile(r'^\s*<link\s+href=["\']https://fonts\.googleapis\.com/css2\?family=Roboto[^"\']*["\'][^>]*>\s*\n?', re.I | re.M),
]


def normalize_href(href: str) -> str:
    value = href.split('#', 1)[0].split('?', 1)[0].replace('\\', '/')
    while value.startswith('../'):
        value = value[3:]
    if value.startswith('./'):
        value = value[2:]
    return value.strip('/')


def infer_section(sidebar: str, path: Path) -> str:
    active_href = None
    for tag in ANCHOR_RE.findall(sidebar):
        if ACTIVE_RE.search(tag):
            m = HREF_RE.search(tag)
            if m:
                active_href = normalize_href(m.group(1))
                break
    if active_href:
        for suffix, section in SECTION_BY_SUFFIX.items():
            if active_href.endswith(suffix):
                return section
        if active_href == 'index.html' and path == ROOT / 'index.html':
            return 'home'
    if path == ROOT / 'index.html':
        return 'home'
    return ''


def page_root(path: Path) -> str:
    depth = len(path.relative_to(ROOT).parts) - 1
    return '../' * depth


def set_front_matter(text: str, section: str, root: str) -> str:
    m = FRONT_MATTER_RE.match(text)
    if m:
        lines = [
            line for line in m.group(1).splitlines()
            if not re.match(r'^\s*(section|root)\s*:', line)
        ]
        additions = []
        if section:
            additions.append(f'section: {section}')
        additions.append(f'root: "{root}"')
        body = text[m.end():]
        fm = '\n'.join(additions + lines).strip()
        return f'---\n{fm}\n---\n{body}'
    additions = []
    if section:
        additions.append(f'section: {section}')
    additions.append(f'root: "{root}"')
    return '---\n' + '\n'.join(additions) + '\n---\n' + text


def remove_roboto(text: str) -> str:
    for pattern in ROBOTO_LINES:
        text = pattern.sub('', text)
    return text


def harden_blank_targets(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tag = match.group(0)
        if re.search(r'\brel\s*=', tag, re.I):
            return tag
        return tag[:-1] + ' rel="noopener noreferrer">'

    return re.sub(r'<a\b[^>]*\btarget=["\']_blank["\'][^>]*>', repl, text, flags=re.I)


def migrate_sidebars() -> tuple[int, list[str]]:
    changed = 0
    sections = []
    for path in ROOT.rglob('*.html'):
        rel = path.relative_to(ROOT)
        if any(part in {'.git', '_site', 'vendor'} for part in rel.parts):
            continue
        text = path.read_text(encoding='utf-8')
        m = SIDEBAR_RE.search(text)
        if not m:
            cleaned = harden_blank_targets(remove_roboto(text))
            if cleaned != text:
                path.write_text(cleaned, encoding='utf-8')
            continue

        section = infer_section(m.group(0), path)
        new = text[:m.start()] + '{% include sidebar.html %}' + text[m.end():]
        new = remove_roboto(new)
        new = harden_blank_targets(new)
        if rel.as_posix() != 'chatting/_content.html':
            new = set_front_matter(new, section, page_root(path))
        if new != text:
            path.write_text(new, encoding='utf-8')
            changed += 1
            sections.append(f'{rel.as_posix()}: {section or "(no active main section)"}')
    return changed, sections


def update_sidebar_include() -> None:
    path = ROOT / '_includes/sidebar.html'
    text = path.read_text(encoding='utf-8')
    text = re.sub(
        r'(<a href="\{\{ page\.root \| default: \'\' \}\}blog/index\.html")\s+target="_blank"\s+rel="noopener noreferrer"',
        r'\1',
        text,
    )
    path.write_text(text, encoding='utf-8')


def clean_global_css() -> None:
    path = ROOT / 'mystyles.css'
    css = path.read_text(encoding='utf-8')
    css = re.sub(
        r'\n\.content:not\(:has\(\.content0:target\)\).*?\n/\* Standalone conversations and stories: reading mode \*/',
        '\n/* Standalone conversations and stories: reading mode */',
        css,
        flags=re.DOTALL,
    )
    accessibility = '''\n\n/* Pre-production accessibility and resilience */\nimg, video {\n    max-width: 100%;\n    height: auto;\n}\n\niframe {\n    max-width: 100%;\n}\n\npre {\n    max-width: 100%;\n    overflow-x: auto;\n}\n\na:focus-visible,\n.sidebar a:focus-visible,\n.nav a:focus-visible {\n    outline: 2px solid var(--color-accent) !important;\n    outline-offset: 3px;\n}\n\n@media (prefers-reduced-motion: reduce) {\n    html {\n        scroll-behavior: auto;\n    }\n}\n'''
    if '/* Pre-production accessibility and resilience */' not in css:
        css += accessibility
    path.write_text(css, encoding='utf-8')


def clean_blog_css() -> None:
    path = ROOT / 'blog/style.css'
    css = path.read_text(encoding='utf-8')
    extra = '''\n\n/* Keyboard and reduced-motion support */\na:focus-visible,\nbutton:focus-visible {\n    outline: 2px solid var(--color-accent);\n    outline-offset: 3px;\n}\n\n@media (prefers-reduced-motion: reduce) {\n    html {\n        scroll-behavior: auto;\n    }\n}\n'''
    if '/* Keyboard and reduced-motion support */' not in css:
        css += extra
    path.write_text(css, encoding='utf-8')


def delete_legacy_backup() -> None:
    path = ROOT / 'chatting/index.bak.html'
    if path.exists():
        path.unlink()


def local_target_exists(source: Path, raw_ref: str) -> bool:
    ref = html.unescape(raw_ref).strip()
    if not ref or ref.startswith(('#', 'mailto:', 'tel:', 'javascript:', 'data:', '//')):
        return True
    if '{{' in ref or '{%' in ref:
        return True
    parts = urlsplit(ref)
    if parts.scheme in {'http', 'https'}:
        return True
    path_part = unquote(parts.path)
    if not path_part:
        return True
    if path_part.startswith('/'):
        target = ROOT / path_part.lstrip('/')
    else:
        target = source.parent / path_part
    target = Path(os.path.normpath(target))
    if target.exists():
        return True
    if target.is_dir() and (target / 'index.html').exists():
        return True
    if not target.suffix and (target / 'index.html').exists():
        return True
    return False


def audit() -> None:
    sidebar_left = []
    roboto_left = []
    broken = []
    images_without_alt = []

    attr_re = re.compile(r'\b(?:href|src)=["\']([^"\']+)["\']', re.I)
    img_re = re.compile(r'<img\b[^>]*>', re.I | re.DOTALL)

    for path in ROOT.rglob('*.html'):
        rel = path.relative_to(ROOT)
        if any(part in {'.git', '_site', 'vendor'} for part in rel.parts):
            continue
        text = path.read_text(encoding='utf-8')
        if rel.as_posix() != '_includes/sidebar.html' and SIDEBAR_RE.search(text):
            sidebar_left.append(rel.as_posix())
        if 'family=Roboto' in text:
            roboto_left.append(rel.as_posix())
        for raw in attr_re.findall(text):
            if not local_target_exists(path, raw):
                broken.append(f'{rel.as_posix()} -> {raw}')
        for tag in img_re.findall(text):
            if not re.search(r'\balt\s*=', tag, re.I):
                images_without_alt.append(f'{rel.as_posix()}: {tag[:100]}')

    print(f'OLD_SIDEBARS_REMAINING={len(sidebar_left)}')
    for item in sidebar_left:
        print('  sidebar:', item)
    print(f'ROBOTO_REFERENCES_REMAINING={len(roboto_left)}')
    for item in roboto_left:
        print('  roboto:', item)
    print(f'BROKEN_LOCAL_REFERENCES={len(broken)}')
    for item in broken[:100]:
        print('  broken:', item)
    print(f'IMAGES_WITHOUT_ALT={len(images_without_alt)}')
    for item in images_without_alt[:100]:
        print('  alt:', item)

    if sidebar_left or roboto_left:
        raise SystemExit('Structural cleanup incomplete.')


def remove_one_shot_files() -> None:
    workflow = ROOT / '.github/workflows/preprod-cleanup.yml'
    if workflow.exists():
        workflow.unlink()
    script = Path(__file__).resolve()
    if script.exists():
        script.unlink()
    tools = ROOT / 'tools'
    try:
        tools.rmdir()
    except OSError:
        pass
    workflows = ROOT / '.github/workflows'
    try:
        workflows.rmdir()
    except OSError:
        pass
    github = ROOT / '.github'
    try:
        github.rmdir()
    except OSError:
        pass


def main() -> None:
    changed, sections = migrate_sidebars()
    update_sidebar_include()
    clean_global_css()
    clean_blog_css()
    delete_legacy_backup()
    print(f'MIGRATED_SIDEBARS={changed}')
    for item in sections:
        print('  migrated:', item)
    audit()
    remove_one_shot_files()


if __name__ == '__main__':
    main()
