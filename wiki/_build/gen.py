#!/usr/bin/env python3
"""MindIE-LLM Wiki 页面生成器（支持顶层与子目录页面）

读取 _build/pages/<path>.html 中的内容片段（YAML 头 + article 内部），
用统一模板包裹后输出到 wiki/<path>.html。

YAML 头格式（前 3 行）：
---
title: 页面标题
prev: <slug 或 '' >        # 可省，自动按顺序推断
next: <slug 或 '' >
group: <显示在面包屑的分组>  # 可省
---

支持子目录：例如 features/multi-lora.html → 输出 wiki/features/multi-lora.html，
对应内容片段 _build/pages/features/multi-lora.html。子目录页面会自动用 '../' 前缀
修正所有静态资源与侧栏链接。
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / '_build' / 'pages'

# 顶层页面顺序（用于 prev/next 推断与面包屑）
ORDER = [
    ('index',          '首页',                 None),
    ('introduction',   '项目简介',             '入门'),
    ('quick-start',    '快速入门',             '入门'),
    ('installation',   '安装与编译',           '入门'),
    ('architecture',   '架构总览',             '架构与内核'),
    ('directory',      '目录结构',             '架构与内核'),
    ('server-api',     '服务端与 API',         '架构与内核'),
    ('cpp-engine',     'C++ 引擎内核',         '架构与内核'),
    ('scheduler',      '调度器',               '架构与内核'),
    ('block-manager',  'KV Cache 与 Block',    '架构与内核'),
    ('python-runtime', 'Python 推理框架',      '架构与内核'),
    ('features',       '特性总览',             '特性'),
    ('quantization',   '量化特性',             '特性'),
    ('parallelism',    '并行策略',             '特性'),
    ('acceleration',   '加速特性',             '特性'),
    ('development',    '开发与贡献',           '开发'),
    ('faq',            '常见问题',             '参考'),
    ('references',     '参考资料',             '参考'),
]
SLUG_TITLE = {s: t for s, t, _ in ORDER}
SLUG_GROUP = {s: g for s, _, g in ORDER}
SLUGS = [s for s, _, _ in ORDER]

# 特性详解页（subdir = features/）
FEATURE_GROUPS = [
    ('基础并行', [
        ('multi-lora',        'Multi-LoRA'),
        ('moe',               'MoE'),
        ('mla',               'MLA'),
        ('expert-load-balance', 'EPLB 负载均衡'),
        ('shared-expert',     '共享专家外置'),
        ('expert-parallel',   'Expert Parallel'),
        ('data-parallel',     'Data Parallel'),
        ('tensor-parallel',   'Tensor Parallel'),
    ]),
    ('量化', [
        ('anti-outlier',      '离群值抑制'),
        ('pdmix-quant',       'PD MIX 量化'),
        ('w8a8',              'W8A8'),
        ('w4a8',              'W4A8 混合'),
        ('w8a16',             'W8A16'),
        ('attention-quant',   'Attention 量化'),
        ('fa3-quant',         'FA3 量化'),
        ('kv-int8',           'KV Cache Int8'),
        ('w8a8sc',            'W8A8SC 稀疏'),
        ('w16a16sc',          'W16A16SC 稀疏'),
    ]),
    ('长序列', [
        ('context-parallel',  'Context Parallel'),
        ('sequence-parallel', 'Sequence Parallel'),
    ]),
    ('调度', [
        ('async-scheduling',  '异步调度'),
        ('splitfuse',         'SplitFuse'),
        ('slo-scheduling',    'SLO 调度优化'),
    ]),
    ('加速', [
        ('micro-batch',       'Micro Batch'),
        ('speculative-decoding', '投机解码 / Lookahead'),
        ('mtp',               'MTP'),
        ('prefix-cache',      'Prefix Cache'),
        ('kv-cache-pool',     'KV Cache 池化'),
    ]),
    ('交互与其他', [
        ('function-call',     'Function Call'),
        ('reasoning',         '思考解析与预算'),
        ('structured-output', '结构化输出'),
        ('offline-weight',    '权重离线切分'),
    ]),
]
FEATURE_SLUG_TITLE: dict[str, str] = {}
FEATURE_LINEAR: list[str] = []
for _g, items in FEATURE_GROUPS:
    for s, t in items:
        FEATURE_SLUG_TITLE[s] = t
        FEATURE_LINEAR.append(s)


def render_top_sidebar(prefix: str = '') -> str:
    """生成统一侧栏（top + 特性详解二级组）。prefix 用于子目录页面（'../'）"""
    p = prefix
    sub_html = ''
    for gi, (gname, items) in enumerate(FEATURE_GROUPS):
        list_id = f'feat-sub-{gi}'
        items_html = '\n'.join(
            f'          <li><a href="{p}features/{slug}.html">{title}</a></li>'
            for slug, title in items
        )
        sub_html += (
            f'        <li>\n'
            f'          <button type="button" class="nav-sub-toggle" '
            f'aria-controls="{list_id}" aria-expanded="false">'
            f'<span>{gname}</span><span class="caret">▸</span></button>\n'
            f'          <ul class="nav-sub-list" id="{list_id}">\n'
            f'{items_html}\n'
            f'          </ul>\n'
            f'        </li>\n'
        )

    return f"""<input class="sidebar-filter" placeholder="筛选目录…" aria-label="筛选目录">
    <nav>
      <div class="nav-group">
        <div class="nav-group-title">入门</div>
        <ul>
          <li><a href="{p}index.html">首页</a></li>
          <li><a href="{p}introduction.html">项目简介</a></li>
          <li><a href="{p}quick-start.html">快速入门</a></li>
          <li><a href="{p}installation.html">安装与编译</a></li>
        </ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">架构与内核</div>
        <ul>
          <li><a href="{p}architecture.html">架构总览</a></li>
          <li><a href="{p}directory.html">目录结构</a></li>
          <li><a href="{p}server-api.html">服务端与 API</a></li>
          <li><a href="{p}cpp-engine.html">C++ 引擎内核</a></li>
          <li><a href="{p}scheduler.html">调度器</a></li>
          <li><a href="{p}block-manager.html">KV Cache &amp; Block</a></li>
          <li><a href="{p}python-runtime.html">Python 推理框架</a></li>
        </ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">特性</div>
        <ul>
          <li><a href="{p}features.html">特性总览（索引）</a></li>
          <li><a href="{p}quantization.html">量化特性</a></li>
          <li><a href="{p}parallelism.html">并行策略</a></li>
          <li><a href="{p}acceleration.html">加速特性</a></li>
{sub_html}        </ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">开发</div>
        <ul>
          <li><a href="{p}development.html">开发与贡献</a></li>
        </ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">参考</div>
        <ul>
          <li><a href="{p}faq.html">常见问题</a></li>
          <li><a href="{p}references.html">参考资料</a></li>
        </ul>
      </div>
    </nav>"""


def render_header(prefix: str = '') -> str:
    p = prefix
    return f"""<header class="site-header">
  <a class="brand" href="{p}index.html">
    <img src="{p}assets/img/ascend-logo.png" alt="Ascend">
    <span>MindIE-<span class="accent">LLM</span> Wiki</span>
  </a>
  <div class="header-spacer"></div>
  <nav class="header-links">
    <a href="https://github.com/Lyr-GW/MindIE-LLM" target="_blank" rel="noopener">GitHub</a>
    <a href="https://gitcode.com/Ascend/MindIE-LLM" target="_blank" rel="noopener">GitCode</a>
    <a href="https://www.hiascend.com/" target="_blank" rel="noopener">昇腾社区</a>
  </nav>
  <button class="theme-toggle" type="button">☾ 暗色</button>
  <button class="menu-toggle" type="button" aria-label="菜单">☰</button>
</header>"""


TPL = """<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · MindIE-LLM Wiki</title>
<meta name="description" content="{title} — MindIE-LLM Wiki">
<link rel="icon" href="{prefix}assets/img/favicon.ico">
<link rel="stylesheet" href="{prefix}assets/css/wiki.css">
</head>
<body>
{header}

<div class="layout">
  <aside class="sidebar">
    {sidebar}
  </aside>

  <main class="content">
    {breadcrumb}
    <article>
{body}
    </article>

    <footer class="page-footer">
      {prev_html}
      {next_html}
      <div class="copyright">
        © 2025–2026 MindIE-LLM Wiki · 内容基于源码归纳，与官方文档可能存在表述差异
      </div>
    </footer>
  </main>

  <aside class="toc"></aside>
</div>

<script src="{prefix}assets/js/wiki.js"></script>
</body>
</html>
"""


def parse_page(text: str):
    m = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', text, re.DOTALL)
    if not m:
        raise ValueError('页面缺少 YAML 头')
    head = {}
    for line in m.group(1).splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            head[k.strip()] = v.strip()
    body = m.group(2)
    return head, body


def neighbor_link(slug: str, prefix: str, in_features: bool) -> tuple[str, str]:
    """返回 (prev_html, next_html)。slug 形如 'features/moe' 或 'scheduler'。"""
    if in_features:
        # 特性详解页：在 FEATURE_LINEAR 中找邻居
        bare = slug.split('/', 1)[1]
        if bare not in FEATURE_LINEAR:
            return ('<span></span>', '<span></span>')
        i = FEATURE_LINEAR.index(bare)
        prev_html = '<span></span>'
        next_html = '<span></span>'
        if i > 0:
            ps = FEATURE_LINEAR[i - 1]
            prev_html = f'<a class="pager-prev" href="{ps}.html">{FEATURE_SLUG_TITLE[ps]}</a>'
        else:
            prev_html = f'<a class="pager-prev" href="{prefix}features.html">特性总览</a>'
        if i < len(FEATURE_LINEAR) - 1:
            ns = FEATURE_LINEAR[i + 1]
            next_html = f'<a class="pager-next" href="{ns}.html">{FEATURE_SLUG_TITLE[ns]}</a>'
        return prev_html, next_html
    # 顶层页面
    if slug not in SLUGS:
        return ('<span></span>', '<span></span>')
    idx = SLUGS.index(slug)
    prev_slug = SLUGS[idx - 1] if idx > 0 else None
    next_slug = SLUGS[idx + 1] if idx < len(SLUGS) - 1 else None
    prev_html = (
        f'<a class="pager-prev" href="{prev_slug}.html">{SLUG_TITLE[prev_slug]}</a>'
        if prev_slug else '<span></span>'
    )
    next_html = (
        f'<a class="pager-next" href="{next_slug}.html">{SLUG_TITLE[next_slug]}</a>'
        if next_slug else '<span></span>'
    )
    return prev_html, next_html


def render(slug: str):
    page_path = PAGES_DIR / f'{slug}.html'
    head, body = parse_page(page_path.read_text(encoding='utf-8'))

    in_features = slug.startswith('features/')
    prefix = '../' if in_features else ''

    # title
    if in_features:
        bare = slug.split('/', 1)[1]
        title = head.get('title') or FEATURE_SLUG_TITLE.get(bare, bare)
    else:
        title = head.get('title') or SLUG_TITLE.get(slug, slug)

    # neighbors
    prev_html_default, next_html_default = neighbor_link(slug, prefix, in_features)
    # YAML 显式指定时覆盖
    if 'prev' in head:
        v = head['prev'].strip()
        if not v:
            prev_html_default = '<span></span>'
        else:
            target_title = SLUG_TITLE.get(v) or FEATURE_SLUG_TITLE.get(v) or v
            href = (f'../{v}.html' if (in_features and '/' not in v
                                        and v not in FEATURE_LINEAR) else f'{v}.html')
            prev_html_default = f'<a class="pager-prev" href="{href}">{target_title}</a>'
    if 'next' in head:
        v = head['next'].strip()
        if not v:
            next_html_default = '<span></span>'
        else:
            target_title = SLUG_TITLE.get(v) or FEATURE_SLUG_TITLE.get(v) or v
            href = (f'../{v}.html' if (in_features and '/' not in v
                                        and v not in FEATURE_LINEAR) else f'{v}.html')
            next_html_default = f'<a class="pager-next" href="{href}">{target_title}</a>'

    # 面包屑分组
    if in_features:
        group_html = (
            f'<span class="sep">›</span>'
            f'<a href="{prefix}features.html">特性</a>'
        )
    elif slug == 'index':
        group_html = ''
    else:
        group = SLUG_GROUP.get(slug)
        group_html = f'<span class="sep">›</span><span>{group}</span>' if group else ''

    if slug == 'index':
        breadcrumb = ''
    else:
        breadcrumb = (
            f'<nav class="breadcrumb">\n'
            f'      <a href="{prefix}index.html">首页</a>{group_html}'
            f'<span class="sep">›</span><span>{title}</span>\n'
            f'    </nav>'
        )
    out = TPL.format(
        title=title,
        prefix=prefix,
        header=render_header(prefix),
        sidebar=render_top_sidebar(prefix),
        body=body,
        breadcrumb=breadcrumb,
        prev_html=prev_html_default,
        next_html=next_html_default,
    )
    out_path = ROOT / f'{slug}.html'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding='utf-8')
    print(f'wrote {out_path.relative_to(ROOT)}')


def discover_all() -> list[str]:
    out = []
    for p in PAGES_DIR.rglob('*.html'):
        rel = p.relative_to(PAGES_DIR).with_suffix('')
        out.append(str(rel).replace(os.sep, '/'))
    return sorted(out)


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = discover_all()
    for slug in targets:
        render(slug)


if __name__ == '__main__':
    main()
