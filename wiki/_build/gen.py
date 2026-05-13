#!/usr/bin/env python3
"""MindIE-LLM Wiki 页面生成器

读取 _build/pages/<slug>.html 中的内容片段（仅 <article> 内部 + 元数据 YAML 头），
用统一的 header/sidebar/footer 模板包裹后输出到 wiki/<slug>.html。

YAML 头格式（仅前 3 行）：
---
title: 页面标题
prev: prev-slug | 空
next: next-slug | 空
---

剩余内容是 article 的 innerHTML。
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / '_build' / 'pages'

# 顺序（用于自动 prev/next 推断、面包屑）
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

SIDEBAR = """<input class="sidebar-filter" placeholder="筛选目录…" aria-label="筛选目录">
    <nav>
      <div class="nav-group">
        <div class="nav-group-title">入门</div>
        <ul>
          <li><a href="index.html">首页</a></li>
          <li><a href="introduction.html">项目简介</a></li>
          <li><a href="quick-start.html">快速入门</a></li>
          <li><a href="installation.html">安装与编译</a></li>
        </ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">架构与内核</div>
        <ul>
          <li><a href="architecture.html">架构总览</a></li>
          <li><a href="directory.html">目录结构</a></li>
          <li><a href="server-api.html">服务端与 API</a></li>
          <li><a href="cpp-engine.html">C++ 引擎内核</a></li>
          <li><a href="scheduler.html">调度器</a></li>
          <li><a href="block-manager.html">KV Cache &amp; Block</a></li>
          <li><a href="python-runtime.html">Python 推理框架</a></li>
        </ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">特性</div>
        <ul>
          <li><a href="features.html">特性总览</a></li>
          <li><a href="quantization.html">量化特性</a></li>
          <li><a href="parallelism.html">并行策略</a></li>
          <li><a href="acceleration.html">加速特性</a></li>
        </ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">开发</div>
        <ul>
          <li><a href="development.html">开发与贡献</a></li>
        </ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">参考</div>
        <ul>
          <li><a href="faq.html">常见问题</a></li>
          <li><a href="references.html">参考资料</a></li>
        </ul>
      </div>
    </nav>"""

HEADER = """<header class="site-header">
  <a class="brand" href="index.html">
    <img src="assets/img/ascend-logo.png" alt="Ascend">
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
<meta name="description" content="{title} — MindIE-LLM Wiki（华为昇腾大语言模型推理引擎）">
<link rel="icon" href="assets/img/favicon.ico">
<link rel="stylesheet" href="assets/css/wiki.css">
</head>
<body>
{header}

<div class="layout">
  <aside class="sidebar">
    {sidebar}
  </aside>

  <main class="content">
    <nav class="breadcrumb">
      <a href="index.html">首页</a>{group_html}<span class="sep">›</span><span>{title}</span>
    </nav>
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

<script src="assets/js/wiki.js"></script>
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


def render(slug: str):
    page_path = PAGES_DIR / f'{slug}.html'
    head, body = parse_page(page_path.read_text(encoding='utf-8'))
    title = head.get('title') or SLUG_TITLE.get(slug, slug)

    # prev / next: 优先 YAML 指定，其次自动按顺序推断
    idx = SLUGS.index(slug) if slug in SLUGS else -1
    prev_slug = head.get('prev')
    next_slug = head.get('next')
    if prev_slug is None and idx > 0:
        prev_slug = SLUGS[idx - 1]
    if next_slug is None and 0 <= idx < len(SLUGS) - 1:
        next_slug = SLUGS[idx + 1]
    prev_slug = (prev_slug or '').strip() or None
    next_slug = (next_slug or '').strip() or None

    prev_html = ''
    if prev_slug:
        prev_html = f'<a class="pager-prev" href="{prev_slug}.html">{SLUG_TITLE[prev_slug]}</a>'
    else:
        prev_html = '<span></span>'

    next_html = ''
    if next_slug:
        next_html = f'<a class="pager-next" href="{next_slug}.html">{SLUG_TITLE[next_slug]}</a>'
    else:
        next_html = '<span></span>'

    group = SLUG_GROUP.get(slug)
    group_html = ''
    if group:
        group_html = f'<span class="sep">›</span><span>{group}</span>'

    out = TPL.format(
        title=title,
        header=HEADER,
        sidebar=SIDEBAR,
        body=body,
        group_html=group_html,
        prev_html=prev_html,
        next_html=next_html,
    )
    out_path = ROOT / f'{slug}.html'
    out_path.write_text(out, encoding='utf-8')
    print(f'wrote {out_path.relative_to(ROOT)}')


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = [p.stem for p in PAGES_DIR.glob('*.html')]
    for slug in targets:
        render(slug)


if __name__ == '__main__':
    main()
