/* MindIE-LLM Wiki — wiki.js
 * 无外部依赖。提供：侧栏激活高亮 / 主题切换 / 自动 TOC / 代码复制 / 侧栏过滤 / 移动端菜单
 */
(function () {
  'use strict';

  // ===== 1. 主题切换 =====
  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem('mindie-wiki-theme'); } catch (e) {}
    var prefersDark = window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches;
    var theme = saved || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);

    var btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    function syncLabel() {
      var t = document.documentElement.getAttribute('data-theme');
      btn.textContent = t === 'dark' ? '☀ 浅色' : '☾ 暗色';
      btn.setAttribute('aria-label', '切换主题');
    }
    syncLabel();
    btn.addEventListener('click', function () {
      var cur = document.documentElement.getAttribute('data-theme');
      var next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('mindie-wiki-theme', next); } catch (e) {}
      syncLabel();
    });
  }

  // ===== 2. 侧栏当前页高亮 =====
  function highlightSidebar() {
    var path = location.pathname.split('/').pop() || 'index.html';
    if (path === '') path = 'index.html';
    var links = document.querySelectorAll('.sidebar nav a');
    links.forEach(function (a) {
      var href = (a.getAttribute('href') || '').split('/').pop();
      if (href === path) {
        a.classList.add('active');
        // 滚动到可见
        var rect = a.getBoundingClientRect();
        if (rect.top > window.innerHeight - 120 || rect.top < 64) {
          a.scrollIntoView({ block: 'center' });
        }
      }
    });
  }

  // ===== 3. 自动生成右侧 TOC =====
  function buildToc() {
    var tocEl = document.querySelector('.toc');
    if (!tocEl) return;
    var article = document.querySelector('article');
    if (!article) return;
    var headings = article.querySelectorAll('h2, h3');
    if (headings.length === 0) {
      tocEl.style.display = 'none';
      return;
    }
    var html = '<div class="toc-title">本页目录</div><ul>';
    headings.forEach(function (h, i) {
      if (!h.id) {
        h.id = 'h-' + i + '-' + (h.textContent || '')
          .trim().toLowerCase()
          .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
          .replace(/^-+|-+$/g, '')
          .slice(0, 60);
      }
      var lvl = h.tagName === 'H3' ? 'level-3' : 'level-2';
      html += '<li class="' + lvl + '"><a href="#' + h.id + '">'
        + (h.textContent || '').trim() + '</a></li>';
    });
    html += '</ul>';
    tocEl.innerHTML = html;

    // 滚动激活
    var tocLinks = tocEl.querySelectorAll('a');
    function onScroll() {
      var fromTop = window.scrollY + 120;
      var current = null;
      headings.forEach(function (h) {
        if (h.offsetTop <= fromTop) current = h;
      });
      tocLinks.forEach(function (a) {
        a.classList.remove('active');
        if (current && a.getAttribute('href') === '#' + current.id) {
          a.classList.add('active');
        }
      });
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // ===== 4. 代码块复制按钮 =====
  function addCopyButtons() {
    var pres = document.querySelectorAll('pre');
    pres.forEach(function (pre) {
      var btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.type = 'button';
      btn.textContent = '复制';
      btn.addEventListener('click', function () {
        var code = pre.querySelector('code') || pre;
        var text = code.innerText;
        var done = function () {
          btn.textContent = '已复制';
          btn.classList.add('copied');
          setTimeout(function () {
            btn.textContent = '复制';
            btn.classList.remove('copied');
          }, 1400);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done).catch(function () {
            fallbackCopy(text); done();
          });
        } else {
          fallbackCopy(text); done();
        }
      });
      pre.appendChild(btn);
    });
  }
  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }

  // ===== 5. 侧栏过滤 =====
  function initSidebarFilter() {
    var input = document.querySelector('.sidebar-filter');
    if (!input) return;
    var items = document.querySelectorAll('.sidebar nav li');
    var groups = document.querySelectorAll('.sidebar nav .nav-group');
    input.addEventListener('input', function () {
      var q = input.value.trim().toLowerCase();
      items.forEach(function (li) {
        var a = li.querySelector('a');
        var t = (a ? a.textContent : '').toLowerCase();
        li.style.display = !q || t.indexOf(q) !== -1 ? '' : 'none';
      });
      // 隐藏全为空的分组标题
      groups.forEach(function (g) {
        var visible = g.querySelectorAll('li')
          && Array.prototype.slice.call(g.querySelectorAll('li'))
            .some(function (li) { return li.style.display !== 'none'; });
        var title = g.querySelector('.nav-group-title');
        if (title) title.style.display = visible ? '' : 'none';
      });
    });
  }

  // ===== 6.5 侧栏二级折叠 =====
  function initSubGroups() {
    var toggles = document.querySelectorAll('.sidebar .nav-sub-toggle');
    toggles.forEach(function (btn) {
      var listId = btn.getAttribute('aria-controls');
      var list = listId && document.getElementById(listId);
      if (!list) return;
      // 当前页若位于该子组则默认展开
      var hasActive = list.querySelector('a.active');
      var open = !!hasActive;
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      list.classList.toggle('open', open);
      btn.addEventListener('click', function () {
        var nowOpen = list.classList.toggle('open');
        btn.setAttribute('aria-expanded', nowOpen ? 'true' : 'false');
      });
    });
  }

  // ===== 7. 移动端汉堡菜单 =====
  function initMenuToggle() {
    var btn = document.querySelector('.menu-toggle');
    var sidebar = document.querySelector('.sidebar');
    if (!btn || !sidebar) return;
    btn.addEventListener('click', function () {
      sidebar.classList.toggle('open');
    });
    // 点击主内容区时关闭
    document.querySelector('.content') &&
      document.querySelector('.content').addEventListener('click', function () {
        sidebar.classList.remove('open');
      });
  }

  // ===== 启动 =====
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }
  ready(function () {
    initTheme();
    highlightSidebar();
    buildToc();
    addCopyButtons();
    initSidebarFilter();
    initSubGroups();
    initMenuToggle();
  });
})();
