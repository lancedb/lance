/* Lance docs theme behaviours: theme toggle, GitHub stars, code block chrome,
   TOC scroll-spy, search overlay, mermaid rendering. */
(function () {
  "use strict";

  var BASE = (window.LD_BASE || ".").replace(/\/$/, "");

  /* ---------- theme toggle ---------- */
  var themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      var next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("ld-theme", next); } catch (e) { /* private mode */ }
    });
  }

  /* ---------- GitHub stars ---------- */
  (function loadStars() {
    var el = document.getElementById("gh-stars");
    if (!el) return;
    var TTL = 3600e3;
    function show(count) {
      if (!(count > 0)) return;
      var label = count >= 1000 ? (count / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(count);
      el.textContent = "★ " + label;
      el.hidden = false;
    }
    try {
      var cached = JSON.parse(localStorage.getItem("ld-gh-stars") || "null");
      if (cached && Date.now() - cached.t < TTL) { show(cached.v); return; }
    } catch (e) { /* fall through to fetch */ }
    fetch("https://api.github.com/repos/lance-format/lance")
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (d) {
        show(d.stargazers_count);
        try { localStorage.setItem("ld-gh-stars", JSON.stringify({ v: d.stargazers_count, t: Date.now() })); } catch (e) { /* ignore */ }
      })
      .catch(function () { /* rate-limited or offline: button still works without the count */ });
  })();

  /* ---------- article enhancements ---------- */
  var article = document.querySelector(".ld-article");

  if (article) {
    // Code blocks: wrap .highlight in a window with a language bar + copy button.
    article.querySelectorAll("div.highlight").forEach(function (hl) {
      if (hl.closest(".ld-code")) return;
      var code = hl.querySelector("code");
      var m = (hl.className + " " + (code ? code.className : "")).match(/language-([\w+-]+)/);
      var lang = m ? m[1] : "text";
      var wrap = document.createElement("div");
      wrap.className = "ld-code";
      var bar = document.createElement("div");
      bar.className = "ld-code__bar";
      bar.innerHTML = "<span></span><button class=\"ld-copy\" type=\"button\">Copy</button>";
      bar.querySelector("span").textContent = lang;
      hl.parentNode.insertBefore(wrap, hl);
      wrap.appendChild(bar);
      wrap.appendChild(hl);
    });

    document.addEventListener("click", function (e) {
      var copy = e.target.closest(".ld-copy");
      if (!copy) return;
      var pre = copy.closest(".ld-code").querySelector("pre");
      if (navigator.clipboard && pre) navigator.clipboard.writeText(pre.textContent);
      copy.textContent = "Copied";
      setTimeout(function () { copy.textContent = "Copy"; }, 1400);
    });

    // Tables: wrap for horizontal overflow.
    article.querySelectorAll("table").forEach(function (t) {
      if (t.closest(".ld-tablewrap")) return;
      var wrap = document.createElement("div");
      wrap.className = "ld-tablewrap";
      t.parentNode.insertBefore(wrap, t);
      wrap.appendChild(t);
    });

    // Mermaid: render fenced diagrams on demand.
    var mermaidNodes = article.querySelectorAll("pre.mermaid, div.mermaid");
    if (mermaidNodes.length) {
      import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs").then(function (mod) {
        var mermaid = mod.default;
        mermaidNodes.forEach(function (node) {
          if (node.tagName === "PRE") {
            var div = document.createElement("div");
            div.className = "mermaid";
            div.textContent = node.textContent;
            node.replaceWith(div);
          }
        });
        mermaid.initialize({ startOnLoad: false, securityLevel: "loose", theme: "neutral" });
        mermaid.run({ querySelector: ".ld-article div.mermaid" });
      }).catch(function () { /* offline: leave the diagram source visible */ });
    }
  }

  /* ---------- TOC scroll-spy ---------- */
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".ld-toc a"));
  if (tocLinks.length && article) {
    var targets = tocLinks.map(function (a) {
      var id = decodeURIComponent((a.getAttribute("href") || "").replace(/^#/, ""));
      return document.getElementById(id);
    });
    var update = function () {
      var active = 0;
      for (var i = 0; i < targets.length; i++) {
        if (targets[i] && targets[i].getBoundingClientRect().top <= 90) active = i;
      }
      tocLinks.forEach(function (a, i) { a.classList.toggle("active", i === active); });
    };
    window.addEventListener("scroll", update, { passive: true });
    update();
  }

  /* ---------- search overlay ---------- */
  var overlay = document.getElementById("search-overlay");
  var input = document.getElementById("search-input");
  var results = document.getElementById("search-results");
  var indexPromise = null;

  function loadIndex() {
    if (!indexPromise) {
      indexPromise = fetch(BASE + "/search/search_index.json")
        .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
        .then(function (d) {
          return d.docs.map(function (doc) {
            return {
              location: doc.location,
              title: doc.title || "",
              text: (doc.text || "").replace(/\s+/g, " "),
            };
          });
        });
    }
    return indexPromise;
  }

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlight(text, terms) {
    var out = esc(text);
    terms.forEach(function (t) {
      out = out.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>");
    });
    return out;
  }

  function search(docs, query) {
    var terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    var scored = [];
    docs.forEach(function (doc) {
      var title = doc.title.toLowerCase();
      var text = doc.text.toLowerCase();
      var score = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i];
        var inTitle = title.indexOf(t) !== -1;
        var inText = text.indexOf(t) !== -1;
        if (!inTitle && !inText) { score = 0; break; }
        score += (inTitle ? 10 : 0) + (inText ? 1 : 0);
      }
      // Prefer page-level entries slightly over deep anchors.
      if (score > 0) scored.push({ doc: doc, score: score + (doc.location.indexOf("#") === -1 ? 2 : 0) });
    });
    scored.sort(function (a, b) { return b.score - a.score; });
    return scored.slice(0, 12).map(function (s) { return s.doc; });
  }

  function snippet(text, terms) {
    var lower = text.toLowerCase();
    var pos = -1;
    for (var i = 0; i < terms.length; i++) {
      pos = lower.indexOf(terms[i]);
      if (pos !== -1) break;
    }
    if (pos === -1) pos = 0;
    var start = Math.max(0, pos - 60);
    var s = (start > 0 ? "…" : "") + text.slice(start, start + 160) + (start + 160 < text.length ? "…" : "");
    return s;
  }

  function render(docs, query) {
    var terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!docs.length) {
      results.innerHTML = "<div class=\"ld-search__empty\">No results</div>";
      return;
    }
    results.innerHTML = docs.map(function (doc) {
      var crumb = doc.location.split("#")[0].replace(/\/$/, "").replace(/\//g, " / ") || "home";
      return "<a class=\"ld-search__hit\" href=\"" + BASE + "/" + doc.location + "\">" +
        "<div class=\"ld-search__hit-crumb\">" + esc(crumb) + "</div>" +
        "<div class=\"ld-search__hit-title\">" + highlight(doc.title, terms) + "</div>" +
        "<div class=\"ld-search__hit-text\">" + highlight(snippet(doc.text, terms), terms) + "</div>" +
        "</a>";
    }).join("");
  }

  function openSearch() {
    overlay.hidden = false;
    input.value = "";
    results.innerHTML = "";
    input.focus();
    loadIndex();
  }
  function closeSearch() { overlay.hidden = true; }

  if (overlay && input && results) {
    var openBtn = document.getElementById("search-open");
    if (openBtn) openBtn.addEventListener("click", openSearch);
    overlay.addEventListener("click", function (e) {
      if (e.target.closest("[data-search-close]")) closeSearch();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && !overlay.hidden) { closeSearch(); return; }
      var typing = /^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement || {}).tagName || "");
      if ((e.key === "/" && !typing) || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k")) {
        e.preventDefault();
        if (overlay.hidden) openSearch(); else closeSearch();
      }
    });
    var pending = 0;
    input.addEventListener("input", function () {
      var q = input.value.trim();
      var seq = ++pending;
      if (q.length < 2) { results.innerHTML = ""; return; }
      loadIndex().then(function (docs) {
        if (seq !== pending) return;
        render(search(docs, q), q);
      }).catch(function () {
        results.innerHTML = "<div class=\"ld-search__empty\">Search index unavailable</div>";
      });
    });
  }
})();
