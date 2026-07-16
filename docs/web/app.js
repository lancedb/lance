/* Lance docs SPA controller: hash routing, chrome rendering, page loading. */
(function () {
  const LD = window.LanceDocs;
  const $ = (id) => document.getElementById(id);

  let route = "home";
  let loadSeq = 0;

  function renderTabs(secId) {
    $("top-tabs").innerHTML = LD.SECTIONS.map((s) => {
      const first = s.id === "home" ? "home" : LD.flatSection(s)[0].path;
      const cls = "ld-toptab" + (s.id === secId ? " active" : "");
      return '<a class="' + cls + '" href="#/' + first + '">' + s.label + "</a>";
    }).join("");
  }

  function renderSidenav(sec) {
    $("sidenav").innerHTML = sec.groups.map((g) =>
      '<div class="ld-sidenav__group"><div class="ld-sidenav__label">' + g.label + "</div>" +
      g.items.map((it) => {
        const href = it.path ? "#/" + it.path : it.ext;
        const target = it.path ? "" : ' target="_blank" rel="noopener"';
        const cls = it.path === route ? ' class="active"' : "";
        const ext = it.path ? "" : ' <span class="ext">↗</span>';
        return "<a" + cls + ' href="' + href + '"' + target + ">" + it.label + ext + "</a>";
      }).join("") +
      "</div>"
    ).join("");
  }

  function renderCrumbs(sec, label) {
    $("crumbs").innerHTML =
      "<span>Docs</span><span>/</span><span>" + sec.label + "</span><span>/</span><b>" + label + "</b>";
  }

  function renderPagenav(sec) {
    const flat = LD.flatSection(sec);
    const idx = flat.findIndex((it) => it.path === route);
    const prev = idx > 0 ? flat[idx - 1] : null;
    const next = idx >= 0 && idx < flat.length - 1 ? flat[idx + 1] : null;
    $("pagenav").innerHTML =
      (prev
        ? '<a href="#/' + prev.path + '"><div class="ld-pagenav__k">← Previous</div><div class="ld-pagenav__t">' + prev.label + "</div></a>"
        : "<span></span>") +
      (next
        ? '<a class="next" href="#/' + next.path + '"><div class="ld-pagenav__k">Next →</div><div class="ld-pagenav__t">' + next.label + "</div></a>"
        : "");
  }

  function renderChrome() {
    const secId = route === "home" ? "home" : LD.sectionOf(route);
    const isHome = secId === "home";
    renderTabs(secId);
    $("view-home").hidden = !isHome;
    $("view-docs").hidden = isHome;
    if (isHome) {
      document.title = "Lance Docs";
      return;
    }
    const sec = LD.SECTIONS.find((s) => s.id === secId);
    const found = LD.findItem(route);
    const label = found ? found.item.label : "";
    document.title = label ? label + " — Lance Docs" : "Lance Docs";
    renderSidenav(sec);
    renderCrumbs(sec, label);
    renderPagenav(sec);
  }

  function load(r) {
    if (r === "home" || !LD.findItem(r)) {
      route = "home";
      renderChrome();
      window.scrollTo(0, 0);
      return;
    }
    route = r;
    renderChrome();
    $("article").innerHTML = "";
    $("loading").hidden = false;
    window.scrollTo(0, 0);
    const seq = ++loadSeq;
    fetch(LD.CONTENT_BASE + r + ".md")
      .then((resp) => { if (!resp.ok) throw new Error(resp.status); return resp.text(); })
      .then((md) => {
        if (seq !== loadSeq) return;
        $("loading").hidden = true;
        $("article").innerHTML = LD.renderMarkdown(md, r).html;
      })
      .catch(() => {
        if (seq !== loadSeq) return;
        $("loading").hidden = true;
        $("article").innerHTML = "<p>Failed to load this page.</p>";
      });
  }

  function onHash() {
    const h = window.location.hash;
    load(h.startsWith("#/") ? h.slice(2) : "home");
  }

  document.addEventListener("click", (e) => {
    const copy = e.target.closest(".ld-copy");
    if (copy) {
      const pre = copy.closest(".ld-code").querySelector("pre code");
      if (navigator.clipboard) navigator.clipboard.writeText(pre.textContent);
      copy.textContent = "Copied";
      setTimeout(() => { copy.textContent = "Copy"; }, 1400);
      return;
    }
    const tab = e.target.closest(".ld-tabs__bar button");
    if (tab) {
      const root = tab.closest(".ld-tabs");
      const idx = +tab.dataset.tab;
      root.querySelectorAll(".ld-tabs__bar button").forEach((b, i) => b.classList.toggle("active", i === idx));
      root.querySelectorAll(":scope > .ld-tabs__panel").forEach((p, i) => p.classList.toggle("active", i === idx));
      return;
    }
    const anchor = e.target.closest("a[data-anchor]");
    if (anchor) {
      e.preventDefault();
      const el = document.getElementById(anchor.dataset.anchor);
      if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 76 });
    }
  });

  window.addEventListener("hashchange", onHash);
  onHash();
})();
