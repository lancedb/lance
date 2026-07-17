/* Lance docs engine: nav model + mkdocs-flavored markdown renderer.
   Pages and assets are fetched from CONTENT_BASE (the mkdocs docs_dir). */
(function () {
  const CONTENT_BASE = "../src/";

  const P = (label, path) => ({ label, path });
  const X = (label, ext) => ({ label, ext });

  const SECTIONS = [
    { id: "home", label: "Home" },
    {
      id: "quickstart", label: "Getting Started", groups: [
        { label: "Getting Started", items: [
          P("Getting Started with Lance", "quickstart/index"),
          P("Versioning", "quickstart/versioning"),
          P("Vector Search", "quickstart/vector-search"),
          P("Full-Text Search", "quickstart/full-text-search"),
        ]},
      ],
    },
    {
      id: "guide", label: "User Guide", groups: [
        { label: "User Guide", items: [
          P("Read and Write", "guide/read_and_write"),
          P("Data Types", "guide/data_types"),
          P("Data Evolution", "guide/data_evolution"),
          P("Blob API", "guide/blob"),
          P("JSON Support", "guide/json"),
          P("Tags and Branches", "guide/tags_and_branches"),
          P("Object Store Configuration", "guide/object_store"),
          P("Observability", "guide/observability"),
          P("Distributed Write", "guide/distributed_write"),
          P("Distributed Indexing", "guide/distributed_indexing"),
          P("Migration Guide", "guide/migration"),
          P("Performance Guide", "guide/performance"),
          P("Tokenizer", "guide/tokenizer"),
          P("Extension Arrays", "guide/arrays"),
        ]},
      ],
    },
    {
      id: "format", label: "Specification", groups: [
        { label: "Specification", items: [P("Overview", "format/index")] },
        { label: "File Format", items: [
          P("Specification", "format/file/index"),
          P("Encoding Strategy", "format/file/encoding"),
          P("Versioning", "format/file/versioning"),
        ]},
        { label: "Table Format", items: [
          P("Overview", "format/table/index"),
          P("Schema", "format/table/schema"),
          P("Versioning", "format/table/versioning"),
          P("Transactions", "format/table/transaction"),
          P("Layout", "format/table/layout"),
          P("Branch & Tag", "format/table/branch_tag"),
          P("Row ID & Lineage", "format/table/row_id_lineage"),
          P("MemTable & WAL", "format/table/mem_wal"),
        ]},
        { label: "Index Formats", items: [P("Overview", "format/index/index")] },
        { label: "Scalar Indices", items: [
          P("BTree", "format/index/scalar/btree"),
          P("Bitmap", "format/index/scalar/bitmap"),
          P("Label List", "format/index/scalar/label_list"),
          P("Zone Map", "format/index/scalar/zonemap"),
          P("Bloom Filter", "format/index/scalar/bloom_filter"),
          P("Full Text Search", "format/index/scalar/fts"),
          P("N-gram", "format/index/scalar/ngram"),
          P("RTree", "format/index/scalar/rtree"),
        ]},
        { label: "Vector Indices", items: [P("Overview", "format/index/vector/index")] },
        { label: "System Indices", items: [
          P("Fragment Reuse", "format/index/system/frag_reuse"),
          P("MemWAL", "format/index/system/mem_wal"),
        ]},
        { label: "Catalog & Namespace", items: [
          X("Catalog Specs", "https://lance.org/format/catalog/"),
          X("Namespace Client Spec", "https://lance.org/format/namespace/"),
        ]},
      ],
    },
    {
      id: "examples", label: "Examples", groups: [
        { label: "Python", items: [
          P("LLM Dataset Creation", "examples/python/llm_dataset_creation"),
          P("LLM Training", "examples/python/llm_training"),
          P("Multimodal Dataset Creation", "examples/python/flickr8k_dataset_creation"),
          P("Multimodal Training", "examples/python/clip_training"),
          P("Deep Learning Artifact Management", "examples/python/artifact_management"),
        ]},
        { label: "Rust", items: [
          P("Write/Read Dataset", "examples/rust/write_read_dataset"),
          P("HNSW Vector Index", "examples/rust/hnsw"),
          P("LLM Dataset Creation", "examples/rust/llm_dataset_creation"),
        ]},
      ],
    },
    {
      id: "integrations", label: "Integrations", groups: [
        { label: "Integrations", items: [
          P("Overview", "integrations/index"),
          P("Apache DataFusion", "integrations/datafusion"),
          X("Apache Flink", "https://github.com/lance-format/lance-flink"),
          X("Apache Spark", "https://lance.org/integrations/spark/"),
          X("DuckDB", "https://lance.org/integrations/duckdb/"),
          X("Hugging Face", "https://lance.org/integrations/huggingface/"),
          X("PostgreSQL", "https://github.com/lance-format/pglance"),
          P("PyTorch", "integrations/pytorch"),
          X("Ray", "https://lance.org/integrations/ray/"),
          P("TensorFlow", "integrations/tensorflow"),
          X("Trino", "https://lance.org/integrations/trino/"),
        ]},
      ],
    },
    {
      id: "community", label: "Community", groups: [
        { label: "Community", items: [
          P("Overview", "community/index"),
          P("Maintainers", "community/maintainers"),
          P("PMC", "community/pmc"),
          P("Communication", "community/communication"),
          P("Voting", "community/voting"),
          P("Contributing", "community/contributing"),
          P("Release", "community/release"),
          P("Project Specific Guidelines", "community/project-specific/index"),
        ]},
      ],
    },
    {
      id: "sdk", label: "SDK Docs", groups: [
        { label: "SDK Docs", items: [P("SDK Documentation", "sdk_docs")] },
      ],
    },
  ];

  function sectionOf(path) {
    if (path === "sdk_docs") return "sdk";
    const head = path.split("/")[0];
    return SECTIONS.some((s) => s.id === head) ? head : "home";
  }
  function findItem(path) {
    for (const s of SECTIONS) {
      for (const g of s.groups || []) {
        for (const it of g.items) if (it.path === path) return { section: s, group: g, item: it };
      }
    }
    return null;
  }
  function flatSection(sec) {
    const out = [];
    for (const g of sec.groups || []) for (const it of g.items) if (it.path) out.push(it);
    return out;
  }

  /* ---------- markdown renderer ---------- */
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const slug = (s) => s.toLowerCase().replace(/<[^>]*>/g, "").replace(/[^\w\s-]/g, "").trim().replace(/\s+/g, "-");

  function resolvePath(baseDir, href) {
    const parts = (baseDir ? baseDir.split("/") : []).filter(Boolean);
    for (const seg of href.split("/")) {
      if (seg === "" || seg === ".") continue;
      else if (seg === "..") parts.pop();
      else parts.push(seg);
    }
    return parts.join("/");
  }

  function resolveLink(baseDir, href) {
    if (/^(https?:|mailto:)/.test(href)) return { ext: href };
    if (href.startsWith("#")) return { anchor: href.slice(1) };
    const [p0] = href.split("#");
    let p = resolvePath(baseDir, p0).replace(/\.md$/, "").replace(/\/$/, "");
    if (/\.(png|gif|svg|jpg|jpeg|webp)$/i.test(p)) return { asset: CONTENT_BASE + p };
    const cand = findItem(p) ? p : findItem(p + "/index") ? p + "/index" : null;
    if (cand) return { route: cand };
    // Content not bundled in this prototype (subproject docs assembled by
    // make-full-website.sh) lives on lance.org; mkdocs serves `foo/index.md`
    // at `foo/`, so drop the trailing `/index` and keep a directory URL.
    return { ext: "https://lance.org/" + p.replace(/\/index$/, "") + "/" };
  }

  function inline(text, baseDir) {
    let out = esc(text);
    const stash = [];
    const keep = (html) => { stash.push(html); return "\u0000" + (stash.length - 1) + "\u0000"; };
    out = out.replace(/`([^`]+)`/g, (_, c) => keep("<code>" + c + "</code>"));
    out = out.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, alt, src) => {
      const r = resolveLink(baseDir, src);
      const url = r.asset || r.ext || src;
      return keep('<span class="ld-fig"><img src="' + url + '" alt="' + alt + '" loading="lazy"></span>');
    });
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, txt, href) => {
      const r = resolveLink(baseDir, href);
      if (r.route) return keep('<a href="#/' + r.route + '">' + txt + "</a>");
      if (r.anchor) return keep('<a href="#' + r.anchor + '" data-anchor="' + r.anchor + '">' + txt + "</a>");
      const url = r.asset || r.ext;
      return keep('<a href="' + url + '" target="_blank" rel="noopener">' + txt + "</a>");
    });
    // autolink bare URLs (markdown-authored links are already stashed above)
    out = out.replace(/(^|[\s(|])(https?:\/\/[^\s<>()|]+?)([.,;:!?]?)(?=$|[\s<>()|])/g, (_, pre, url, tail) =>
      pre + keep('<a href="' + url + '" target="_blank" rel="noopener">' + url + "</a>") + tail);
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
    out = out.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[+i]);
    return out;
  }

  const TONES = {
    note: ["info", "Note"], info: ["info", "Info"], tip: ["info", "Tip"], hint: ["info", "Hint"],
    example: ["info", "Example"], abstract: ["info", "Abstract"], question: ["info", "Question"],
    warning: ["warn", "Warning"], caution: ["warn", "Caution"], attention: ["warn", "Attention"],
    danger: ["danger", "Danger"], error: ["danger", "Error"], bug: ["danger", "Bug"], failure: ["danger", "Failure"],
    success: ["ok", "Success"], check: ["ok", "Success"],
  };

  function collectIndented(lines, i) {
    const body = [];
    while (i < lines.length) {
      const l = lines[i];
      if (l.trim() === "") { body.push(""); i++; continue; }
      if (/^(\t| {4})/.test(l)) { body.push(l.replace(/^(\t| {4})/, "")); i++; continue; }
      break;
    }
    while (body.length && body[body.length - 1] === "") body.pop();
    return { body, next: i };
  }

  function protoBlock(fq) {
    const name = fq.split(".").pop();
    return '<div class="ld-proto"><span class="ld-proto__k">protobuf</span><code>message ' + esc(name) + '</code><a href="https://github.com/lance-format/lance/tree/main/protos" target="_blank" rel="noopener">View definition in protos/ →</a></div>';
  }

  function renderLines(lines, baseDir) {
    let html = "", i = 0, para = [];
    const flush = () => { if (para.length) { html += "<p>" + inline(para.join(" "), baseDir) + "</p>"; para = []; } };
    while (i < lines.length) {
      let line = lines[i];
      if (line.trim() === "") { flush(); i++; continue; }
      let m;
      // fenced code
      if ((m = line.match(/^```(\S*)/))) {
        flush();
        const lang = m[1] || "";
        const code = [];
        i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) { code.push(lines[i]); i++; }
        i++;
        const only = code.filter((l) => l.trim() !== "");
        let pm;
        if (only.length === 1 && (pm = only[0].match(/^%{3}\s*(\S+)\s*%{3}\s*$/))) {
          html += protoBlock(pm[1]);
          continue;
        }
        const label = lang === "mermaid" ? "mermaid · diagram source" : lang || "text";
        html += '<div class="ld-code"><div class="ld-code__bar"><span>' + esc(label) + '</span><button class="ld-copy" type="button">Copy</button></div><pre><code>' + esc(code.join("\n")) + "</code></pre></div>";
        continue;
      }
      // proto include
      if ((m = line.match(/^%{3}\s*(\S+)\s*%{3}\s*$/))) {
        flush();
        html += protoBlock(m[1]);
        i++; continue;
      }
      // details / summary blocks
      if (/^\s*<details/i.test(line)) {
        flush();
        // collect until matching </details> (may nest fences inside)
        const block = [];
        let depth = 0;
        while (i < lines.length) {
          const l = lines[i];
          if (/<details/i.test(l)) depth++;
          if (/<\/details>/i.test(l)) depth--;
          block.push(l);
          i++;
          if (depth <= 0) break;
        }
        // strip outer tags, extract summary
        let inner = block.join("\n").replace(/^\s*<details[^>]*>/i, "").replace(/<\/details>\s*$/i, "");
        let summary = "Details";
        const sm = inner.match(/<summary[^>]*>([\s\S]*?)<\/summary>/i);
        if (sm) { summary = sm[1].replace(/<[^>]+>/g, "").trim(); inner = inner.replace(sm[0], ""); }
        html += '<details class="ld-details"><summary>' + esc(summary) + "</summary><div>" + renderLines(inner.split("\n"), baseDir) + "</div></details>";
        continue;
      }
      if (/^\s*<\/\w+>\s*$/.test(line)) { i++; continue; } // stray closing tag
      // snippet include
      if ((m = line.match(/^--8<--\s+"([^"]+)"/))) {
        flush();
        html += '<div class="ld-proto"><span class="ld-proto__k">include</span><code>' + esc(m[1]) + '</code><a href="https://github.com/lance-format/lance" target="_blank" rel="noopener">View source on GitHub →</a></div>';
        i++; continue;
      }
      // admonition
      if ((m = line.match(/^(!!!|\?\?\?\+?)\s+(\w+)(?:\s+"([^"]*)")?\s*$/))) {
        flush();
        const [tone, defTitle] = TONES[m[2].toLowerCase()] || ["info", m[2]];
        const { body, next } = collectIndented(lines, i + 1);
        i = next;
        html += '<div class="ld-callout ld-callout--' + tone + '"><div class="ld-callout__t">' + esc(m[3] || defTitle) + '</div><div class="ld-callout__b">' + renderLines(body, baseDir) + "</div></div>";
        continue;
      }
      // content tabs
      if ((m = line.match(/^===\s+"([^"]+)"\s*$/))) {
        flush();
        const tabs = [];
        while (i < lines.length && (m = lines[i].match(/^===\s+"([^"]+)"\s*$/))) {
          const { body, next } = collectIndented(lines, i + 1);
          tabs.push({ label: m[1], html: renderLines(body, baseDir) });
          i = next;
          while (i < lines.length && lines[i].trim() === "") i++;
        }
        html += '<div class="ld-tabs"><div class="ld-tabs__bar">' +
          tabs.map((t, k) => '<button type="button" class="' + (k === 0 ? "active" : "") + '" data-tab="' + k + '">' + esc(t.label) + "</button>").join("") +
          "</div>" +
          tabs.map((t, k) => '<div class="ld-tabs__panel' + (k === 0 ? " active" : "") + '">' + t.html + "</div>").join("") +
          "</div>";
        continue;
      }
      // heading
      if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
        flush();
        const lvl = m[1].length;
        const txt = inline(m[2].replace(/\s*\{[:#][^}]*\}\s*$/, ""), baseDir);
        html += "<h" + lvl + ' id="' + slug(m[2]) + '">' + txt + "</h" + lvl + ">";
        i++; continue;
      }
      // hr
      if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flush(); html += "<hr>"; i++; continue; }
      // blockquote
      if (/^>\s?/.test(line)) {
        flush();
        const q = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) { q.push(lines[i].replace(/^>\s?/, "")); i++; }
        html += "<blockquote>" + renderLines(q, baseDir) + "</blockquote>";
        continue;
      }
      // table
      if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
        flush();
        const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
        const head = cells(line);
        i += 2;
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(cells(lines[i])); i++; }
        html += '<div class="ld-tablewrap"><table><thead><tr>' + head.map((h) => "<th>" + inline(h, baseDir) + "</th>").join("") + "</tr></thead><tbody>" +
          rows.map((r) => "<tr>" + r.map((c) => "<td>" + inline(c, baseDir) + "</td>").join("") + "</tr>").join("") + "</tbody></table></div>";
        continue;
      }
      // lists
      if (/^(\s*)([-*+]|\d+[.)])\s+/.test(line)) {
        flush();
        const res = renderList(lines, i, baseDir);
        html += res.html;
        i = res.next;
        continue;
      }
      // raw html block
      if (/^\s*<\w+/.test(line)) {
        flush();
        const raw = [];
        while (i < lines.length && lines[i].trim() !== "") { raw.push(lines[i]); i++; }
        html += raw.join("\n").replace(/src="([^"]+)"/g, (all, src) => {
          if (/^(https?:|data:)/.test(src)) return all;
          return 'src="' + CONTENT_BASE + resolvePath(baseDir, src) + '"';
        });
        continue;
      }
      para.push(line.trim());
      i++;
    }
    flush();
    return html;
  }

  function renderList(lines, i, baseDir) {
    const itemRe = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
    const first = lines[i].match(itemRe);
    const baseIndent = first[1].length;
    const ordered = /\d/.test(first[2]);
    let html = ordered ? "<ol>" : "<ul>";
    while (i < lines.length) {
      const m = lines[i] && lines[i].match(itemRe);
      if (!m || m[1].length < baseIndent) break;
      if (m[1].length > baseIndent) { const sub = renderList(lines, i, baseDir); html = html.replace(/<\/li>$/, sub.html + "</li>"); i = sub.next; continue; }
      let content = m[3];
      i++;
      // continuation lines (indented text under the item)
      const cont = [];
      while (i < lines.length) {
        const l = lines[i];
        if (l.trim() === "") {
          if (i + 1 < lines.length && /^\s{2,}/.test(lines[i + 1]) && !itemRe.test(lines[i + 1])) { cont.push(""); i++; continue; }
          break;
        }
        if (itemRe.test(l)) { const mm = l.match(itemRe); if (mm[1].length <= baseIndent) break; else break; }
        if (/^\s{2,}/.test(l)) { cont.push(l.replace(new RegExp("^\\s{0," + (baseIndent + 4) + "}"), "")); i++; continue; }
        break;
      }
      let extra = "";
      if (cont.length) extra = renderLines(cont, baseDir);
      html += "<li>" + inline(content, baseDir) + extra + "</li>";
      // nested list directly after?
      if (i < lines.length && lines[i] && itemRe.test(lines[i])) {
        const mm = lines[i].match(itemRe);
        if (mm[1].length > baseIndent) { const sub = renderList(lines, i, baseDir); html = html.replace(/<\/li>$/, sub.html + "</li>"); i = sub.next; }
      }
    }
    html += ordered ? "</ol>" : "</ul>";
    return { html, next: i };
  }

  function renderMarkdown(md, pagePath) {
    const baseDir = pagePath.includes("/") ? pagePath.split("/").slice(0, -1).join("/") : "";
    let src = md.replace(/\r\n/g, "\n");
    let title = null;
    const fm = src.match(/^---\n([\s\S]*?)\n---\n/);
    if (fm) {
      const t = fm[1].match(/^title:\s*(.+)$/m);
      if (t) title = t[1].trim().replace(/^["']|["']$/g, "");
      src = src.slice(fm[0].length);
    }
    src = src.replace(/\{:[^}]*\}/g, "");
    const html = renderLines(src.split("\n"), baseDir);
    return { html, title };
  }

  window.LanceDocs = { CONTENT_BASE, SECTIONS, sectionOf, findItem, flatSection, renderMarkdown };
})();
