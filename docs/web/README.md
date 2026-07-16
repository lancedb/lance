# Lance Docs — design prototype site

Static implementation of the "Lance Docs" design from the Claude Design project
(`Lance Docs.dc.html`). It is a self-contained hash-routed site that renders the
existing markdown sources in `docs/src` — no content is duplicated here.

## Preview

Serve the `docs/` directory (the app fetches pages from `../src/`):

```bash
cd docs
python3 -m http.server 8000
```

Then open <http://localhost:8000/web/>.

## Layout

- `index.html` — page skeleton: header, home view, docs view, footer.
- `site.css` — page chrome and article prose styles.
- `app.js` — hash router and chrome rendering.
- `docs-app.js` — navigation model and mkdocs-flavored markdown renderer.
  `CONTENT_BASE` at the top points at the mkdocs `docs_dir`.
- `ds/tokens/` — Lance design-system tokens (colors, typography, spacing,
  effects, base), synced from the design project.

## Notes

- The markdown renderer covers the mkdocs extensions used in `docs/src`
  (admonitions, content tabs, tables, `<details>`, proto includes); unsupported
  external links fall back to `https://lance.org/<path>`.
- Navigation lives in `SECTIONS` in `docs-app.js`; adding a page to the site
  means adding it there (mkdocs `.pages` files are not read).
