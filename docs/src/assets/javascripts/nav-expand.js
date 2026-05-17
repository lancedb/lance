// Auto-expand sidebar navigation to 2 levels on page load.
// Level 1 sections are already expanded by navigation.sections.
// This expands level 2 (e.g. Operations, Models become visible)
// but leaves level 3+ collapsed.
document.addEventListener("DOMContentLoaded", function () {
  // In mkdocs-material with navigation.sections, the top-level items
  // are rendered as non-collapsible sections. The collapsible items
  // start at the next level. We want to expand one more level.
  //
  // Selector: inside the primary nav, find toggle checkboxes that are
  // exactly 2 nesting levels deep (the second-level sections).
  var toggles = document.querySelectorAll(
    ".md-sidebar--primary .md-nav--primary > .md-nav__list > .md-nav__item > .md-nav > .md-nav__list > .md-nav__item > .md-nav__toggle"
  );
  toggles.forEach(function (t) { t.checked = true; });
});
