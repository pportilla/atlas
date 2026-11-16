// Atlas documentation: navigation drawer, theme, search, copy buttons and maths.
"use strict";

const root = document.body.dataset.root || "";

// Navigation drawer on narrow screens.
const toggle = document.querySelector(".nav-toggle");
toggle?.addEventListener("click", () => {
  const open = document.body.classList.toggle("nav-open");
  toggle.setAttribute("aria-expanded", String(open));
  toggle.setAttribute("aria-label", open ? "Hide navigation" : "Show navigation");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("nav-open")) toggle.click();
});

// Theme: follow the system, or keep the reader's choice.
const themes = ["auto", "light", "dark"];
const themeButton = document.querySelector(".theme-toggle");
function showTheme() {
  const theme = document.documentElement.dataset.theme || "auto";
  themeButton.title = theme === "auto" ? "Theme follows the system" : `${theme[0].toUpperCase()}${theme.slice(1)} theme`;
}
themeButton?.addEventListener("click", () => {
  const next = themes[(themes.indexOf(document.documentElement.dataset.theme || "auto") + 1) % themes.length];
  if (next === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = next;
  try {
    if (next === "auto") localStorage.removeItem("atlas-theme");
    else localStorage.setItem("atlas-theme", next);
  } catch (error) { /* The choice lasts for this page only. */ }
  showTheme();
});
if (themeButton) showTheme();

// Maths, set by KaTeX when the page loaded it.
if (window.katex) {
  for (const element of document.querySelectorAll(".math")) {
    katex.render(element.textContent, element, {
      displayMode: element.classList.contains("math-display"), throwOnError: false,
    });
  }
}

// Copy buttons copy the input lines of a transcript, without the prompt or output.
for (const block of document.querySelectorAll("pre.code")) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy-button";
  button.textContent = "Copy";
  button.addEventListener("click", async () => {
    const inputs = block.querySelectorAll(".line.in");
    const text = inputs.length
      ? [...inputs].map((line) => line.textContent.replace(/^(atlas>|\.\.\.) ?/, "")).join("\n")
      : block.textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
    } catch (error) {
      button.textContent = "Select and copy";
    }
    setTimeout(() => { button.textContent = "Copy"; }, 1600);
  });
  block.append(button);
}

// The section in view is marked in the local table of contents.
const tocLinks = [...document.querySelectorAll(".local-toc a")];
if (tocLinks.length && "IntersectionObserver" in window) {
  const byId = new Map(tocLinks.map((link) => [decodeURIComponent(link.hash.slice(1)), link]));
  const visible = new Set();
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) visible.add(entry.target.id);
      else visible.delete(entry.target.id);
    }
    const headings = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);
    const current = headings.find((heading) => visible.has(heading.id))
      || headings.filter((heading) => heading.getBoundingClientRect().top < 120).pop();
    tocLinks.forEach((link) => link.classList.toggle("active", current && byId.get(current.id) === link));
  }, { rootMargin: "-72px 0px -60% 0px" });
  for (const id of byId.keys()) {
    const heading = document.getElementById(id);
    if (heading) observer.observe(heading);
  }
}

// Search over the pages, sections and functions listed in search-index.js.
const input = document.getElementById("search");
const results = document.getElementById("search-results");
function searchable(text) {
  return text.normalize("NFKC").toLowerCase().replace(/\ba_e\b|\ba_e(?=_)/g, "ae")
    .replace(/[_‐‑–-]+/g, " ").replace(/\s+/g, " ").trim();
}
const entries = (window.ATLAS_SEARCH || []).map((entry) => ({
  ...entry,
  haystack: searchable(`${entry.title} ${entry.page} ${entry.text}`),
  names: [entry.title, ...(entry.aliases || [])].map(searchable),
}));
let selected = -1;

function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function mark(text, terms) {
  const words = terms.filter((term) => term.length >= 2).map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!words.length) return escapeHtml(text);
  return text.split(new RegExp(`(${words.join("|")})`, "gi"))
    .map((part, i) => i % 2 ? `<mark>${escapeHtml(part)}</mark>` : escapeHtml(part)).join("");
}
function snippet(text, terms) {
  const lower = text.toLowerCase();
  const hits = terms.map((term) => lower.indexOf(term)).filter((i) => i >= 0);
  const at = hits.length ? Math.max(0, Math.min(...hits) - 60) : 0;
  return (at > 0 ? "…" : "") + text.slice(at, at + 180);
}
function search(query) {
  const phrase = searchable(query);
  const terms = phrase.split(" ").filter(Boolean);
  if (!terms.length) return [];
  const scored = [];
  for (const entry of entries) {
    if (!terms.every((term) => entry.haystack.includes(term))) continue;
    let score = entry.names.includes(phrase) ? 80 : 0;
    for (const term of terms) {
      if (entry.names.some((name) => name.startsWith(term))) score += 18;
      else if (entry.names.some((name) => name.includes(term))) score += 10;
      if (entry.page.toLowerCase().includes(term)) score += 3;
    }
    if (entry.kind === "function") {
      score += 4;
      if (searchable(entry.text).includes(phrase)) score += 20;
    }
    scored.push([score, entry]);
  }
  return scored.sort((a, b) => b[0] - a[0]).slice(0, 14).map(([, entry]) => entry);
}
function render() {
  const query = input.value.trim();
  if (!query) { results.hidden = true; return; }
  const terms = searchable(query).split(" ").filter(Boolean);
  const found = search(query);
  selected = found.length ? 0 : -1;
  results.innerHTML = found.length
    ? found.map((entry, i) => `<li><a role="option" href="${root}${entry.url}" class="${entry.kind === "function" ? "result-function" : ""}"
        aria-selected="${i === 0}"><span class="result-title">${mark(entry.title, terms)}</span>
        <span class="result-page">${escapeHtml(entry.kind === "function" ? "Function" : entry.kind)}${entry.title !== entry.page ? " · " + escapeHtml(entry.page) : ""}</span>
        <span class="result-text">${mark(snippet(entry.text, terms), terms)}</span></a></li>`).join("")
    : `<li class="empty">Nothing matches “${escapeHtml(query)}”.</li>`;
  results.hidden = false;
}
function choose(step) {
  const links = [...results.querySelectorAll("a")];
  if (!links.length) return;
  selected = (selected + step + links.length) % links.length;
  links.forEach((link, i) => link.setAttribute("aria-selected", String(i === selected)));
  links[selected].scrollIntoView({ block: "nearest" });
}
input?.addEventListener("input", render);
input?.addEventListener("focus", () => { if (input.value.trim()) render(); });
input?.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown") { event.preventDefault(); choose(1); }
  else if (event.key === "ArrowUp") { event.preventDefault(); choose(-1); }
  else if (event.key === "Enter") {
    const link = results.querySelectorAll("a")[selected];
    if (link) { event.preventDefault(); window.location.href = link.href; }
  } else if (event.key === "Escape") { input.value = ""; results.hidden = true; input.blur(); }
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".search")) results.hidden = true;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && !event.target.closest("input, textarea") && input) {
    event.preventDefault();
    input.focus();
  }
});
