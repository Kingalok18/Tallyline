// Theme toggle, side panel and keyboard shortcuts.
(function () {
  var root = document.documentElement;
  var themeButton = document.getElementById("theme-toggle");

  function currentTheme() {
    var set = root.getAttribute("data-theme");
    if (set) return set;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function labelTheme() {
    themeButton.textContent = currentTheme() === "dark" ? "Light theme" : "Dark theme";
  }
  themeButton.addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("tallyline-theme", next); } catch (e) {}
    labelTheme();
  });
  labelTheme();

  var addCaseForm = document.getElementById("add-case-form");
  if (addCaseForm) {
    addCaseForm.addEventListener("submit", function () {
      addCaseForm.querySelector("button[type=submit]").disabled = true;
      document.getElementById("add-case-progress").classList.add("htmx-request");
    });
  }

  var panel = document.getElementById("side-panel");
  function closePanel() { panel.classList.remove("open"); }
  document.getElementById("side-panel-close").addEventListener("click", closePanel);

  document.body.addEventListener("htmx:afterSwap", function (event) {
    var target = event.detail.target;
    if (target.id === "side-panel-body") {
      panel.classList.add("open");
      var marked = target.querySelector(".hl");
      if (marked) marked.scrollIntoView({ block: "center" });
    }
    if (target.id === "doc-view") {
      var mark = target.querySelector(".hl");
      if (mark) mark.scrollIntoView({ block: "center" });
    }
    if (target.id === "case-rows") selected = -1;
  });

  document.body.addEventListener("click", function (event) {
    var tab = event.target.closest(".tab");
    if (tab) {
      document.querySelectorAll(".tab").forEach(function (t) { t.classList.remove("active"); });
      tab.classList.add("active");
    }
    var row = event.target.closest(".case-row");
    if (row && !event.target.closest("a")) window.location = row.dataset.href;
  });

  var selected = -1;
  function rows() { return Array.prototype.slice.call(document.querySelectorAll(".case-row")); }
  function select(next) {
    var all = rows();
    if (!all.length) return;
    selected = Math.max(0, Math.min(all.length - 1, next));
    all.forEach(function (row, i) { row.classList.toggle("selected", i === selected); });
    all[selected].scrollIntoView({ block: "nearest" });
  }

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closePanel();
      if (document.activeElement) document.activeElement.blur();
      return;
    }
    var tag = (event.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "/") {
      var search = document.getElementById("search");
      if (search) { event.preventDefault(); search.focus(); }
    } else if (event.key === "a") {
      var question = document.getElementById("question");
      if (question) { event.preventDefault(); question.focus(); }
    } else if (event.key === "j") {
      select(selected + 1);
    } else if (event.key === "k") {
      select(selected - 1);
    } else if (event.key === "Enter" && selected >= 0 && tag !== "button" && tag !== "a") {
      var all = rows();
      if (all[selected]) window.location = all[selected].dataset.href;
    }
  });
})();
