// Shared helpers. Everything that goes into innerHTML must pass through esc().
// Values here come from file names, CSV files, the sample table and error messages, so none of them is trusted.
(function () {
  const ENTITIES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

  window.esc = function (value) {
    if (value === null || value === undefined) return "";
    return String(value).replace(/[&<>"']/g, (c) => ENTITIES[c]);
  };

  // data the server put into the page as a JSON block (no inline scripts: the CSP forbids them)
  window.pageData = function () {
    const el = document.getElementById("page-data");
    return el ? JSON.parse(el.textContent) : {};
  };

  // only same-site, root-relative URLs may go into href/src
  window.safeUrl = function (url) {
    return typeof url === "string" && url.startsWith("/") && !url.startsWith("//") ? url : "#";
  };

  // a status word that is used as a CSS class: letters only
  window.cssClass = function (value) {
    return String(value || "").toLowerCase().replace(/[^a-z_-]/g, "");
  };
})();
