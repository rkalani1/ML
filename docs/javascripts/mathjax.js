window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

(function loadMathJaxOnlyWhenNeeded() {
  const bundle = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js";

  function renderMath() {
    if (!document.querySelector(".arithmatex")) return;
    if (typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise();
      return;
    }
    if (document.querySelector("script[data-mathjax-bundle]")) return;
    const script = document.createElement("script");
    script.src = bundle;
    script.defer = true;
    script.integrity = "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei";
    script.crossOrigin = "anonymous";
    script.dataset.mathjaxBundle = "3.2.2";
    document.head.appendChild(script);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMath, { once: true });
  } else {
    renderMath();
  }
  if (typeof document$ !== "undefined") document$.subscribe(renderMath);
})();
