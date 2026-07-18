// Lazy-load below-the-fold figures. The first image on a page stays eager
// (likely the LCP element); the rest load lazily. Re-runs on instant navigation.
(function () {
  function lazyify() {
    var imgs = document.querySelectorAll(".md-content img");
    imgs.forEach(function (img, i) {
      if (i > 0) img.setAttribute("loading", "lazy");
      else img.setAttribute("loading", "eager");
    });
  }
  if (typeof document$ !== "undefined") {
    document$.subscribe(lazyify);
  } else if (document.readyState !== "loading") {
    lazyify();
  } else {
    document.addEventListener("DOMContentLoaded", lazyify, { once: true });
  }
})();
