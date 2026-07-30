(() => {
  const selector = [
    'label.md-header__button[for="__drawer"]',
    'label.md-header__button[for="__search"]',
  ].join(",");

  const enhance = () => {
    document.querySelectorAll(selector).forEach((button) => {
      if (button.dataset.keyboardToggle === "true") return;
      const toggle = document.getElementById(button.htmlFor);
      if (!toggle) return;

      const sync = () => {
        button.setAttribute("aria-expanded", String(toggle.checked));
      };

      button.dataset.keyboardToggle = "true";
      button.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        event.stopPropagation();
        if (event.repeat) return;
        toggle.click();
      });
      button.addEventListener("keyup", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        event.stopPropagation();
      });
      toggle.addEventListener("change", sync);
      sync();
    });
    document.querySelectorAll("button.md-code__button").forEach((button) => {
      if (!button.hasAttribute("aria-label")) {
        button.setAttribute("aria-label", "Copy code to clipboard");
      }
      if (!button.hasAttribute("type")) button.setAttribute("type", "button");
    });

    const query = document.querySelector('[data-md-component="search-query"]');
    const result = document.querySelector('[data-md-component="search-result"]');
    const meta = result && result.querySelector(".md-search-result__meta");
    if (meta) {
      meta.setAttribute("role", "status");
      meta.setAttribute("aria-live", "polite");
      meta.setAttribute("aria-atomic", "true");
    }
    if (query && result && query.dataset.searchReadiness !== "true") {
      query.dataset.searchReadiness = "true";

      // Root cause recorded 2026-07-25: Material's search worker initializes
      // asynchronously. A value entered or pasted before its subscriber mounts
      // remains visible, but no query event is replayed after initialization.
      // Re-emit the key event Material consumes for paste, autofill, and values
      // entered before the subscriber mounts.
      const replayStrandedQuery = (expectedValue) => {
        if (query.value !== expectedValue) return;
        query.dispatchEvent(new KeyboardEvent("keyup", {
          bubbles: true,
          key: query.value.slice(-1) || "Unidentified",
        }));
      };
      const scheduleReplay = () => {
        const expectedValue = query.value;
        [0, 350, 1000].forEach((delay) => {
          window.setTimeout(() => replayStrandedQuery(expectedValue), delay);
        });
      };
      query.addEventListener("input", scheduleReplay);
      scheduleReplay();
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhance, { once: true });
  } else {
    enhance();
  }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(enhance);
  }
})();
