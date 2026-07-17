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
