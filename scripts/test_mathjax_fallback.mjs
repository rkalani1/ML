import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(
  new URL("../docs/javascripts/mathjax.js", import.meta.url),
  "utf8"
);

const notices = [];
let appendedScript = null;
let appendCount = 0;
let navigationCallback = null;
const mathNode = {
  textContent: "\\(a^2+b^2=c^2\\)",
  parentNode: {
    insertBefore(node, reference) {
      assert.equal(reference, mathNode);
      notices.push(node);
    }
  }
};

function element(tag) {
  const listeners = new Map();
  return {
    tagName: tag.toUpperCase(),
    dataset: {},
    className: "",
    attributes: {},
    textContent: "",
    addEventListener(name, callback) {
      listeners.set(name, callback);
    },
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
    dispatch(name) {
      listeners.get(name)?.();
    },
    remove() {
      this.removed = true;
    }
  };
}

globalThis.window = {};
globalThis.document = {
  readyState: "complete",
  head: {
    appendChild(node) {
      appendedScript = node;
      appendCount += 1;
      node.dispatch("error");
    }
  },
  createElement: element,
  querySelector(selector) {
    if (selector === ".arithmatex") return mathNode;
    if (selector === ".mathjax-fallback-notice") return notices[0] ?? null;
    if (selector === "script[data-mathjax-bundle]") {
      return appendedScript?.removed ? null : appendedScript;
    }
    return null;
  },
  addEventListener() {
    throw new Error("DOMContentLoaded listener is not expected for a complete document");
  }
};
globalThis.document$ = {
  subscribe(callback) {
    navigationCallback = callback;
  }
};

vm.runInThisContext(source, { filename: "docs/javascripts/mathjax.js" });

assert.ok(appendedScript, "MathJax script should be requested when math is present");
assert.equal(
  appendedScript.src,
  "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
);
assert.equal(
  appendedScript.integrity,
  "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei"
);
assert.equal(appendedScript.crossOrigin, "anonymous");
assert.equal(appendedScript.referrerPolicy, "no-referrer");
assert.equal(appendedScript.defer, true);
assert.equal(appendedScript.dataset.mathjaxBundle, "3.2.2");
assert.equal(appendedScript.removed, true, "a failed loader must not block navigation");
assert.equal(appendCount, 1);
assert.equal(notices.length, 1, "initial failure must emit one notice");
assert.equal(notices[0].attributes.role, "status");
assert.match(notices[0].textContent, /source notation remains visible/i);
assert.equal(mathNode.textContent, "\\(a^2+b^2=c^2\\)");

notices.length = 0;
assert.ok(navigationCallback, "instant-navigation callback should be registered");
navigationCallback();
assert.equal(appendCount, 1, "known failure must not leave or duplicate a loader");
assert.equal(notices.length, 1, "a new math page must receive its own fallback notice");
assert.equal(notices[0].attributes.role, "status");

console.log(
  "MATHJAX_FALLBACK_OK navigation_notice=1 failed_loader_removed=1 " +
  "source_notation_preserved=1"
);
