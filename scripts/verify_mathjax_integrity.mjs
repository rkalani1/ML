import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";

const url =
  "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js";
const expected =
  "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei";
const loader = fs.readFileSync(
  new URL("../docs/javascripts/mathjax.js", import.meta.url),
  "utf8"
);

assert.ok(loader.includes(url), "loader URL differs from the reviewed bundle");
assert.ok(
  loader.includes(`script.integrity = "${expected}"`),
  "loader SRI differs from the reviewed digest"
);

const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
assert.equal(response.status, 200, `MathJax fetch returned ${response.status}`);
const bytes = Buffer.from(await response.arrayBuffer());
const actual = `sha384-${crypto.createHash("sha384").update(bytes).digest("base64")}`;
assert.equal(actual, expected, "fetched MathJax bytes do not match pinned SRI");

console.log(
  `MATHJAX_INTEGRITY_OK version=3.2.2 bytes=${bytes.length} digest=${actual}`
);
