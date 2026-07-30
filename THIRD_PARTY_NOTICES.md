# Third-party notices

This repository's ISC and CC BY 4.0 grants do not replace the licenses below.
The versions listed are the versions locked for the 30 July 2026 release.
The 29-component CycloneDX inventory in `sbom.cdx.json` covers the complete
transitive Python runtime/site-build environment. The separate 41-package CI
audit and validation toolchain is completely and hash-locked in
`requirements-audit.lock`. Its pip and setuptools build tools are first
installed from wheel-only hashes in `requirements-audit-bootstrap.lock`, then
the full toolchain is installed without build isolation. These tools execute
before deployment but are not shipped in the site or represented as part of
the runtime SBOM.

| Component | Use | Version | License |
| --- | --- | --- | --- |
| [MkDocs](https://www.mkdocs.org/) | Static-site generator | 1.6.1 | BSD 2-Clause |
| [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) | Site theme and bundled browser assets | 9.7.6 | MIT |
| [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/) | Markdown build extensions | 11.0.1 | MIT/BSD-derived components |
| [MathJax](https://www.mathjax.org/) | Math rendering, loaded from jsDelivr at runtime only on pages with math | 3.2.2 | Apache License 2.0 |
| [pip-audit](https://github.com/pypa/pip-audit) | CI vulnerability scanner; not a site runtime dependency | 2.10.1 | Apache License 2.0 |
| [cffconvert](https://github.com/citation-file-format/cff-converter-python) | CI citation-metadata validator; not a site runtime dependency | 2.0.0 | Apache License 2.0 |
| [Gitleaks](https://github.com/gitleaks/gitleaks) | CI scanner for release-reachable and current-pull-request history; not a site runtime dependency | 8.30.1 | MIT |
| [Material Design Icons](https://pictogrammers.com/library/mdi/) | Interface icons bundled through Material for MkDocs | theme-bundled | Apache License 2.0 and upstream licenses |
| [Font Awesome Free](https://fontawesome.com/license/free) | GitHub brand icon bundled through Material for MkDocs | theme-bundled | Icons CC BY 4.0; code MIT; fonts OFL 1.1 |

The complete [Apache License 2.0](docs/assets/licenses/Apache-2.0.txt) text is
included for MathJax and the applicable Pictogrammers icon assets. Material for
MkDocs 9.7.6 bundles the Pictogrammers icon notice and the Font Awesome Free
notice; those upstream attributions and license classifications are reproduced
above and retained in the locked theme distribution.

No third-party font files are stored in this repository. The site uses local
system-font fallbacks. Product, project, and standards names are used
descriptively; their owners retain all trademark rights.

## Required license notices

### MkDocs — BSD 2-Clause

Copyright © 2014-present, Tom Christie. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

### Material for MkDocs — MIT

Copyright (c) 2016-2025 Martin Donath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### PyMdown Extensions — MIT

Copyright (c) 2014-2025 Isaac Muse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The complete upstream notices for derived PyMdown components and theme-bundled
icon sets remain available in their linked distributions.

Gitleaks is downloaded only in CI from its versioned GitHub release; the Linux
x64 archive must match SHA-256
`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`
before execution and is not included in the publication artifact. Its
[upstream MIT license](https://github.com/gitleaks/gitleaks/blob/v8.30.1/LICENSE)
is maintained by the Gitleaks project.
