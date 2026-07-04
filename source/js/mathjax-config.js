(function() {
  window.MathJax = {
    tex: {
      packages: { '[+]': ['ams'] },
      inlineMath: [['\\(', '\\)'], ['$', '$']],
      displayMath: [['\\[', '\\]'], ['$$', '$$']],
      macros: {
        p: '\\partial',
        R: '\\mathbb{R}',
        E: '\\mathbb{E}',
        avg: ['\\left\\langle #1 \\right\\rangle', 1],
        vareps: '\\varepsilon',
        d: '\\mathrm{d}',
        dd: '\\,\\mathrm{d}',
        tr: '\\mathrm{tr}',
        abs: ['\\left| #1 \\right|', 1],
        norm: ['\\left\\| #1 \\right\\|', 1],
        paren: ['\\left( #1 \\right)', 1],
        bracket: ['\\left[ #1 \\right]', 1],
        braces: ['\\left\\{ #1 \\right\\}', 1],
        set: ['\\left\\{ #1 \\right\\}', 1],
        bm: ['\\begin{bmatrix} #1 \\end{bmatrix}', 1]
      }
    },
    loader: {
      load: ['[tex]/ams', 'ui/menu']
    },
    options: {
      renderActions: {
        insertedScript: [200, function() {
          document.querySelectorAll('mjx-container').forEach(function(node) {
            var target = node.parentNode;
            if (target && target.nodeName.toLowerCase() === 'li') {
              target.parentNode.classList.add('has-jax');
            }
          });
        }, '', false]
      }
    }
  };
}());
