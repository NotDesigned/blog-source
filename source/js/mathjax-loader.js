(function() {
  if (!document.querySelector('.math')) return;
  if (document.querySelector('script[data-mathjax-browser-loader]')) return;

  var script = document.createElement('script');
  script.src = 'https://lib.baomitu.com/mathjax/3.2.2/es5/tex-mml-chtml.js';
  script.async = true;
  script.dataset.mathjaxBrowserLoader = 'true';
  document.head.appendChild(script);
}());
