document.querySelectorAll('[data-auto-submit]').forEach(select => {
  select.addEventListener('change', () => select.form.requestSubmit());
});
document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', () => {
    const button = form.querySelector('button[type="submit"]');
    if (button) { button.disabled = true; button.setAttribute('aria-busy', 'true'); }
  });
});
window.addEventListener('pageshow', () => {
  document.querySelectorAll('button[aria-busy="true"]').forEach(button => {
    button.disabled = false; button.removeAttribute('aria-busy');
  });
});
document.querySelectorAll('.crest img').forEach(img => {
  const fallback = () => {
    if (!img.isConnected) return;
    const badge = document.createElement('span');
    badge.className = 'crest-fallback';
    badge.setAttribute('aria-hidden', 'true');
    badge.textContent = img.closest('.team').children[1].textContent.trim().slice(0, 2).toUpperCase();
    img.replaceWith(badge);
  };
  img.addEventListener('error', fallback, { once: true });
  if (img.complete && img.naturalWidth === 0) fallback();
});


