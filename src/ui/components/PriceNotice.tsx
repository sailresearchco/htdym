/**
 * First-visit disclaimer: the default chip prices are ballpark, and price
 * drives every performance/$ ranking. Shown once per browser; dismissal is
 * remembered so the tool doesn't nag across reloads.
 */
import { useEffect, useRef, useState } from 'react';

// bump when the copy changes, so browsers that dismissed an older
// wording are shown the new one
const NOTICE_STORAGE_KEY = 'htdym-price-notice-2';

export function PriceNotice() {
  const [open, setOpen] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.localStorage.getItem(NOTICE_STORAGE_KEY) !== 'dismissed';
  });
  const okRef = useRef<HTMLButtonElement>(null);

  const dismiss = () => {
    window.localStorage.setItem(NOTICE_STORAGE_KEY, 'dismissed');
    setOpen(false);
  };

  useEffect(() => {
    if (!open) return;
    okRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dismiss();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  if (!open) return null;
  return (
    <div className="notice-backdrop">
      <div className="notice" role="dialog" aria-modal="true" aria-labelledby="notice-title">
        <h2 className="notice-title" id="notice-title">
          On Prices
        </h2>
        <p>
          The default chip prices are ballpark numbers for illustrative purposes. Your cost of
          compute likely differs.
        </p>
        <p>
          Consider this when interpreting performance/$ results. Input your own prices for accurate
          results.
        </p>
        <button className="notice-ok" type="button" ref={okRef} onClick={dismiss}>
          Got it
        </button>
      </div>
    </div>
  );
}
