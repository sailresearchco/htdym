import { useEffect, useState, type ReactNode } from 'react';
import { disableAnalytics, initializeAnalytics } from './analytics';
import {
  globalPrivacyControlEnabled,
  readAnalyticsConsent,
  type AnalyticsConsent,
} from './privacy-consent';
import { CookieConsent } from './ui/CookieConsent';

export function AnalyticsProvider({ children }: { children: ReactNode }) {
  const [showConsent, setShowConsent] = useState(false);

  useEffect(() => {
    if (!import.meta.env.VITE_POSTHOG_KEY) return;

    if (globalPrivacyControlEnabled()) {
      disableAnalytics();
      return;
    }

    const saved = readAnalyticsConsent();
    if (saved === 'granted') {
      initializeAnalytics();
      return;
    }
    if (saved === 'denied') return;

    const controller = new AbortController();
    fetch('/api/privacy-region', { cache: 'no-store', signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`privacy region returned ${response.status}`);
        return response.json() as Promise<{ requiresConsent: boolean }>;
      })
      .then(({ requiresConsent }) => {
        if (requiresConsent) setShowConsent(true);
        else initializeAnalytics();
      })
      .catch((error: unknown) => {
        if ((error as { name?: string }).name !== 'AbortError') setShowConsent(true);
      });

    return () => controller.abort();
  }, []);

  const chooseConsent = (consent: AnalyticsConsent) => {
    setShowConsent(false);
    if (consent === 'granted') initializeAnalytics();
    else disableAnalytics();
  };

  return (
    <>
      {children}
      <CookieConsent open={showConsent} onChoose={chooseConsent} />
    </>
  );
}
