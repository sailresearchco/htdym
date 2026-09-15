export const ANALYTICS_CONSENT_COOKIE = 'sail_analytics_consent';

export type AnalyticsConsent = 'granted' | 'denied';

const CONSENT_MAX_AGE_SECONDS = 60 * 60 * 24 * 180;

export function readAnalyticsConsent(): AnalyticsConsent | null {
  const prefix = `${ANALYTICS_CONSENT_COOKIE}=`;
  const raw = document.cookie
    .split(';')
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix))
    ?.slice(prefix.length);

  return raw === 'granted' || raw === 'denied' ? raw : null;
}

export function writeAnalyticsConsent(consent: AnalyticsConsent): void {
  const onSailDomain = window.location.hostname.endsWith('sailresearch.com');
  const domain = onSailDomain ? '; domain=.sailresearch.com' : '';
  const secure = window.location.protocol === 'https:' ? '; secure' : '';

  document.cookie =
    `${ANALYTICS_CONSENT_COOKIE}=${consent}; path=/; max-age=${CONSENT_MAX_AGE_SECONDS}` +
    `; samesite=lax${secure}${domain}`;
}

export function globalPrivacyControlEnabled(): boolean {
  return (
    (navigator as Navigator & { globalPrivacyControl?: boolean }).globalPrivacyControl === true
  );
}
