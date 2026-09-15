import { writeAnalyticsConsent, type AnalyticsConsent } from '../privacy-consent';

export function CookieConsent({
  open,
  onChoose,
}: {
  open: boolean;
  onChoose: (consent: AnalyticsConsent) => void;
}) {
  if (!open) return null;

  const choose = (consent: AnalyticsConsent) => {
    writeAnalyticsConsent(consent);
    onChoose(consent);
  };

  return (
    <aside aria-label="Cookie preferences" aria-live="polite" style={styles.container}>
      <p style={styles.copy}>
        We use analytics cookies to understand how people use Sail. You can allow or decline them.
        Read our{' '}
        <a
          href="https://www.sailresearch.com/privacy#information-from-cookies-and-similar-technologies"
          style={styles.link}
        >
          privacy policy
        </a>
        .
      </p>
      <div style={styles.actions}>
        <button type="button" onClick={() => choose('denied')} style={styles.decline}>
          Decline
        </button>
        <button type="button" onClick={() => choose('granted')} style={styles.allow}>
          Allow analytics
        </button>
      </div>
    </aside>
  );
}

const styles = {
  container: {
    position: 'fixed',
    zIndex: 1000,
    left: 20,
    bottom: 20,
    width: 'min(430px, calc(100vw - 40px))',
    padding: 18,
    border: '1px solid #d7dce4',
    borderRadius: 12,
    background: '#fbfcfd',
    color: '#172033',
    boxShadow: '0 12px 36px rgba(22, 32, 55, 0.14)',
    fontFamily: "'Geist', ui-sans-serif, system-ui, sans-serif",
  },
  copy: { margin: 0, fontSize: 14, lineHeight: 1.55, letterSpacing: '-0.01em' },
  link: { color: '#2c4681', textDecoration: 'underline', textUnderlineOffset: 2 },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 16,
  },
  decline: {
    minHeight: 38,
    padding: '8px 15px',
    border: '1px solid #bdc4cf',
    borderRadius: 7,
    background: 'transparent',
    color: '#202a3d',
    font: 'inherit',
    fontSize: 13,
    fontWeight: 550,
    cursor: 'pointer',
  },
  allow: {
    minHeight: 38,
    padding: '8px 15px',
    border: '1px solid #2c4681',
    borderRadius: 7,
    background: '#2c4681',
    color: 'white',
    font: 'inherit',
    fontSize: 13,
    fontWeight: 550,
    cursor: 'pointer',
  },
} as const;
