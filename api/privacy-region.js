const PRIOR_CONSENT_COUNTRIES = new Set([
  'AT',
  'BE',
  'BG',
  'HR',
  'CY',
  'CZ',
  'DE',
  'DK',
  'EE',
  'ES',
  'FI',
  'FR',
  'GB',
  'GR',
  'HU',
  'IE',
  'IS',
  'IT',
  'LI',
  'LT',
  'LU',
  'LV',
  'MT',
  'NL',
  'NO',
  'PL',
  'PT',
  'RO',
  'SE',
  'SI',
  'SK',
]);

export default function handler(request, response) {
  const header = request.headers['x-vercel-ip-country'];
  const country = (Array.isArray(header) ? header[0] : header)?.toUpperCase();

  response.setHeader('Cache-Control', 'private, no-store');
  response.status(200).json({
    // Vercel supplies the country in production. Unknown regions fail safe;
    // this also makes local previews display the banner for visual QA.
    requiresConsent: country ? PRIOR_CONSENT_COUNTRIES.has(country) : true,
  });
}
