import posthog from 'posthog-js';

// Keep these in sync with Sail's shared attribution cookie configuration. This
// site does not stamp attribution itself, but it must preserve attribution set
// on sailresearch.com, docs, or the app while sharing the visitor identity.
const ATTRIBUTION_COOKIE_KEYS = [
  'initial_llm_source',
  'acquisition_channel',
  'initial_referrer',
  'initial_referring_domain',
  'initial_landing_path',
  'initial_utm_source',
  'initial_utm_medium',
  'initial_utm_campaign',
  'initial_utm_content',
  'initial_utm_term',
  'initial_gclid',
  'initial_attribution_at',
  'llm_source',
];

const key = import.meta.env.VITE_POSTHOG_KEY;

if (key) {
  posthog.init(key, {
    // Keep ingestion first-party so analytics still works for visitors whose
    // blockers reject direct requests to PostHog.
    api_host: '/ingest',
    ui_host: 'https://us.posthog.com',

    // Reuse the Sail identity cookie. Visitors already identified in the app
    // will therefore remain identified on this subdomain.
    cross_subdomain_cookie: true,
    persistence: 'localStorage+cookie',
    cookie_persisted_properties: ATTRIBUTION_COOKIE_KEYS,
    person_profiles: 'identified_only',

    capture_pageview: false,
    capture_pageleave: true,
    autocapture: false,

    // The controls contain model configuration, so mask every input value in
    // replay even though the tool has no account or free-form personal data.
    session_recording: { maskAllInputs: true, maskTextSelector: '[data-ph-mask]' },

    loaded: (ph) => {
      ph.capture('$pageview', {
        $current_url: window.location.href,
        page_name: 'How to Deploy Your Model',
        pathname: window.location.pathname,
        surface: 'htdym',
      });
    },
  });
}
