import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/geist/wght.css';
import '@fontsource-variable/geist-mono/wght.css';
import { App } from './ui/App';
import './ui/theme.css';

if (window.matchMedia('(pointer: coarse) and (max-width: 820px)').matches)
  alert('This tool is built for a wide screen, please use a computer.');

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
