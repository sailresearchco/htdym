import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // PORT lets the preview harness assign a free port when 5173 is taken.
  server: { port: Number(process.env.PORT) || 5173 },
});
