import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Language grammars are deliberately excluded: naming a chunk
          // here overrides the split a dynamic import would otherwise
          // produce, so forcing lang-markdown in would pull its 490 kB back
          // into the initial download that ``useCodemirror`` defers it out
          // of. Rollup chunks them itself — statically imported grammars
          // land with the app, the dynamic one gets its own async chunk.
          if (
            id.includes('/node_modules/@codemirror/lang-') ||
            id.includes('/node_modules/@lezer/')
          ) {
            return undefined
          }
          if (
            id.includes('/node_modules/@codemirror/') ||
            id.includes('/node_modules/codemirror/')
          ) {
            return 'codemirror-vendor'
          }
          if (id.includes('/node_modules/vue/')) {
            return 'vue-vendor'
          }
          return undefined
        },
      },
    },
  },
})
