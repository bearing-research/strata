/**
 * Theme mode (system / light / dark), resolved against prefers-color-scheme
 * into the <html> data-theme attribute that style.css keys off.
 *
 * Module-scoped so every useTheme() call shares one state.
 */
import { computed, ref, watchEffect } from 'vue'

export type ThemeMode = 'system' | 'light' | 'dark'
export type ResolvedTheme = 'light' | 'dark'

const STORAGE_KEY = 'strata.theme'
const VALID_MODES: readonly ThemeMode[] = ['system', 'light', 'dark'] as const

function loadStoredMode(): ThemeMode {
  if (typeof window === 'undefined') return 'system'
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw && (VALID_MODES as readonly string[]).includes(raw)) {
      return raw as ThemeMode
    }
  } catch {
    // localStorage disabled (private mode, quota): fall through.
  }
  return 'system'
}

function systemPrefersDark(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return true
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

const mode = ref<ThemeMode>(loadStoredMode())
// Mirror of the OS preference so `mode === 'system'` can react live.
const systemDark = ref<boolean>(systemPrefersDark())

if (typeof window !== 'undefined' && window.matchMedia) {
  const mql = window.matchMedia('(prefers-color-scheme: dark)')
  mql.addEventListener('change', (e) => {
    systemDark.value = e.matches
  })
}

const resolved = computed<ResolvedTheme>(() => {
  if (mode.value === 'system') return systemDark.value ? 'dark' : 'light'
  return mode.value
})

// Runs once on registration, so first paint matches.
watchEffect(() => {
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = resolved.value
  }
  if (typeof window !== 'undefined') {
    try {
      window.localStorage.setItem(STORAGE_KEY, mode.value)
    } catch {
      // Persisting is best-effort.
    }
  }
})

function setMode(next: ThemeMode) {
  mode.value = next
}

function cycleMode() {
  // Same order as the UI.
  const order: ThemeMode[] = ['system', 'light', 'dark']
  const idx = order.indexOf(mode.value)
  mode.value = order[(idx + 1) % order.length]
}

export function useTheme() {
  return {
    mode,
    resolved,
    setMode,
    cycleMode,
  }
}
