/** Recent notebooks in localStorage (up to 20, newest first). */

import { ref } from 'vue'

const STORAGE_KEY = 'strata:recentNotebooks'
const MAX_ENTRIES = 20

export interface RecentNotebookEntry {
  name: string
  path: string
  lastOpened: number
  sessionId?: string | null
}

export function normalizeRecentNotebookEntries(input: unknown): RecentNotebookEntry[] {
  if (!Array.isArray(input)) return []
  return input
    .filter(
      (entry: any) =>
        typeof entry?.name === 'string' &&
        typeof entry?.path === 'string' &&
        typeof entry?.lastOpened === 'number' &&
        (entry?.sessionId == null || typeof entry.sessionId === 'string'),
    )
    .map((entry: any) => ({
      name: entry.name,
      path: entry.path,
      lastOpened: entry.lastOpened,
      sessionId: typeof entry.sessionId === 'string' ? entry.sessionId : null,
    }))
    .sort((a, b) => b.lastOpened - a.lastOpened)
    .slice(0, MAX_ENTRIES)
}

export function recordRecentNotebookEntries(
  currentEntries: RecentNotebookEntry[],
  name: string,
  path: string,
  sessionId?: string | null,
  now: number = Date.now(),
): RecentNotebookEntry[] {
  const nextEntry: RecentNotebookEntry = {
    name,
    path,
    lastOpened: now,
    sessionId: sessionId ?? null,
  }
  const remaining = currentEntries.filter((entry) => entry.path !== path)
  return [nextEntry, ...remaining].slice(0, MAX_ENTRIES)
}

export function findRecentNotebookBySessionId(
  currentEntries: RecentNotebookEntry[],
  sessionId: string,
): RecentNotebookEntry | null {
  return currentEntries.find((entry) => entry.sessionId === sessionId) ?? null
}

export function removeRecentNotebookEntries(
  currentEntries: RecentNotebookEntry[],
  path: string,
): RecentNotebookEntry[] {
  return currentEntries.filter((entry) => entry.path !== path)
}

const entries = ref<RecentNotebookEntry[]>(load())

function load(): RecentNotebookEntry[] {
  if (typeof localStorage === 'undefined') {
    return []
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    return normalizeRecentNotebookEntries(JSON.parse(raw))
  } catch {
    return []
  }
}

function save() {
  if (typeof localStorage === 'undefined') {
    return
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.value))
  } catch {
    // localStorage full or disabled: keep the in-memory list.
  }
}

export function useRecentNotebooks() {
  function record(name: string, path: string, sessionId?: string | null) {
    entries.value = recordRecentNotebookEntries(entries.value, name, path, sessionId)
    save()
  }

  function remove(path: string) {
    entries.value = removeRecentNotebookEntries(entries.value, path)
    save()
  }

  function clear() {
    entries.value = []
    save()
  }

  function findBySessionId(sessionId: string): RecentNotebookEntry | null {
    return findRecentNotebookBySessionId(entries.value, sessionId)
  }

  return { entries, record, remove, clear, findBySessionId }
}
