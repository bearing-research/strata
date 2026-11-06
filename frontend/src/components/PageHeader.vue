<script setup lang="ts">
import { ArrowLeft, Diamond } from '@lucide/vue'
import ThemeToggle from './ThemeToggle.vue'

defineEmits<{ back: [] }>()
</script>

<!-- The header of a page outside a notebook: Back, the home link, the page's breadcrumb (default
slot, as `.crumb` separators and `.page-name` parts) and its links (`actions` slot, as `.nav-link`). -->
<template>
  <header class="page-header" data-testid="page-header">
    <div class="page-title">
      <button
        type="button"
        class="nav-link back-btn"
        data-testid="page-back"
        @click="$emit('back')"
      >
        <ArrowLeft /> Back
      </button>
      <router-link to="/" class="home-link" title="Back to notebooks"
        ><Diamond fill="currentColor" /> strata</router-link
      >
      <span class="crumbs"><slot /></span>
    </div>
    <div class="page-actions">
      <slot name="actions" />
      <ThemeToggle />
    </div>
  </header>
</template>

<style scoped>
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.page-title,
.crumbs {
  display: flex;
  align-items: baseline;
  gap: 8px;
  min-width: 0;
}

/* Only the notebook's name gives way to a narrow window: Back, home, the page's own name and the
links keep their width. */
.back-btn,
.home-link,
.page-actions,
.crumbs :slotted(.crumb),
.crumbs :slotted(.page-name:last-child) {
  flex-shrink: 0;
}

.page-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.home-link {
  font-weight: 700;
  font-size: 20px;
  color: var(--accent-primary);
  text-decoration: none;
}

.crumbs :slotted(.crumb) {
  color: var(--text-muted);
}

.crumbs :slotted(.page-name) {
  font-size: 20px;
  font-weight: 300;
  color: var(--text-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}

.nav-link,
.page-actions :slotted(.nav-link) {
  font-size: 13px;
  color: var(--text-secondary);
  text-decoration: none;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
}
/* .nav-link on a <button> (Back): reset the browser's default chrome. */
.back-btn {
  font-family: inherit;
  line-height: 1.4;
  background: transparent;
  cursor: pointer;
}

.nav-link:hover,
.page-actions :slotted(.nav-link:hover) {
  color: var(--text-primary);
  border-color: var(--border-strong);
}

/* A phone: Back, home and the links on the first row, the breadcrumb on its own row below. */
@media (max-width: 600px) {
  .page-header {
    flex-wrap: wrap;
    justify-content: flex-start;
    gap: 8px;
  }
  .page-title {
    display: contents;
  }
  .page-actions {
    margin-left: auto;
    gap: 8px;
  }
  .crumbs {
    order: 1;
    flex-basis: 100%;
  }
  .crumbs :slotted(.crumb:first-child) {
    display: none;
  }
}
</style>
