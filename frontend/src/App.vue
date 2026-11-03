<template>
  <router-view />
</template>

<style>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 16px;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-subtle);
  flex-shrink: 0;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
  min-width: 0;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  margin-left: 16px;
  white-space: nowrap;
}

.logo {
  font-weight: 700;
  font-size: 16px;
  color: var(--accent-primary);
  letter-spacing: -0.5px;
  white-space: nowrap;
  cursor: pointer;
}
.logo:hover {
  color: var(--accent-primary-hover);
}
.notebook-name {
  font-size: 14px;
  color: var(--text-secondary);
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 4px;
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.notebook-name:hover {
  background: var(--bg-hover);
}
.name-input {
  font-size: 14px;
  background: var(--bg-input);
  border: 1px solid var(--accent-primary);
  color: var(--text-primary);
  padding: 2px 6px;
  border-radius: 4px;
  outline: none;
}

.connection {
  width: 8px;
  height: 8px;
  flex-shrink: 0;
  border-radius: 50%;
  border: 1.5px solid var(--text-muted);
}
.connection.connected {
  background: var(--accent-success);
  border-color: var(--accent-success);
}
.connection.connecting {
  border-style: dashed;
}

.error-banner {
  background: var(--tint-danger);
  border-bottom: 1px solid var(--accent-danger);
  color: var(--accent-danger);
  padding: 8px 16px;
  font-size: 13px;
  display: flex;
  align-items: center;
}

.btn {
  background: var(--accent-primary);
  color: var(--text-on-accent);
  border: none;
  padding: 6px 14px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.btn:hover {
  background: var(--accent-primary-hover);
}
.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.btn-secondary {
  background: var(--bg-input);
  color: var(--text-primary);
}
.btn-secondary:hover {
  background: var(--bg-pressed);
}

.workspace {
  display: flex;
  flex: 1;
  flex-direction: column;
  overflow: hidden;
  min-height: 0;
}
.main {
  display: flex;
  flex: 1;
  overflow: hidden;
  min-height: 0;
}
.cells-panel {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  min-width: 0;
}
.dag-drawer-resizer {
  height: 8px;
  flex-shrink: 0;
  cursor: row-resize;
  touch-action: none;
  position: relative;
}
.dag-drawer-resizer::before {
  content: '';
  position: absolute;
  left: 16px;
  right: 16px;
  top: 50%;
  height: 1px;
  transform: translateY(-50%);
  background: var(--border-subtle);
}
.dag-drawer-resizer:hover::before {
  background: var(--accent-primary);
}
.dag-drawer {
  flex-shrink: 0;
  height: var(--dag-drawer-height, 320px);
  background: var(--bg-surface);
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.dag-drawer.collapsed {
  height: 32px;
}
.dag-drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 16px;
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 600;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border-subtle);
}
.dag-drawer.collapsed .dag-drawer-header {
  border-bottom: none;
}
.dag-drawer-header:hover {
  color: var(--text-primary);
}
.dag-drawer-hint {
  font-weight: 500;
  font-size: 10px;
  text-transform: none;
  letter-spacing: 0;
  color: var(--text-muted);
}
.dag-drawer-body {
  flex: 1;
  display: flex;
  min-height: 0;
  overflow: hidden;
}
.dag-drawer-graph {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  padding: 8px 8px 12px 16px;
  display: flex;
  flex-direction: column;
}
.dag-drawer-profiling {
  width: 360px;
  flex-shrink: 0;
  border-left: 1px solid var(--border-subtle);
  overflow-y: auto;
  padding: 12px 16px;
}
.sidebar-resizer {
  width: 12px;
  flex-shrink: 0;
  position: relative;
  cursor: col-resize;
  touch-action: none;
}
.sidebar-resizer::before {
  content: '';
  position: absolute;
  top: 16px;
  bottom: 16px;
  left: 50%;
  width: 1px;
  transform: translateX(-50%);
  background: var(--border-subtle);
}
.sidebar-resizer:hover::before {
  background: var(--accent-primary);
}
.sidebar {
  width: var(--sidebar-width);
  flex-shrink: 0;
  min-width: 280px;
  max-width: 560px;
  padding: 16px 16px 16px 4px;
  overflow-y: auto;
}

.add-cell-btn {
  width: 100%;
  padding: 12px;
  background: none;
  border: 1px dashed var(--border);
  border-radius: 8px;
  color: var(--text-muted);
  font-size: 13px;
  cursor: pointer;
  margin-top: 4px;
}
.add-cell-btn:hover {
  border-color: var(--accent-primary);
  color: var(--accent-primary);
}
.add-cell-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--overlay-scrim);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-dialog {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 24px;
  min-width: 400px;
  box-shadow: var(--shadow-modal);
}

.modal-dialog h2 {
  margin-bottom: 16px;
  font-size: 18px;
  color: var(--text-primary);
}

.modal-input {
  width: 100%;
  padding: 8px 12px;
  background: var(--bg-input);
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 14px;
  margin-bottom: 16px;
  box-sizing: border-box;
}

.modal-input:focus {
  outline: none;
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 2px var(--ring-focus);
}

.modal-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

body.resizing-sidebar {
  cursor: col-resize;
  user-select: none;
}

@media (max-width: 980px) {
  .main {
    flex-direction: column;
  }

  .cells-panel {
    padding-bottom: 8px;
  }

  .sidebar-resizer {
    display: none;
  }

  .sidebar {
    width: auto;
    min-width: 0;
    max-width: none;
    padding: 12px 16px 16px;
    border-top: 1px solid var(--border-subtle);
  }
}
</style>
