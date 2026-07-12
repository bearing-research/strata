import { createRouter, createWebHashHistory } from 'vue-router'
import HomePage from './views/HomePage.vue'

const loadNotebookPage = () => import('./views/NotebookPage.vue')
let notebookPagePrefetch: Promise<unknown> | null = null

export function preloadNotebookRoute(): Promise<unknown> {
  notebookPagePrefetch ??= loadNotebookPage()
  return notebookPagePrefetch
}

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/', name: 'home', component: HomePage },
    {
      path: '/notebook/:sessionId',
      name: 'notebook',
      component: loadNotebookPage,
      props: true,
    },
    {
      path: '/app/:sessionId',
      name: 'app',
      component: () => import('./views/AppView.vue'),
      props: true,
    },
    {
      path: '/logs',
      name: 'logs',
      component: () => import('./views/LogsPage.vue'),
    },
    {
      path: '/artifacts',
      name: 'artifacts',
      component: () => import('./views/ArtifactsPage.vue'),
    },
  ],
})

export default router
