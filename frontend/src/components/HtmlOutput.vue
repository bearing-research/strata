<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useStrata } from '../composables/useStrata'
import { heightTracker, reportedHeight } from '../utils/htmlOutput'

// A page a cell showed (plotly, altair, a styled table), run in a sandbox: an opaque
// origin, so its script reaches nothing of this page. It tells us only its height.
const props = defineProps<{ notebookId: string; cellId: string; artifactUri: string }>()

const frame = ref<HTMLIFrameElement | null>(null)
const height = ref(150)
const src = computed(() =>
  useStrata().cellHtmlOutputUrl(props.notebookId, props.cellId, props.artifactUri),
)

const track = heightTracker()

function onMessage(event: MessageEvent) {
  const reported = reportedHeight(event, frame.value?.contentWindow)
  const next = reported === null ? null : track(reported)
  if (next !== null) height.value = next
}

onMounted(() => window.addEventListener('message', onMessage))
onUnmounted(() => window.removeEventListener('message', onMessage))
</script>

<template>
  <iframe
    ref="frame"
    class="output-html"
    :src="src"
    :style="{ height: `${height}px` }"
    sandbox="allow-scripts allow-popups allow-downloads"
    referrerpolicy="no-referrer"
    loading="lazy"
    title="Cell output"
    data-testid="html-output"
  ></iframe>
</template>

<style scoped>
.output-html {
  display: block;
  width: 100%;
  border: 0;
  background: #fff;
  border-radius: 4px;
}
</style>
