/**
 * Compile-time assertions about the generated WS payload types.
 *
 * Not a `.test.ts` file: `tsconfig.app.json` excludes those, and `node --test`
 * strips types without checking them, so a narrowing assertion written there
 * verifies nothing — the runtime sees only `message.type === frame`. These live
 * in a checked file, so `vue-tsc -b` (which CI runs via `npm run build`) is what
 * enforces them.
 *
 * Types only, so nothing reaches the bundle.
 */
import type { TypedWsMessage } from './notebook'
import type { ErrorPayload, CellStatusPayload, WsServerPayloadMap } from './ws-payloads.generated'

type Expect<T extends true> = T
type Equal<A, B> =
  (<G>() => G extends A ? 1 : 2) extends <G>() => G extends B ? 1 : 2 ? true : false

/** Narrowing a frame yields *that frame's* payload, not `unknown`. */
export type _PayloadNarrows = Expect<Equal<TypedWsMessage<'error'>['payload'], ErrorPayload>>
export type _PayloadNarrowsPerFrame = Expect<
  Equal<TypedWsMessage<'cell_status'>['payload'], CellStatusPayload>
>

/** The error codes stay a union, so a mistyped branch cannot compile. */
export type _CodeIsAUnion = Expect<
  Equal<
    ErrorPayload['code'],
    'ENVIRONMENT_BUSY' | 'cell_busy' | 'read_only' | 'insufficient_scope' | null | undefined
  >
>

/** A frame with no payload model is not a key of the map. */
export type _UnmodelledFrameIsNotTyped = Expect<
  Equal<'notebook_state' extends keyof WsServerPayloadMap ? true : false, false>
>

/** Fields the wire always carries are not optional. */
export type _AlwaysPresentFieldIsRequired = Expect<Equal<ErrorPayload['error'], string>>
