## Fenced code with a language

```python
def materialize(inputs, transform):
    """The single primitive Strata exposes."""
    provenance = sha256(serialize(inputs) + serialize(transform))
    if cache.has(provenance):
        return cache.get(provenance)
    result = transform.apply(inputs)
    cache.put(provenance, result)
    return result
```

## Fenced code without a language

```
$ strata-notebook --port 8765
Listening on http://0.0.0.0:8765
```

## Inline + fenced together

The runtime injects `Markdown` into every cell namespace:

```python
display(Markdown("# Live!"))
```

So you can produce documentation from data without importing anything.

## Edge cases

A fence containing markdown special chars:

```markdown
# This stays literal

- *no italics here*
- `no code spans`
- [no links](http://example.com)
```

Fenced HTML — should be HTML-escaped, not rendered:

```html
<script>alert("hi")</script>
```
