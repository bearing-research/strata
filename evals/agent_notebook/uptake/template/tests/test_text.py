from harbor.text import slugify


def test_slugify_joins_words_with_single_hyphens():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  many   spaces ") == "many-spaces"
