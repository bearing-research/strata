"""Cell paths moved out of ``CellExecutor``, one path per module.

``CellExecutor``'s dispatch still calls each by its old method. A path takes the session as an
``ExecutorHost`` and a ``PathContext``, never the executor.
"""
