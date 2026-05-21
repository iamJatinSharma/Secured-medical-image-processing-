import asyncio
from typing import Any, Callable, Coroutine


class WebSocketCallback:
    """Callback that broadcasts training metrics via WebSocket.

    Accepts a broadcast function (async) and calls it at the end
    of each epoch with the collected metrics dictionary.
    """

    def __init__(self, broadcast_fn: Callable[[dict], Coroutine[Any, Any, None]]):
        self.broadcast_fn = broadcast_fn
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop for scheduling broadcasts from threads."""
        self._loop = loop

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Called at the end of each training epoch.

        Args:
            epoch: Current epoch number (1-indexed).
            metrics: Dictionary containing training metrics for this epoch,
                     e.g. train_loss, val_loss, train_acc, val_acc, etc.
        """
        payload = {"epoch": epoch, **metrics}
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self.broadcast_fn(payload), self._loop)
        else:
            # Fallback: try running in current event loop
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(self.broadcast_fn(payload), loop=loop)
            except RuntimeError:
                # No running loop — skip broadcast silently
                pass

    def on_training_complete(self, summary: dict):
        """Called when training finishes.

        Args:
            summary: Dictionary with final training summary,
                     e.g. best_val_acc, total_time_seconds.
        """
        payload = {"status": "complete", **summary}
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self.broadcast_fn(payload), self._loop)
        else:
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(self.broadcast_fn(payload), loop=loop)
            except RuntimeError:
                pass
