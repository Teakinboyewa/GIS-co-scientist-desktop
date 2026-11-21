import sys
from typing import Dict, Any
class Agent:
    """Base class for all agents."""
    name = "Agent"

    def log(self, ctx, message):
        """Add a log message to the context."""
        line = f"[{self.name}]: {message}"
        ctx.setdefault("logs", []).append(f"[{self.name}]: {message}")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def handle(self, ctx, task):
        """Handle a task. Override this in subclasses."""
        raise NotImplementedError("Subclasses must implement handle()")

    def handle_simple_step(self, ctx, task):
        """Handle a simple pipeline step (direct execution). Override in subclasses for custom behavior."""
        return self.handle(ctx, task)

    def handle_medium_step(self, ctx, task):
        """Handle a medium pipeline step (with validation). Override in subclasses for custom behavior."""
        return self.handle(ctx, task)

    def handle_complex_step(self, ctx, task):
        """Handle a complex pipeline step (with planning + validation). Override in subclasses for custom behavior."""
        return self.handle(ctx, task)
