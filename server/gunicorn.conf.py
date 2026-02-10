# Generated with assistance from Claude (Anthropic) via Claude Code
# https://github.com/anthropics/claude-code
"""Gunicorn configuration for depth estimation server."""

bind = "127.0.0.1:8000"

# Single worker to avoid duplicating GPU model in memory.
# Threads handle concurrent requests sharing the same model.
workers = 1
threads = 4

# Model loading + large-image inference can take a while.
timeout = 120

# Load app in master before forking so the model is loaded once.
preload_app = True

accesslog = "-"
errorlog = "-"
loglevel = "info"
