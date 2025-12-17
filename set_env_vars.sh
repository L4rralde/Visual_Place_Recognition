#!/usr/bin/env bash

VPR_GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
  || { echo "Not inside a git repository"; return 1; }

export VPR_GIT_ROOT