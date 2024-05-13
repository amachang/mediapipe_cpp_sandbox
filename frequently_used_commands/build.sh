#!/bin/zsh

bazel build --config debug --define MEDIAPIPE_DISABLE_GPU=1 $1

