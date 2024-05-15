#!/bin/zsh

bazel build --config --define MEDIAPIPE_DISABLE_GPU=1 $1

