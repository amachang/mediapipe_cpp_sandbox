#!/bin/zsh

bazel build --define MEDIAPIPE_DISABLE_GPU=1 $1

