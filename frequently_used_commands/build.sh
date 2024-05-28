#!/bin/zsh

bazel build --subcommands --define MEDIAPIPE_DISABLE_GPU=1 $1

