#!/bin/zsh

bazel run --config debug --define MEDIAPIPE_DISABLE_GPU=1 refresh_compile_commands
sed -i -e 's/"-Ibazel-out\/darwin-fastbuild\/bin\/external\/com_github_glog_glog\/_virtual_includes\/glog"/"-Ibazel-out\/darwin-fastbuild\/bin\/external\/com_github_glog_glog\/_virtual_includes\/glog", "-Iexternal\/com_github_glog_glog\/src"/g' compile_commands.json

