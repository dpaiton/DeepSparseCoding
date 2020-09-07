#! /bin/sh

download_dir="${1:-./}"
mkdir -p "$download_dir/dsprites"
git clone https://github.com/deepmind/dsprites-dataset.git $download_dir/dsprites
