#! /bin/sh

root_dir="${1:-./}"
mkdir -p "$root_dir/data"
git clone https://github.com/deepmind/dsprites-dataset.git $root_dir/data/dsprites-dataset
cd $root_dir/data/dsprites-dataset
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
