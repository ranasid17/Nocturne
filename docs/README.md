# Nocturne Docs

This folder is the Jekyll source for the Nocturne GitHub Pages site. The
[overview](index.md) is the current entry point.

With Ruby 3.3 and Bundler installed, run from this directory:

```sh
export BUNDLE_PATH=/tmp/nocturne-docs-bundle
bundle install
JEKYLL_ENV=production bundle exec jekyll build --destination /tmp/nocturne-site
bundle exec jekyll serve --baseurl /Nocturne
```

The local preview is at `http://127.0.0.1:4000/Nocturne/`.
