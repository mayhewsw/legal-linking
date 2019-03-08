#!/usr/bin/env bash

python -m allennlp.service.server_simple \
        --archive-path model/model.tar.gz \
        --predictor legal_predictor \
        --include-package mylib \
        --title "Constitution" \
        --field-name graf \
        --static-dir demo_files \
        --port 3283



