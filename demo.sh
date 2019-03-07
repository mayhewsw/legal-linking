
python -m allennlp.service.server_simple \
        --archive-path model/model.tar.gz \
        --predictor legal_predictor \
        --include-package mylib \
        --title "Constitution" \
        --field-name graf \
        --field-name const \
        --static-dir demo_files \
        --port 5988

