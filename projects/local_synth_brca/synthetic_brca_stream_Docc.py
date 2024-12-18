def stream_labelled_docs(doc_session, doc_stream_cfg):
    print(f"Connected to Doccano as user: {doc_session.username}")

    # iterator
    labelled_samples = doc_session.get_labelled_samples(doc_stream_cfg["PROJECT_ID"])

    # print labelled samples
    for i, (text, labels) in enumerate(labelled_samples, 1):
        print(f"\nSample {i}:")
        print(f"Text: {text[:50]}...")
        print(f"Labels: {labels}")
