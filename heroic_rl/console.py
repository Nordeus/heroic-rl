def run():
    """
    This is entrypoint for CLI script provided by poetry.
    """
    import os
    from .cli import main

    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    main()
