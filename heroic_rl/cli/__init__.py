"""
This module contains CLI commands, argument and option parsing.
"""


def main_callable():
    from functools import partial
    from .commands import render, resume, serve, simulate, train  # noqa f401

    import click

    @click.group()
    def main():
        pass

    main.add_command(train)
    main.add_command(resume)
    main.add_command(serve)
    main.add_command(render)
    main.add_command(simulate)

    return partial(main, auto_envvar_prefix="HEROIC_RL")


main = main_callable()
